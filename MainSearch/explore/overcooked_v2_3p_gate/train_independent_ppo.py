"""Exploratory three-independent-policy PPO for the adapted OvercookedV2 map.

Three actors/critics have separate parameters and optimizer moments. No ETM,
hand-written role controller, hidden-state input, or checkpoint switching.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import serialization

import jaxmarl
from identity_obs import add_identity_channels
from jaxmarl.environments.overcooked_v2.layouts import (
    Layout,
    cramped_room_v2,
    grounded_coord_ring,
    test_time_wide,
)


HERE = Path(__file__).resolve().parent
NUM_AGENTS = 3
ACTION_DIM = 6
HORIZON = 200


class ActorCritic(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, obs):
        x = nn.tanh(nn.Dense(self.hidden_size)(obs))
        x = nn.tanh(nn.Dense(self.hidden_size)(x))
        logits = nn.Dense(ACTION_DIM)(x)
        value = nn.Dense(1)(x).squeeze(-1)
        return logits, value


def make_env(layout_name, random_reset=False):
    if layout_name == "ring3":
        layout_text = grounded_coord_ring.replace("W       W", "W   A   W", 1)
    elif layout_name == "cramped3":
        layout_text = cramped_room_v2.replace("W   R", "W A R", 1)
    elif layout_name == "wide3":
        layout_text = test_time_wide.replace("1    1", "1  A 1", 1)
    else:
        raise ValueError(f"Unknown adapted layout: {layout_name}")
    layout = Layout.from_string(
        layout_text,
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    assert len(layout.agent_positions) == NUM_AGENTS
    return jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=HORIZON,
        agent_view_size=2,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
        random_agent_positions=True,
        random_reset=random_reset,
    )


def encode_batch(obs, state):
    encoded = jax.vmap(add_identity_channels)(obs, state)
    obs_dim = encoded.shape[-3] * encoded.shape[-2] * encoded.shape[-1]
    return jnp.transpose(encoded, (1, 0, 2, 3, 4)).reshape(NUM_AGENTS, -1, obs_dim)


def categorical_log_prob(logits, action):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1).squeeze(-1)


def make_train_step(env, model, optimizer, num_envs, rollout_steps, shaping_horizon):
    def apply_all(params, obs):
        return jax.vmap(model.apply)(params, obs)

    def train_step(params, opt_state, obs, state, rng, total_steps):
        def env_step(carry, _):
            obs, state, rng = carry
            local = encode_batch(obs, state)
            logits, values = apply_all(params, local)
            rng, action_key, step_key = jax.random.split(rng, 3)
            actions = jax.random.categorical(action_key, logits, axis=-1)
            log_prob = categorical_log_prob(logits, actions)
            action_dict = {f"agent_{i}": actions[i] for i in range(NUM_AGENTS)}
            step_keys = jax.random.split(step_key, num_envs)
            next_obs, next_state, reward, done, info = jax.vmap(env.step)(
                step_keys, state, action_dict
            )
            raw = jnp.stack([reward[f"agent_{i}"] for i in range(NUM_AGENTS)])
            shaped = jnp.stack(
                [info["shaped_reward"][f"agent_{i}"] for i in range(NUM_AGENTS)]
            )
            factor = jnp.maximum(0.0, 1.0 - total_steps / shaping_horizon)
            combined = raw + factor * shaped
            data = (local, actions, log_prob, values, combined, done["__all__"], raw, shaped)
            return (next_obs, next_state, rng), data

        (obs, state, rng), trajectory = jax.lax.scan(
            env_step, (obs, state, rng), None, rollout_steps
        )
        local, actions, old_log_prob, old_values, rewards, dones, raw, shaped = trajectory
        _, last_values = apply_all(params, encode_batch(obs, state))
        dones_agents = jnp.broadcast_to(dones[:, None, :], rewards.shape)

        def gae_step(carry, transition):
            gae, next_value = carry
            reward, value, done = transition
            delta = reward + 0.99 * next_value * (1.0 - done) - value
            gae = delta + 0.99 * 0.95 * (1.0 - done) * gae
            return (gae, value), gae

        (_, _), advantages = jax.lax.scan(
            gae_step,
            (jnp.zeros_like(last_values), last_values),
            (rewards, old_values, dones_agents),
            reverse=True,
        )
        targets = advantages + old_values
        # For each agent, flatten time and parallel worlds, never combine parameters.
        transpose_flat = lambda x: jnp.swapaxes(x, 0, 1).reshape(NUM_AGENTS, -1)
        batch_obs = jnp.swapaxes(local, 0, 1).reshape(NUM_AGENTS, -1, local.shape[-1])
        batch_actions = transpose_flat(actions)
        batch_old_log_prob = transpose_flat(old_log_prob)
        batch_old_values = transpose_flat(old_values)
        batch_targets = transpose_flat(targets)
        batch_adv = transpose_flat(advantages)
        batch_adv = (batch_adv - batch_adv.mean(1, keepdims=True)) / (
            batch_adv.std(1, keepdims=True) + 1e-8
        )

        def loss_fn(candidate_params):
            logits, values = apply_all(candidate_params, batch_obs)
            log_prob = categorical_log_prob(logits, batch_actions)
            ratio = jnp.exp(log_prob - batch_old_log_prob)
            actor_loss = -jnp.minimum(
                ratio * batch_adv,
                jnp.clip(ratio, 0.8, 1.2) * batch_adv,
            ).mean(axis=1)
            clipped_values = batch_old_values + jnp.clip(
                values - batch_old_values, -0.2, 0.2
            )
            value_loss = 0.5 * jnp.maximum(
                jnp.square(values - batch_targets),
                jnp.square(clipped_values - batch_targets),
            ).mean(axis=1)
            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -(probs * jax.nn.log_softmax(logits, axis=-1)).sum(-1).mean(axis=1)
            loss_each = actor_loss + 0.5 * value_loss - 0.01 * entropy
            return loss_each.sum(), (actor_loss, value_loss, entropy)

        def update_epoch(carry, _):
            params, opt_state = carry
            (loss, aux), gradients = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(gradients, opt_state, params)
            params = optax.apply_updates(params, updates)
            return (params, opt_state), (loss, aux)

        (params, opt_state), (losses, aux) = jax.lax.scan(
            update_epoch, (params, opt_state), None, 4
        )
        metrics = {
            "loss": losses[-1],
            "actor_loss_each": aux[0][-1],
            "value_loss_each": aux[1][-1],
            "entropy_each": aux[2][-1],
            "raw_reward_per_env_step": raw[0].mean(),
            "shaped_reward_per_env_step": shaped.mean(),
            "positive_reward_events": jnp.sum(raw[0] > 0),
            "completed_episodes": jnp.sum(dones),
        }
        return params, opt_state, obs, state, rng, metrics

    return jax.jit(train_step)


def make_eval(env, model):
    def evaluate(params, seeds):
        keys = jax.vmap(jax.random.PRNGKey)(seeds)
        keys, reset_keys = jax.vmap(jax.random.split)(keys)[:, 0], jax.vmap(jax.random.split)(keys)[:, 1]
        obs, state = jax.vmap(env.reset)(reset_keys)

        def step(carry, _):
            obs, state, keys = carry
            local = encode_batch(obs, state)
            logits, _ = jax.vmap(model.apply)(params, local)
            actions = jnp.argmax(logits, axis=-1)
            action_dict = {f"agent_{i}": actions[i] for i in range(NUM_AGENTS)}
            split_keys = jax.vmap(jax.random.split)(keys)
            keys, step_keys = split_keys[:, 0], split_keys[:, 1]
            # step_env intentionally does not auto-reset: one fixed 200-step game.
            obs, state, rewards, _, _ = jax.vmap(env.step_env)(step_keys, state, action_dict)
            return (obs, state, keys), (
                rewards["agent_0"], state.new_correct_delivery
            )

        (_, _, _), (rewards, deliveries) = jax.lax.scan(
            step, (obs, state, keys), None, HORIZON
        )
        return rewards.sum(axis=0), deliveries.sum(axis=0)

    return jax.jit(evaluate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout", choices=("ring3", "cramped3", "wide3"), default="ring3")
    parser.add_argument("--random-reset", action="store_true")
    args = parser.parse_args()
    steps_per_update = args.num_envs * args.rollout_steps
    if args.steps <= 0 or args.steps % steps_per_update:
        parser.error("--steps must be a positive multiple of num-envs × rollout-steps")
    if args.eval_episodes <= 0 or args.eval_every <= 0:
        parser.error("eval-episodes and eval-every must be positive")

    env = make_env(args.layout, args.random_reset)
    obs_dim = 5 * 5 * (env.observation_space().shape[-1] + NUM_AGENTS)
    model = ActorCritic()
    rng = jax.random.PRNGKey(args.seed)
    rng, init_key, reset_key = jax.random.split(rng, 3)
    init_keys = jax.random.split(init_key, NUM_AGENTS)
    params = jax.vmap(model.init, in_axes=(0, None))(
        init_keys, jnp.zeros((args.num_envs, obs_dim), dtype=jnp.float32)
    )
    optimizer = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4))
    opt_state = optimizer.init(params)
    reset_keys = jax.random.split(reset_key, args.num_envs)
    obs, state = jax.vmap(env.reset)(reset_keys)
    update_fn = make_train_step(
        env, model, optimizer, args.num_envs, args.rollout_steps, args.steps
    )
    eval_fn = make_eval(env, model)
    eval_seeds = jnp.arange(10000, 10000 + args.eval_episodes)
    run_dir = HERE / "results" / f"independent_ippo_seed{args.seed}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running",
        "method": "independent_feedforward_ppo_no_etm",
        "map": args.layout,
        "jax_devices": [str(device) for device in jax.devices()],
        "config": vars(args),
        "fixed_eval_seeds": [int(x) for x in eval_seeds],
        "evaluations": [],
        "training": [],
    }
    started = time.perf_counter()

    def save_record():
        (run_dir / "metrics.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def evaluate_and_save(steps):
        returns, soups = eval_fn(params, eval_seeds)
        returns, soups = jax.device_get((returns, soups))
        entry = {
            "steps": steps,
            "games": args.eval_episodes,
            "mean_return": float(returns.mean()),
            "mean_correct_soups": float(soups.mean()),
            "games_with_soup": int((soups > 0).sum()),
            "returns": [float(x) for x in returns],
            "correct_soups": [int(x) for x in soups],
        }
        record["evaluations"].append(entry)
        (run_dir / f"params_{steps}.msgpack").write_bytes(serialization.to_bytes(params))
        save_record()
        print(
            f"eval steps={steps} soups={entry['mean_correct_soups']:.3f}/game "
            f"games_with_soup={entry['games_with_soup']}/{entry['games']} "
            f"return={entry['mean_return']:.2f}", flush=True
        )

    try:
        evaluate_and_save(0)
        updates = args.steps // steps_per_update
        for update in range(updates):
            params, opt_state, obs, state, rng, metrics = update_fn(
                params, opt_state, obs, state, rng, update * steps_per_update
            )
            metrics = jax.device_get(metrics)
            steps = (update + 1) * steps_per_update
            row = {"steps": steps}
            row.update({key: value.tolist() for key, value in metrics.items()})
            record["training"].append(row)
            if steps % args.eval_every == 0 or steps == args.steps:
                evaluate_and_save(steps)
        record["status"] = "completed"
    except Exception as exc:
        record["status"] = "failed"
        record["error"] = repr(exc)
        raise
    finally:
        record["wall_seconds"] = round(time.perf_counter() - started, 2)
        save_record()
        print(f"status={record['status']} output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
