"""RWARE 4-agent learning gate: IPPO with learned low-level actions (Jumanji RobotWarehouse).

Purpose: establish a reproducible shelves-delivered learning curve on this machine for the
candidate main-experiment environment, before any ETM work. Hyperparameters and the scenario
follow InstaDeep Mava's published RWARE `tiny-4ag` setup (rollout 128, 4 epochs, 2 minibatches,
LR 2.5e-4, clip 0.2, ent 0.01, gamma 0.99, gae 0.95, MLP [128, 128], agent-ID one-hot appended
to each view, time_limit 500), so our curve is comparable with their published reference.

This first pass uses *shared* parameters across the four agents (standard IPPO). The ETM stage
needs four fixed-identity policies with independent parameters; that is a separate variant, and
the only place that has to change is the actor/critic call site.

Frozen evaluation: fixed reset keys, deterministic (argmax) actions, reports delivered shelves
and episode length per episode, and writes an initial-state fingerprint into the run JSON so the
same evaluation set is provably reused.

Usage (absolute interpreter path is required on this machine):
  python train_rware_ippo.py --updates 1000 --segment-updates 50 --num-envs 64
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState
from jumanji.environments.routing.robot_warehouse.env import RobotWarehouse
from jumanji.environments.routing.robot_warehouse.generator import RandomGenerator
from jumanji.wrappers import AutoResetWrapper

HERE = Path(__file__).resolve().parent

# Mava's RWARE `tiny-4ag` scenario config.
TINY_4AG = dict(
    shelf_rows=1,
    shelf_columns=3,
    column_height=8,
    num_agents=4,
    sensor_range=1,
    request_queue_size=4,
)


def make_env(time_limit: int = 500, sensor_range: int = 1) -> AutoResetWrapper:
    scenario = dict(TINY_4AG)
    scenario["sensor_range"] = sensor_range
    env = RobotWarehouse(generator=RandomGenerator(**scenario), time_limit=time_limit)
    return AutoResetWrapper(env, next_obs_in_extras=False)


class ActorCritic(nn.Module):
    """Separate actor and critic MLPs, as in Mava's mlp network config ([128, 128], relu)."""

    action_dim: int

    @nn.compact
    def __call__(self, features):
        actor = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(features)
        actor = nn.relu(actor)
        actor = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)

        critic = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(features)
        critic = nn.relu(critic)
        critic = nn.Dense(128, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


def preprocess(observation, num_agents: int):
    """Agent views as float, with an explicit agent-ID one-hot (Mava's `add_agent_id`)."""
    view = observation.agents_view.astype(jnp.float32)
    ids = jnp.broadcast_to(
        jnp.eye(num_agents, dtype=jnp.float32), view.shape[:-1] + (num_agents,)
    )
    return jnp.concatenate([ids, view], axis=-1), observation.action_mask


def masked_logits(logits, action_mask):
    return jnp.where(action_mask, logits, -1e9)


def log_prob_of(logits, action):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, action[..., None], axis=-1)[..., 0]


def entropy_of(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(jnp.where(probs > 0, probs * log_probs, 0.0), axis=-1)


def build_config(args: argparse.Namespace) -> dict:
    num_agents = TINY_4AG["num_agents"]
    return {
        "num_envs": args.num_envs,
        "rollout_length": args.rollout_length,
        "num_agents": num_agents,
        "action_dim": 5,
        "num_features": 66 + num_agents,
        "lr": 2.5e-4,
        "ppo_epochs": 4,
        "num_minibatches": 2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "time_limit": 500,
        "seed": args.seed,
    }


def make_train(config: dict, run_updates: int, env: AutoResetWrapper):
    num_envs = config["num_envs"]
    num_agents = config["num_agents"]
    num_actors = num_envs * num_agents
    network = ActorCritic(action_dim=config["action_dim"])

    def init_runner_state(rng):
        rng, init_rng, reset_rng = jax.random.split(rng, 3)
        params = network.init(
            init_rng, jnp.zeros((num_actors, config["num_features"]))
        )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            tx=optax.chain(
                optax.clip_by_global_norm(config["max_grad_norm"]),
                optax.adam(config["lr"], eps=1e-5),
            ),
        )
        state, timestep = jax.vmap(env.reset)(jax.random.split(reset_rng, num_envs))
        return (
            train_state,
            state,
            timestep,
            rng,
            jnp.zeros(num_envs),
            jnp.zeros(num_envs, dtype=jnp.int32),
        )

    def policy(train_state, timestep, rng):
        features, action_mask = preprocess(timestep.observation, num_agents)
        obs_flat = features.reshape(num_actors, -1)
        mask_flat = action_mask.reshape(num_actors, -1)
        logits, value = network.apply(train_state.params, obs_flat)
        logits = masked_logits(logits, mask_flat)
        action = jax.random.categorical(rng, logits)
        log_prob = log_prob_of(logits, action)
        return action, log_prob, value, obs_flat, mask_flat

    def rollout(runner_state):
        def _step(carry, _):
            train_state, state, timestep, rng, ep_return, ep_length = carry
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            del step_rng

            action, log_prob, value, obs_flat, mask_flat = policy(
                train_state, timestep, act_rng
            )
            new_state, new_timestep = jax.vmap(env.step)(
                state, action.reshape(num_envs, num_agents)
            )
            last = new_timestep.last()
            reward = new_timestep.reward
            ep_return = ep_return + reward
            ep_length = ep_length + 1

            transition = {
                "obs": obs_flat,
                "mask": mask_flat,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": jnp.repeat(reward, num_agents),
                "done": jnp.repeat(last, num_agents),
            }
            finished_return = jnp.where(last, ep_return, 0.0)
            finished_length = jnp.where(last, ep_length, 0)
            new_carry = (
                train_state,
                new_state,
                new_timestep,
                rng,
                jnp.where(last, 0.0, ep_return),
                jnp.where(last, 0, ep_length),
            )
            return new_carry, (transition, finished_return, finished_length)

        init_carry = runner_state
        carry, (transitions, finished_returns, finished_lengths) = jax.lax.scan(
            _step, init_carry, None, config["rollout_length"]
        )
        train_state, state, timestep, rng, ep_return, ep_length = carry
        _, _, last_value, _, _ = policy(train_state, timestep, jax.random.PRNGKey(0))
        new_runner_state = (train_state, state, timestep, rng, ep_return, ep_length)
        return new_runner_state, transitions, last_value, finished_returns, finished_lengths

    def compute_gae(transitions, last_value):
        def _scan(carry, data):
            gae, next_value = carry
            value = data["value"]
            delta = (
                data["reward"]
                + config["gamma"] * next_value * (1 - data["done"])
                - value
            )
            gae = delta + config["gamma"] * config["gae_lambda"] * (1 - data["done"]) * gae
            return (gae, value), gae

        _, advantages = jax.lax.scan(
            _scan,
            (jnp.zeros_like(last_value), last_value),
            transitions,
            reverse=True,
        )
        return advantages, advantages + transitions["value"]

    def update(train_state, transitions, advantages, returns, rng):
        num_steps = config["rollout_length"]
        batch_size = num_actors * num_steps
        flat = lambda x: x.reshape(batch_size, *x.shape[2:])
        obs_f = flat(transitions["obs"])
        mask_f = flat(transitions["mask"])
        action_f = flat(transitions["action"])
        log_prob_f = flat(transitions["log_prob"])
        value_f = flat(transitions["value"])
        adv_f = advantages.reshape(batch_size)
        ret_f = returns.reshape(batch_size)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        minibatch = batch_size // config["num_minibatches"]

        def _loss(params, mb):
            mb_obs, mb_mask, mb_action, mb_log_prob, mb_adv, mb_ret, mb_value = mb
            logits, new_value = network.apply(params, mb_obs)
            logits = masked_logits(logits, mb_mask)
            new_log_prob = log_prob_of(logits, mb_action)
            ratio = jnp.exp(new_log_prob - mb_log_prob)
            actor_loss = -jnp.minimum(
                ratio * mb_adv,
                jnp.clip(ratio, 1 - config["clip_eps"], 1 + config["clip_eps"]) * mb_adv,
            ).mean()
            value_clipped = mb_value + (new_value - mb_value).clip(
                -config["clip_eps"], config["clip_eps"]
            )
            value_loss = 0.5 * jnp.maximum(
                jnp.square(new_value - mb_ret), jnp.square(value_clipped - mb_ret)
            ).mean()
            entropy = entropy_of(logits).mean()
            total = actor_loss + config["vf_coef"] * value_loss - config["ent_coef"] * entropy
            return total, (actor_loss, value_loss, entropy)

        def _epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            permutation = jax.random.permutation(perm_rng, batch_size)
            idxs = permutation.reshape(config["num_minibatches"], minibatch)

            def _minibatch(carry, idx):
                train_state, _ = carry
                mb = (
                    obs_f[idx],
                    mask_f[idx],
                    action_f[idx],
                    log_prob_f[idx],
                    adv_f[idx],
                    ret_f[idx],
                    value_f[idx],
                )
                (loss, aux), grads = jax.value_and_grad(_loss, has_aux=True)(
                    train_state.params, mb
                )
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, None), {
                    "total": loss,
                    "actor": aux[0],
                    "value": aux[1],
                    "entropy": aux[2],
                }

            (train_state, _), losses = jax.lax.scan(
                _minibatch, (train_state, None), idxs
            )
            return (train_state, rng), losses

        (train_state, rng), losses = jax.lax.scan(
            _epoch, (train_state, rng), None, config["ppo_epochs"]
        )
        return train_state, rng, losses

    def train(rng, runner_state=None):
        if runner_state is None:
            runner_state = init_runner_state(rng)

        def _update_step(runner_state, update_step):
            (
                runner_state,
                transitions,
                last_value,
                finished_returns,
                finished_lengths,
            ) = rollout(runner_state)
            advantages, returns = compute_gae(transitions, last_value)
            train_state, state, timestep, rng, ep_return, ep_length = runner_state
            train_state, rng, losses = update(
                train_state, transitions, advantages, returns, rng
            )
            num_finished = jnp.sum(finished_returns > 0) + jnp.sum(
                (finished_lengths > 0) & (finished_returns == 0)
            )
            metric = {
                "loss_total": losses["total"].mean(),
                "loss_actor": losses["actor"].mean(),
                "loss_value": losses["value"].mean(),
                "entropy": losses["entropy"].mean(),
                "num_finished_episodes": num_finished,
                "finished_return_sum": jnp.sum(finished_returns),
                "finished_return_episodes": jnp.sum(finished_returns > 0),
                "finished_length_sum": jnp.sum(finished_lengths),
                "env_step": (update_step + 1) * config["num_envs"] * config["rollout_length"],
            }
            new_runner_state = (train_state, state, timestep, rng, ep_return, ep_length)
            return new_runner_state, metric

        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(run_updates)
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train, network


def build_eval_fn(config: dict, env: AutoResetWrapper, num_episodes: int, seed: int):
    network = ActorCritic(action_dim=config["action_dim"])
    num_agents = config["num_agents"]
    keys = jax.random.split(jax.random.PRNGKey(seed), num_episodes)

    def _make(sample: bool):
        @jax.jit
        def evaluate(params, keys):
            def _episode(key):
                state, timestep = env.reset(key)

                def _step(carry, t):
                    state, timestep, live, delivered, length = carry
                    features, action_mask = preprocess(timestep.observation, num_agents)
                    logits, _ = network.apply(params, features)
                    logits = masked_logits(logits, action_mask)
                    if sample:
                        action = jax.random.categorical(jax.random.fold_in(key, t), logits)
                    else:
                        action = jnp.argmax(logits, axis=-1)
                    state, timestep = env.step(state, action)
                    delivered = delivered + jnp.where(live, timestep.reward, 0.0)
                    length = length + jnp.where(live, 1, 0)
                    live = live & ~timestep.last()
                    return (state, timestep, live, delivered, length), None

                init = (state, timestep, jnp.array(True), 0.0, jnp.int32(0))
                (_, _, _, delivered, length), _ = jax.lax.scan(
                    _step, init, jnp.arange(config["time_limit"])
                )
                return delivered, length

            return jax.vmap(_episode)(keys)

        return evaluate

    return {
        "greedy": _make(sample=False),
        "sampled": _make(sample=True),
    }, fingerprint_of(env, num_agents), keys


def fingerprint_of(env: AutoResetWrapper, num_agents: int):

    def fingerprint(keys):
        def _first_view(key):
            _, timestep = env.reset(key)
            return timestep.observation.agents_view.astype(jnp.float32)

        views = jax.jit(jax.vmap(_first_view))(keys)
        return {
            "num_episodes": int(np.asarray(views).shape[0]),
            "initial_agents_view_sum": float(np.asarray(views).sum()),
        }

    return fingerprint


def save_checkpoint(path: Path, runner_state, meta: dict) -> None:
    train_state = runner_state[0]
    payload = {
        "params": jax.tree.map(np.asarray, train_state.params),
        "opt_state": jax.tree.map(np.asarray, train_state.opt_state),
        "step": int(train_state.step),
        "meta": meta,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def to_jsonable(value):
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    array = np.asarray(value)
    if array.ndim == 0:
        return int(array) if np.issubdtype(array.dtype, np.integer) else float(array)
    return [float(x) for x in array.ravel()]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--rollout-length", type=int, default=128)
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--segment-updates", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args(argv)
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("rware_ippo_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    env = make_env(config["time_limit"])
    train, _ = make_train(config, args.segment_updates, env)
    train_jit = jax.jit(train)
    evaluate, fingerprint, eval_keys = build_eval_fn(
        config, env, args.eval_episodes, args.eval_seed
    )
    metadata = {
        "kind": "rware_4p_ippo_learning_gate",
        "shared_parameters": True,
        "not_an_etm_result": True,
        "argv": sys.argv[1:],
        "config": config,
        "scenario": TINY_4AG,
        "reference": "Mava RWARE configs (tiny-4ag) with its published RWARE benchmark curves",
        "eval_fingerprint": fingerprint(eval_keys),
        "segments": [],
        "wall_seconds_total": None,
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    rng = jax.random.PRNGKey(args.seed)
    runner_state = None
    num_segments = args.updates // args.segment_updates
    started = time.perf_counter()
    for segment in range(num_segments):
        segment_started = time.perf_counter()
        if runner_state is None:
            result = train_jit(rng)
        else:
            result = train_jit(rng, runner_state)
        jax.block_until_ready(result["metrics"]["env_step"])
        train_seconds = time.perf_counter() - segment_started
        runner_state = result["runner_state"]
        metrics = to_jsonable(jax.device_get(result["metrics"]))

        eval_started = time.perf_counter()
        params = runner_state[0].params
        greedy_delivered, greedy_lengths = jax.device_get(
            evaluate["greedy"](params, eval_keys)
        )
        sampled_delivered, sampled_lengths = jax.device_get(
            evaluate["sampled"](params, eval_keys)
        )
        eval_seconds = time.perf_counter() - eval_started

        finished = float(sum(metrics["num_finished_episodes"]))
        cumulative_env_steps = (
            (segment + 1) * args.segment_updates * args.num_envs * args.rollout_length
        )
        record = {
            "segment": segment + 1,
            "cumulative_env_steps": cumulative_env_steps,
            "segment_env_steps": int(metrics["env_step"][-1]),
            "train_seconds": round(train_seconds, 2),
            "seconds_per_update": round(train_seconds / args.segment_updates, 2),
            "env_steps_per_second": round(
                args.segment_updates
                * args.num_envs
                * args.rollout_length
                / train_seconds,
                1,
            ),
            "eval_seconds": round(eval_seconds, 2),
            "train_episodes_finished": finished,
            "train_delivered_per_episode": (
                float(sum(metrics["finished_return_sum"]) / finished) if finished else 0.0
            ),
            "train_episode_length_mean": (
                float(sum(metrics["finished_length_sum"]) / finished) if finished else 0.0
            ),
            "loss_total": float(np.mean(metrics["loss_total"])),
            "loss_actor": float(np.mean(metrics["loss_actor"])),
            "loss_value": float(np.mean(metrics["loss_value"])),
            "entropy": float(np.mean(metrics["entropy"])),
            "eval_greedy_delivered_mean": float(np.asarray(greedy_delivered).mean()),
            "eval_greedy_delivered_max": float(np.asarray(greedy_delivered).max()),
            "eval_greedy_episode_length_mean": float(np.asarray(greedy_lengths).mean()),
            "eval_sampled_delivered_mean": float(np.asarray(sampled_delivered).mean()),
            "eval_sampled_delivered_max": float(np.asarray(sampled_delivered).max()),
            "eval_sampled_episode_length_mean": float(
                np.asarray(sampled_lengths).mean()
            ),
            "eval_sampled_delivered_per_episode": np.asarray(
                sampled_delivered
            ).tolist(),
            "eval_greedy_delivered_per_episode": np.asarray(
                greedy_delivered
            ).tolist(),
        }
        metadata["segments"].append(record)
        metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
        save_checkpoint(
            out_dir / f"checkpoint_{cumulative_env_steps:010d}.pkl",
            runner_state,
            {"cumulative_env_steps": cumulative_env_steps, "segment": segment + 1},
        )
        (out_dir / "run.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"[segment {segment + 1}/{num_segments}] env_steps={cumulative_env_steps} "
            f"train={train_seconds:.1f}s ({record['env_steps_per_second']:.0f} steps/s) "
            f"episodes={record['train_episodes_finished']:.0f} "
            f"train_delivered/ep={record['train_delivered_per_episode']:.2f} | "
            f"eval_greedy={record['eval_greedy_delivered_mean']:.2f} "
            f"eval_sampled={record['eval_sampled_delivered_mean']:.2f} "
            f"(max {record['eval_sampled_delivered_max']:.0f}) "
            f"len={record['eval_sampled_episode_length_mean']:.1f}",
            flush=True,
        )

    print(
        f"[rware gate] finished {args.updates} updates "
        f"({args.updates * args.num_envs * args.rollout_length} env steps) in "
        f"{metadata['wall_seconds_total']}s; wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
