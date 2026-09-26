"""Where does OvercookedV2's cost actually sit? Env stepping vs policy network.

The earlier calibration measured the full official pipeline (CNN + GRU + PPO) at ~205
environment steps/s on this CPU. This probe separates the two costs:

  (a) env-only   : vectorized env stepping with random valid actions (no network, no update)
  (b) mlp policy : same rollout length, but a small feed-forward MLP over the flattened ego view
                   (5*5*40 = 1000 floats) plus a real PPO update (2 epochs, 2 minibatches)
  (c) cnn+gru    : for reference, the official-style small pipeline, same rollout settings

All three use the same NUM_ENVS / NUM_STEPS so the numbers are directly comparable. This says
nothing about learnability; it only shows what a cheaper policy would buy on this machine.

Usage:
  python probe_policy_cost.py --num-envs 64 --num-steps 32 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))

import jaxmarl  # noqa: E402
from baselines.IPPO.ippo_rnn_overcooked_v2 import (  # noqa: E402
    ActorCriticRNN,
    ScannedRNN,
)

ENV_KWARGS = dict(
    layout="grounded_coord_simple",
    agent_view_size=2,
    negative_rewards=True,
    sample_recipe_on_delivery=True,
    random_agent_positions=True,
)


class MLPPolicy(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        actor = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        actor = nn.relu(actor)
        actor = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(actor)
        actor = nn.relu(actor)
        logits = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(actor)

        critic = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(x)
        critic = nn.relu(critic)
        critic = nn.Dense(256, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2)))(critic)
        critic = nn.relu(critic)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic)
        return logits, jnp.squeeze(value, axis=-1)


def measure_env_only(num_envs: int, num_steps: int, repeats: int) -> dict:
    env = jaxmarl.make("overcooked_v2", **ENV_KWARGS)
    num_agents = env.num_agents

    def rollout(key):
        obs, env_state = env.reset(key)

        def _step(carry, k):
            env_state, obs = carry
            acts = {a: jax.random.randint(k, (), 0, 6) for a in env.agents}
            obs, env_state, reward, done, info = env.step(k, env_state, acts)
            return (env_state, obs), reward[env.agents[0]]

        (env_state, obs), rewards = jax.lax.scan(
            _step, (env_state, obs), jax.random.split(key, num_steps)
        )
        return rewards.sum()

    fn = jax.jit(jax.vmap(rollout))
    keys = jax.random.split(jax.random.PRNGKey(0), num_envs)
    jax.block_until_ready(fn(keys))
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(keys))
        timings.append(time.perf_counter() - t0)
    per_call = float(np.median(timings))
    env_steps = num_envs * num_steps * num_agents
    return {
        "kind": "env_only",
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_agents": num_agents,
        "seconds_per_call": round(per_call, 4),
        "agent_steps_per_second": round(env_steps / per_call, 1),
        "env_steps_per_second": round(num_envs * num_steps / per_call, 1),
    }


def measure_mlp_policy(num_envs: int, num_steps: int, repeats: int) -> dict:
    env = jaxmarl.make("overcooked_v2", **ENV_KWARGS)
    num_agents = env.num_agents
    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n
    obs_flat = int(np.prod(obs_shape))
    network = MLPPolicy(action_dim=action_dim)
    num_actors = num_envs * num_agents

    def make_state(rng):
        params = network.init(rng, jnp.zeros((num_actors, obs_flat)))
        tx = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(2.5e-4))
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    def train_step(state, env_state, obs, rng):
        def _rollout(carry, _):
            state, env_state, obs, rng = carry
            rng, act_rng = jax.random.split(rng)
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(num_actors, -1)
            logits, value = network.apply(state.params, obs_batch)
            action = jax.random.categorical(act_rng, logits)
            env_act = {
                a: action.reshape(num_agents, num_envs)[i]
                for i, a in enumerate(env.agents)
            }
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng, num_envs), env_state, env_act
            )
            return (state, env_state, new_obs, rng), (
                obs_batch,
                action,
                logits,
                jnp.sum(jnp.stack([reward[a] for a in env.agents]), axis=0),
            )

        (state, env_state, obs, rng), (obs_t, act_t, logits_t, rew_t) = jax.lax.scan(
            _rollout, (state, env_state, obs, rng), None, num_steps
        )

        def loss_fn(params):
            flat_obs = obs_t.reshape(-1, obs_flat)
            flat_act = act_t.reshape(-1)
            flat_rew = rew_t.reshape(num_steps, num_envs, 1)
            flat_rew = jnp.broadcast_to(flat_rew, (num_steps, num_envs, num_agents)).reshape(-1)
            new_logits, new_value = network.apply(params, flat_obs)
            old_log_prob = jax.nn.log_softmax(logits_t.reshape(-1, action_dim), axis=-1)
            old_log_prob = jnp.take_along_axis(
                old_log_prob, flat_act[:, None], axis=-1
            )[:, 0]
            new_log_prob = jnp.take_along_axis(
                jax.nn.log_softmax(new_logits, axis=-1), flat_act[:, None], axis=-1
            )[:, 0]
            ratio = jnp.exp(new_log_prob - old_log_prob)
            adv = flat_rew
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            actor_loss = -jnp.minimum(ratio * adv, jnp.clip(ratio, 0.8, 1.2) * adv).mean()
            value_loss = jnp.square(new_value - flat_rew).mean()
            entropy = -jnp.mean(
                jnp.sum(jax.nn.softmax(new_logits, axis=-1) * jax.nn.log_softmax(new_logits, axis=-1), axis=-1)
            )
            return actor_loss + 0.5 * value_loss - 0.01 * entropy

        for _ in range(2):  # two PPO epochs, as in the scaled config
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
        return state, env_state, obs, rng, loss

    step = jax.jit(train_step)
    rng = jax.random.PRNGKey(0)
    state = jax.jit(make_state)(rng)
    reset_rng = jax.random.split(rng, num_envs)
    obsv, env_state = jax.vmap(env.reset)(reset_rng)
    obs = obsv
    jax.block_until_ready(step(state, env_state, obs, rng))
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        state, env_state, obs, rng, loss = step(state, env_state, obs, rng)
        jax.block_until_ready(loss)
        timings.append(time.perf_counter() - t0)
    per_call = float(np.median(timings))
    env_steps = num_envs * num_steps
    return {
        "kind": "mlp_policy_ppo",
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_agents": num_agents,
        "obs_flat": obs_flat,
        "seconds_per_call": round(per_call, 4),
        "env_steps_per_second": round(env_steps / per_call, 1),
    }


def measure_cnn_gru(num_envs: int, num_steps: int, repeats: int) -> dict:
    env = jaxmarl.make("overcooked_v2", **ENV_KWARGS)
    num_agents = env.num_agents
    obs_shape = env.observation_space().shape
    action_dim = env.action_space(env.agents[0]).n
    config = {"GRU_HIDDEN_DIM": 128, "FC_DIM_SIZE": 128, "ACTIVATION": "relu"}
    network = ActorCriticRNN(action_dim, config=config)
    num_actors = num_envs * num_agents

    def make_state(rng):
        init_x = (
            jnp.zeros((1, num_envs, *obs_shape)),
            jnp.zeros((1, num_envs)),
        )
        hidden = ScannedRNN.initialize_carry(num_envs, config["GRU_HIDDEN_DIM"])
        params = network.init(rng, hidden, init_x)
        tx = optax.chain(optax.clip_by_global_norm(0.25), optax.adam(2.5e-4))
        return TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    def train_step(state, env_state, obs, rng):
        hidden = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])

        def _rollout(carry, _):
            state, env_state, obs, rng, hidden = carry
            rng, act_rng = jax.random.split(rng)
            obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(num_actors, *obs_shape)
            new_hidden, pi, value = network.apply(
                state.params, hidden, (obs_batch[None], jnp.zeros((1, num_actors), dtype=bool))
            )
            action = pi.sample(seed=act_rng)[0]
            env_act = {
                a: action.reshape(num_agents, num_envs)[i] for i, a in enumerate(env.agents)
            }
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng, num_envs), env_state, env_act
            )
            return (state, env_state, new_obs, rng, new_hidden), (
                (obs_batch, action),
                jnp.sum(jnp.stack([reward[a] for a in env.agents]), axis=0),
            )

        (state, env_state, obs, rng, hidden), (acts, rew) = jax.lax.scan(
            _rollout, (state, env_state, obs, rng, hidden), None, num_steps
        )

        def loss_fn(params):
            obs_t, act_t = acts
            hidden0 = ScannedRNN.initialize_carry(num_actors, config["GRU_HIDDEN_DIM"])
            _, pi, value = network.apply(
                params, hidden0, (obs_t, jnp.zeros((num_steps, num_actors), dtype=bool))
            )
            log_prob = pi.log_prob(act_t.squeeze())
            return -log_prob.mean() + 0.5 * jnp.square(value).mean()

        for _ in range(2):
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
        return state, env_state, obs, rng, loss

    step = jax.jit(train_step)
    rng = jax.random.PRNGKey(0)
    state = jax.jit(make_state)(rng)
    obsv, env_state = jax.vmap(env.reset)(jax.random.split(rng, num_envs))
    obs = obsv
    jax.block_until_ready(step(state, env_state, obs, rng))
    timings = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        state, env_state, obs, rng, loss = step(state, env_state, obs, rng)
        jax.block_until_ready(loss)
        timings.append(time.perf_counter() - t0)
    per_call = float(np.median(timings))
    return {
        "kind": "cnn_gru_ppo",
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_agents": num_agents,
        "seconds_per_call": round(per_call, 4),
        "env_steps_per_second": round(num_envs * num_steps / per_call, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    argv = parser.parse_args()

    results = [
        measure_env_only(argv.num_envs, argv.num_steps, argv.repeats),
        measure_mlp_policy(argv.num_envs, argv.num_steps, argv.repeats),
        measure_cnn_gru(argv.num_envs, argv.num_steps, argv.repeats),
    ]
    for record in results:
        print(json.dumps(record), flush=True)
        steps = record["env_steps_per_second"]
        print(f"   -> 20M env steps would take {20e6 / steps / 60:.1f} minutes", flush=True)

    out_path = HERE / "results" / "policy_cost_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[probe] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
