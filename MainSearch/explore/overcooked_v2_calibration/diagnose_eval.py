"""Diagnose the frozen-evaluation harness and a trained checkpoint (no training).

Answers two questions the calibration curve alone cannot:
  1. Can this evaluation protocol register shaped events / deliveries at all?
     (reference policies: uniform random, constant "stay", fresh-init network)
  2. What is a trained checkpoint actually doing under argmax vs sampling?
     Reported per policy: deliveries, raw/shaped return, action histogram.

Usage:
  python diagnose_eval.py                       # reference policies only
  python diagnose_eval.py --checkpoint results/<run>/checkpoint_000262144.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_calibration import (  # noqa: E402
    DELIVERY_REWARD,
    build_config,
    eval_keys_for,
    parse_args,
)

import jaxmarl  # noqa: E402
from baselines.IPPO.ippo_rnn_overcooked_v2 import ActorCriticRNN, ScannedRNN  # noqa: E402

ACTION_NAMES = ["right", "down", "left", "up", "stay", "interact"]


def build_rollout(config: dict, mode: str):
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    obs_shape = env.observation_space().shape
    horizon = env.max_steps
    agent_names = list(env.agents)
    action_dim = env.action_space(agent_names[0]).n

    @jax.jit
    def rollout(params, keys):
        def _episode(key):
            obs, state = env.reset(key)
            hidden = ScannedRNN.initialize_carry(
                env.num_agents, config["GRU_HIDDEN_DIM"]
            )
            done = jnp.zeros((env.num_agents,), dtype=bool)

            def _step(carry, t):
                obs, state, hidden, done, live, deliveries, wrong, raw, shaped, hist = carry
                step_key = jax.random.fold_in(key, t)
                if mode in ("net_argmax", "net_sample"):
                    obs_batch = jnp.stack([obs[a] for a in agent_names]).reshape(
                        -1, *obs_shape
                    )
                    hidden, pi, _ = network.apply(
                        params, hidden, (obs_batch[None], done[None])
                    )
                    if mode == "net_argmax":
                        action = pi.mode()[0]
                    else:
                        action = pi.sample(seed=step_key)[0]
                elif mode == "uniform":
                    action = jax.random.randint(
                        step_key, (env.num_agents,), 0, action_dim
                    )
                elif mode == "stay":
                    action = jnp.full((env.num_agents,), 4)
                else:
                    raise ValueError(mode)
                obs, state, reward, done_new, info = env.step(
                    step_key, state, {a: action[i] for i, a in enumerate(agent_names)}
                )
                step_raw = reward[agent_names[0]]
                deliveries = deliveries + (state.new_correct_delivery & live).astype(jnp.int32)
                wrong = wrong + ((step_raw <= -DELIVERY_REWARD) & live).astype(jnp.int32)
                raw = raw + jnp.where(live, step_raw, 0.0)
                shaped = shaped + jnp.where(live, info["shaped_reward"][agent_names[0]], 0.0)
                hist = hist + jax.nn.one_hot(action, action_dim).sum(axis=0)
                done_vec = jnp.stack([done_new[a] for a in agent_names])
                live = live & ~done_new["__all__"]
                return (obs, state, hidden, done_vec, live, deliveries, wrong, raw, shaped, hist), None

            init = (
                obs,
                state,
                hidden,
                done,
                jnp.array(True),
                jnp.int32(0),
                jnp.int32(0),
                jnp.float32(0.0),
                jnp.float32(0.0),
                jnp.zeros((env.num_agents, action_dim)),
            )
            out, _ = jax.lax.scan(_step, init, jnp.arange(horizon))
            return out[5], out[6], out[7], out[8], out[9]

        return jax.vmap(_episode)(keys)

    return rollout


def load_params(path: str):
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return jax.tree.map(jnp.asarray, payload["params"]), payload["meta"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--sample-evals", type=int, default=5)
    argv = parser.parse_args()

    args = parse_args(
        [
            "--layout", "grounded_coord_simple",
            "--num-envs", "64", "--num-steps", "32", "--num-minibatches", "8",
            "--update-epochs", "2", "--gru-dim", "128", "--fc-dim", "128",
            "--updates", "512", "--segment-updates", "32",
            "--eval-episodes", str(argv.eval_episodes), "--eval-seed", str(argv.eval_seed),
        ]
    )
    config = build_config(args, run_updates=0)
    keys = eval_keys_for(args)

    if argv.checkpoint:
        params, meta = load_params(argv.checkpoint)
    else:
        import jaxmarl as _jaxmarl

        env = _jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
        network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
        init_x = (jnp.zeros((1, env.num_agents, *env.observation_space().shape)), jnp.zeros((1, env.num_agents)))
        hidden = ScannedRNN.initialize_carry(env.num_agents, config["GRU_HIDDEN_DIM"])
        # ActorCriticRNN.apply expects the full variables dict, exactly as make_train stores it.
        params = network.init(jax.random.PRNGKey(0), hidden, init_x)
        meta = {"note": "fresh randomly initialised network"}

    report = {"checkpoint": argv.checkpoint, "checkpoint_meta": meta, "policies": {}}
    for mode in ["stay", "uniform", "net_argmax", "net_sample"]:
        repeats = argv.sample_evals if mode == "net_sample" else 1
        runs = []
        for repeat in range(repeats):
            rollout = build_rollout(config, mode)
            repeat_keys = jax.vmap(lambda k: jax.random.fold_in(k, repeat))(keys)
            deliveries, wrong, raw, shaped, hist = jax.device_get(
                rollout(jax.tree.map(jnp.asarray, params), repeat_keys)
            )
            runs.append(
                {
                    "correct_deliveries_mean": float(np.asarray(deliveries).mean()),
                    "wrong_deliveries_mean": float(np.asarray(wrong).mean()),
                    "raw_return_mean": float(np.asarray(raw).mean()),
                    "shaped_return_mean": float(np.asarray(shaped).mean()),
                    "episodes_with_shaping": int((np.asarray(shaped) > 0).sum()),
                    "action_counts": np.asarray(hist[0]).sum(axis=0).astype(int).tolist(),
                }
            )
        report["policies"][mode] = runs if repeats > 1 else runs[0]
        summary = runs[0] if repeats == 1 else {
            "correct_deliveries_mean": float(np.mean([r["correct_deliveries_mean"] for r in runs])),
            "raw_return_mean": float(np.mean([r["raw_return_mean"] for r in runs])),
            "shaped_return_mean": float(np.mean([r["shaped_return_mean"] for r in runs])),
        }
        print(mode, json.dumps(summary), flush=True)

    out_path = HERE / "results" / (
        "diagnose_eval.json" if not argv.checkpoint else f"diagnose_eval_{Path(argv.checkpoint).stem}.json"
    )
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[diagnose] wrote {out_path}", flush=True)
    print(f"[diagnose] action order: {ACTION_NAMES}", flush=True)


if __name__ == "__main__":
    main()
