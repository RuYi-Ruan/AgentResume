"""Bounded three-player Hanabi learnability gate using JaxMARL's IPPO baseline.

This is a shared-parameter baseline, NOT the independent-policy ETM experiment.
The fixed-seed evaluations use greedy actions and do not update policy weights.
"""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(REPO))
BASELINE = REPO / "baselines" / "IPPO" / "ippo_ff_hanabi.py"
spec = importlib.util.spec_from_file_location("jaxmarl_ippo_ff_hanabi", BASELINE)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def evaluate(params, config, n_games, sample_actions):
    env = baseline.jaxmarl.make("hanabi", num_agents=3)
    network = baseline.ActorCritic(env.action_space(env.agents[0]).n, config=config)

    def one_game(key):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)

        def cond(carry):
            _, _, _, done, turns = carry
            return jnp.logical_and(~done, turns < 100)

        def step(carry):
            key, obs, state, _, turns = carry
            key, policy_key, action_key = jax.random.split(key, 3)
            legal = env.get_legal_moves(state)
            obs_batch = jnp.stack([obs[a] for a in env.agents])
            mask_batch = jnp.stack([legal[a] for a in env.agents])
            pi, _ = network.apply(
                params,
                (obs_batch, jnp.zeros((3,), dtype=bool), mask_batch),
            )
            chosen = pi.sample(seed=policy_key) if sample_actions else pi.mode()
            actions = {a: chosen[i] for i, a in enumerate(env.agents)}
            # step() auto-resets on terminal; step_env() preserves terminal score.
            next_obs, next_state, _, dones, _ = env.step_env(action_key, state, actions)
            return key, next_obs, next_state, dones["__all__"], turns + 1

        _, _, state, done, turns = jax.lax.while_loop(
            cond, step, (key, obs, state, jnp.array(False), jnp.array(0))
        )
        return state.score, done, turns

    evaluate_one = jax.jit(one_game)
    records = []
    for seed in range(10_000, 10_000 + n_games):
        score, done, turns = evaluate_one(jax.random.PRNGKey(seed))
        records.append(
            {"seed": seed, "score": int(score), "done": bool(done), "turns": int(turns)}
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--eval-games", type=int, default=32)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--output", type=Path, default=HERE / "shared_gate_seed50")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    config = {
        "LR": 5e-4,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": args.rollout_steps,
        "TOTAL_TIMESTEPS": args.steps,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ENV_NAME": "hanabi",
        "ENV_KWARGS": {"num_agents": 3},
        "ANNEAL_LR": False,
    }
    if args.steps % (args.num_envs * args.rollout_steps):
        raise ValueError("steps must be divisible by num_envs * rollout_steps")
    if (3 * args.num_envs) % config["NUM_MINIBATCHES"]:
        raise ValueError("agent * env count must divide into minibatches")

    metadata = {
        "config": config,
        "seed": args.seed,
        "eval_games": args.eval_games,
        "jax": jax.__version__,
        "devices": [str(x) for x in jax.devices()],
        "baseline": str(BASELINE),
        "shared_parameters": True,
        "evaluation": "same 10000+ seeds, greedy and sampled policies",
        "pid": os.getpid(),
    }
    (args.output / "config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    train_log = args.output / "train_updates.jsonl"
    with train_log.open("w", encoding="utf-8") as log:
        def record(values):
            row = {k: float(v) for k, v in values.items()}
            log.write(json.dumps(row) + "\n")
            log.flush()
            if int(row["env_step"]) % max(1, args.steps // 8) == 0:
                print(f"TRAIN step={int(row['env_step'])} return={row['returns']:.3f}", flush=True)

        baseline.wandb.log = record
        env = baseline.jaxmarl.make("hanabi", num_agents=3)
        net = baseline.ActorCritic(env.action_space(env.agents[0]).n, config=config)
        rng = jax.random.PRNGKey(args.seed)
        _, init_key = jax.random.split(rng)
        init_x = (
            jnp.zeros((1, args.num_envs, env.observation_space(env.agents[0]).shape)),
            jnp.zeros((1, args.num_envs)),
            jnp.zeros((1, args.num_envs, env.action_space(env.agents[0]).n)),
        )
        initial_params = net.init(init_key, init_x)
        t0 = time.perf_counter()
        initial = evaluate(initial_params, config, args.eval_games, False)
        initial_sampled = evaluate(initial_params, config, args.eval_games, True)
        (args.output / "eval_initial_greedy.json").write_text(json.dumps(initial), encoding="utf-8")
        (args.output / "eval_initial_sampled.json").write_text(json.dumps(initial_sampled), encoding="utf-8")
        print(f"INITIAL games={len(initial)} greedy={np.mean([r['score'] for r in initial]):.3f} sampled={np.mean([r['score'] for r in initial_sampled]):.3f}", flush=True)

        trained = jax.jit(baseline.make_train(config))(rng)
        final_params = trained["runner_state"][0][0].params
        jax.block_until_ready(final_params)
        from flax.serialization import to_bytes
        (args.output / "final_params.msgpack").write_bytes(to_bytes(final_params))
        t_train = time.perf_counter() - t0
        print(f"TRAIN_DONE steps={args.steps} elapsed_seconds={t_train:.1f}", flush=True)
        final = evaluate(final_params, config, args.eval_games, False)
        final_sampled = evaluate(final_params, config, args.eval_games, True)
        (args.output / "eval_final_greedy.json").write_text(json.dumps(final), encoding="utf-8")
        (args.output / "eval_final_sampled.json").write_text(json.dumps(final_sampled), encoding="utf-8")
        print(f"FINAL games={len(final)} greedy={np.mean([r['score'] for r in final]):.3f} sampled={np.mean([r['score'] for r in final_sampled]):.3f}", flush=True)
        (args.output / "elapsed.json").write_text(
            json.dumps({"train_and_initial_eval_seconds": t_train, "total_seconds": time.perf_counter()-t0}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
