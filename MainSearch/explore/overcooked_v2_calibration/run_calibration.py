"""Calibrate the official JaxMARL CNN/GRU IPPO pipeline on a native 2-player OvercookedV2 map.

Goal (see ../search_line.md, items 47-49 and ../overcooked_v2_3p_gate/design_direction.md):
check whether the released V2 training pipeline, with learned atomic actions and in-episode
memory, produces a reproducible soup-delivery learning curve on this machine, and record the
real cost (wall time, environment steps, training episodes).

This is a *pipeline calibration*, not an ETM result and not a reproduction of the paper's
30M-step curves. Two players are used only because a native 2-player map with a known
reference implementation is the cheapest way to check the pipeline.

Reference code: ../../benchmark_suitability_smax/jaxmarl_ref (JaxMARL v0.2.0) with two local
patches in baselines/IPPO/ippo_rnn_overcooked_v2.py:
  1. Flax 0.10-compatible `nn.vmap` instead of `jax.vmap` around the CNN.
  2. `train(rng, initial_runner_state=None)` plus `RUN_UPDATES`, so training can be split into
     resumable segments (checkpoints and frozen evaluation are interleaved between segments).
Both patches are recorded as local compatibility changes, not as official reproduction.

Recorded caveats (kept in RESULTS.md, not silently dropped):
  - Episodes auto-reset at `max_steps=400`; frozen evaluation uses the same 400-step horizon on
    a fixed set of initial states (fixed PRNG keys, fingerprinted in the output JSON).
  - `--resume-from` restores parameters, optimizer state and the update counter; environments,
    GRU hidden state and the sampling RNG restart fresh (documented deviation).
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
import wandb
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
if str(JAXMARL_REF) not in sys.path:
    sys.path.insert(0, str(JAXMARL_REF))

import jaxmarl  # noqa: E402
from baselines.IPPO.ippo_rnn_overcooked_v2 import (  # noqa: E402
    ActorCriticRNN,
    ScannedRNN,
    make_train,
)

CONFIG_PATH = JAXMARL_REF / "baselines" / "IPPO" / "config" / "ippo_rnn_overcooked_v2.yaml"
DELIVERY_REWARD = 20.0


def build_config(args: argparse.Namespace, run_updates: int) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH))
    config["ENV_KWARGS"]["layout"] = args.layout
    config.update(
        {
            "NUM_ENVS": args.num_envs,
            "NUM_STEPS": args.num_steps,
            "NUM_MINIBATCHES": args.num_minibatches,
            "UPDATE_EPOCHS": args.update_epochs,
            "GRU_HIDDEN_DIM": args.gru_dim,
            "FC_DIM_SIZE": args.fc_dim,
            "TOTAL_TIMESTEPS": args.updates * args.num_envs * args.num_steps,
            "REW_SHAPING_HORIZON": args.shaping_horizon,
            "SEED": args.seed,
            "RUN_UPDATES": run_updates,
        }
    )
    return config


def make_train_fn(args: argparse.Namespace, run_updates: int):
    config = build_config(args, run_updates)
    train = make_train(config)
    return train, config


def eval_keys_for(args: argparse.Namespace) -> jax.Array:
    return jax.random.split(jax.random.PRNGKey(args.eval_seed), args.eval_episodes)


def build_eval_fn(config: dict, keys: jax.Array):
    """Frozen evaluation: deterministic (argmax) policy, fixed initial states, 400-step horizon."""
    env = jaxmarl.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)
    obs_shape = env.observation_space().shape
    horizon = env.max_steps
    agent_names = list(env.agents)

    @jax.jit
    def evaluate(params, keys):
        def _episode(key):
            obs, state = env.reset(key)
            hidden = ScannedRNN.initialize_carry(
                env.num_agents, config["GRU_HIDDEN_DIM"]
            )
            done = jnp.zeros((env.num_agents,), dtype=bool)

            def _step(carry, t):
                obs, state, hidden, done, live, deliveries, wrong, raw, shaped = carry
                obs_batch = jnp.stack([obs[a] for a in agent_names]).reshape(
                    -1, *obs_shape
                )
                hidden, pi, _ = network.apply(
                    params, hidden, (obs_batch[None], done[None])
                )
                action = pi.mode()[0]
                obs, state, reward, done_new, info = env.step(
                    jax.random.fold_in(key, t),
                    state,
                    {a: action[i] for i, a in enumerate(agent_names)},
                )
                step_raw = reward[agent_names[0]]
                deliveries = deliveries + (state.new_correct_delivery & live).astype(
                    jnp.int32
                )
                wrong = wrong + ((step_raw <= -DELIVERY_REWARD) & live).astype(jnp.int32)
                raw = raw + jnp.where(live, step_raw, 0.0)
                shaped = shaped + jnp.where(
                    live, info["shaped_reward"][agent_names[0]], 0.0
                )
                done_vec = jnp.stack([done_new[a] for a in agent_names])
                live = live & ~done_new["__all__"]
                return (obs, state, hidden, done_vec, live, deliveries, wrong, raw, shaped), None

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
            )
            out, _ = jax.lax.scan(_step, init, jnp.arange(horizon))
            return jnp.stack(
                [out[5].astype(jnp.float32), out[6].astype(jnp.float32), out[7], out[8]]
            )

        return jax.vmap(_episode)(keys)

    def fingerprint(keys) -> dict:
        """Evidence that the frozen evaluation uses the same initial states every time."""
        _, states = jax.jit(jax.vmap(env.reset))(keys)
        return {
            "recipe": np.asarray(states.recipe).tolist(),
            "agent_x": np.asarray(states.agents.pos.x).tolist(),
            "agent_y": np.asarray(states.agents.pos.y).tolist(),
        }

    return evaluate, fingerprint


def init_runner_state(args: argparse.Namespace, rng: jax.Array):
    """Build the reference runner state with zero updates (no wasted training)."""
    _, config = make_train_fn(args, run_updates=0)
    init_jit = jax.jit(make_train(config))
    return init_jit(rng)["runner_state"], config


def plant_checkpoint(runner_state, checkpoint: dict):
    train_state, env_state, obs, done, update_step, hstate, rng = runner_state
    return (
        train_state.replace(
            params=jax.tree.map(jnp.asarray, checkpoint["params"]),
            opt_state=jax.tree.map(jnp.asarray, checkpoint["opt_state"]),
            step=jnp.asarray(checkpoint["step"]),
        ),
        env_state,
        obs,
        done,
        jnp.asarray(checkpoint["update_step"]),
        hstate,
        jnp.asarray(checkpoint["rng"]),
    )


def save_checkpoint(path: Path, runner_state, meta: dict) -> None:
    train_state = runner_state[0]
    payload = {
        "params": jax.tree.map(np.asarray, train_state.params),
        "opt_state": jax.tree.map(np.asarray, train_state.opt_state),
        "step": int(train_state.step),
        "update_step": int(runner_state[4]),
        "hstate": np.asarray(runner_state[5]),
        "rng": np.asarray(runner_state[6]),
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
    if np.issubdtype(array.dtype, np.integer):
        return [int(x) for x in array.ravel()]
    return [float(x) for x in array.ravel()]


def summarize_eval(raw: np.ndarray) -> dict:
    """raw: (num_episodes, 4) -> [correct deliveries, wrong deliveries, raw return, shaped return]."""
    return {
        "correct_deliveries_per_episode": raw[:, 0].tolist(),
        "wrong_deliveries_per_episode": raw[:, 1].tolist(),
        "raw_return_per_episode": raw[:, 2].tolist(),
        "shaped_return_per_episode": raw[:, 3].tolist(),
        "correct_deliveries_mean": float(raw[:, 0].mean()),
        "wrong_deliveries_mean": float(raw[:, 1].mean()),
        "raw_return_mean": float(raw[:, 2].mean()),
        "shaped_return_mean": float(raw[:, 3].mean()),
        "episodes_with_delivery": int((raw[:, 0] > 0).sum()),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", default="grounded_coord_simple")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--gru-dim", type=int, default=128)
    parser.add_argument("--fc-dim", type=int, default=128)
    parser.add_argument("--updates", type=int, default=64, help="total PPO updates for the whole run")
    parser.add_argument("--segment-updates", type=int, default=8, help="updates per jitted segment")
    parser.add_argument("--shaping-horizon", type=float, default=None, help="defaults to half the total env steps, matching the official 1.5e7/3e7 ratio")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume-from", default=None, help="checkpoint .pkl from an earlier run")
    args = parser.parse_args(argv)
    if args.shaping_horizon is None:
        args.shaping_horizon = 0.5 * args.updates * args.num_envs * args.num_steps
    if args.updates % args.segment_updates:
        parser.error("--updates must be divisible by --segment-updates")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    run_name = args.run_name or time.strftime("calib2p_%Y%m%d_%H%M%S")
    out_dir = HERE / "results" / run_name
    out_dir.mkdir(parents=True, exist_ok=False)

    rng = jax.random.PRNGKey(args.seed)
    runner_state, init_config = init_runner_state(args, rng)

    train, config = make_train_fn(args, run_updates=args.segment_updates)
    if config["NUM_ACTORS"] % config["NUM_MINIBATCHES"]:
        raise SystemExit(
            f"NUM_ACTORS={config['NUM_ACTORS']} must be divisible by NUM_MINIBATCHES={config['NUM_MINIBATCHES']}"
        )
    train_jit = jax.jit(train)

    updates_per_segment = config["RUN_UPDATES"]
    steps_per_update = args.num_envs * args.num_steps
    num_agents = config["NUM_ACTORS"] // args.num_envs
    num_segments = args.updates // updates_per_segment
    eval_keys = eval_keys_for(args)
    evaluate, fingerprint = build_eval_fn(config, eval_keys)

    recorded_state = None
    if args.resume_from:
        with Path(args.resume_from).open("rb") as handle:
            recorded_state = pickle.load(handle)
        runner_state = plant_checkpoint(runner_state, recorded_state)

    metadata = {
        "kind": "two_player_overcooked_v2_pipeline_calibration",
        "not_an_etm_result": True,
        "not_official_reproduction": True,
        "argv": sys.argv[1:],
        "config": {k: v for k, v in config.items() if k != "ENV_KWARGS"},
        "env_kwargs": config["ENV_KWARGS"],
        "updates_total": args.updates,
        "updates_per_segment": updates_per_segment,
        "steps_per_update": steps_per_update,
        "wall_seconds_total": None,
        "resumed_from": args.resume_from,
        "eval_fingerprint": fingerprint(eval_keys),
        "jax_devices": [str(d) for d in jax.devices()],
        "reference_script": str(
            JAXMARL_REF / "baselines" / "IPPO" / "ippo_rnn_overcooked_v2.py"
        ),
        "segments": [],
    }
    (out_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(
        f"[calibration] layout={args.layout} envs={args.num_envs} steps={args.num_steps} "
        f"updates={args.updates} segment={updates_per_segment} devices={metadata['jax_devices']}",
        flush=True,
    )

    started = time.perf_counter()
    cumulative_raw_reward = 0.0
    with wandb.init(mode="disabled"):
        for segment in range(num_segments):
            segment_started = time.perf_counter()
            result = train_jit(rng, runner_state)
            jax.block_until_ready(result["metrics"]["env_step"])
            train_seconds = time.perf_counter() - segment_started
            runner_state = result["runner_state"]
            metrics = to_jsonable(jax.device_get(result["metrics"]))

            params = runner_state[0].params
            eval_started = time.perf_counter()
            raw_eval = np.asarray(jax.device_get(evaluate(params, eval_keys)))
            eval_seconds = time.perf_counter() - eval_started

            env_steps = int(metrics["env_step"][-1])
            raw_reward_per_update = np.asarray(metrics["original_reward"], dtype=float)
            raw_reward_sum = float(
                raw_reward_per_update.sum() * config["NUM_ACTORS"] * args.num_steps
            )
            cumulative_raw_reward += raw_reward_sum
            record = {
                "segment": segment + 1,
                "env_steps": env_steps,
                "train_seconds": round(train_seconds, 2),
                "seconds_per_update": round(train_seconds / updates_per_segment, 2),
                "eval_seconds": round(eval_seconds, 2),
                "train_episodes_completed": int(
                    round(
                        sum(metrics["returned_episode"])
                        * config["NUM_ACTORS"]
                        * args.num_steps
                        / num_agents
                    )
                ),
                "train_raw_reward_sum": raw_reward_sum,
                "train_net_deliveries_estimate": raw_reward_sum / DELIVERY_REWARD,
                "train_raw_reward_cumulative": cumulative_raw_reward,
                "train_metrics": metrics,
                "frozen_eval": summarize_eval(raw_eval),
            }
            metadata["segments"].append(record)
            metadata["wall_seconds_total"] = round(time.perf_counter() - started, 2)
            save_checkpoint(
                out_dir / f"checkpoint_{env_steps:09d}.pkl",
                runner_state,
                {"env_steps": env_steps, "segment": segment + 1, "run_name": run_name},
            )
            (out_dir / "run.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"[segment {segment + 1}/{num_segments}] env_steps={env_steps} "
                f"train={train_seconds:.1f}s eval={eval_seconds:.1f}s "
                f"eval_deliveries={record['frozen_eval']['correct_deliveries_mean']:.2f} "
                f"shaped_return={record['frozen_eval']['shaped_return_mean']:.1f}",
                flush=True,
            )

    print(
        f"[calibration] finished {args.updates} updates "
        f"({args.updates * steps_per_update} env steps) in {metadata['wall_seconds_total']}s; "
        f"wrote {out_dir / 'run.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
