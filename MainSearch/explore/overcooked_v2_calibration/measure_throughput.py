"""Measure CPU throughput of the official CNN/GRU IPPO pipeline for candidate configurations.

Writes results/throughput.json. Numbers are per-config, exclude nothing silently: compile time
and steady-state seconds per update are both recorded. This decides the affordable calibration
budget; it is not a learning result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import numpy as np
import wandb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_calibration import init_runner_state, make_train_fn, parse_args  # noqa: E402

CANDIDATES = {
    "small": ["--num-envs", "16", "--num-steps", "64", "--num-minibatches", "4",
              "--update-epochs", "2", "--gru-dim", "64", "--fc-dim", "64"],
    "official_ratio": ["--num-envs", "32", "--num-steps", "64", "--num-minibatches", "8",
                       "--update-epochs", "4", "--gru-dim", "128", "--fc-dim", "128"],
    "many_envs": ["--num-envs", "64", "--num-steps", "32", "--num-minibatches", "8",
                  "--update-epochs", "2", "--gru-dim", "128", "--fc-dim", "128"],
    "long_rollout": ["--num-envs", "32", "--num-steps", "128", "--num-minibatches", "8",
                     "--update-epochs", "2", "--gru-dim", "128", "--fc-dim", "128"],
    # Paper recipe (30M steps): 256 envs x 256 steps, 64 minibatches, 4 epochs, minibatch 2048.
    # Measured in two halves because a full-size update may not fit in 15 GB of CPU RAM.
    "official_rollout": ["--num-envs", "64", "--num-steps", "256", "--num-minibatches", "64",
                         "--update-epochs", "4", "--gru-dim", "128", "--fc-dim", "128"],
    "official_envs": ["--num-envs", "256", "--num-steps", "32", "--num-minibatches", "64",
                      "--update-epochs", "4", "--gru-dim", "128", "--fc-dim", "128"],
}


def measure(extra_args: list[str], updates: int, repeats: int) -> dict:
    args = parse_args(
        extra_args
        + ["--updates", str(updates * 2), "--segment-updates", str(updates), "--eval-episodes", "1"]
    )
    runner_state, _ = init_runner_state(args, jax.random.PRNGKey(0))
    train, config = make_train_fn(args, run_updates=updates)
    train_jit = jax.jit(train)
    steps_per_update = args.num_envs * args.num_steps
    timings = []
    with wandb.init(mode="disabled"):
        for call in range(repeats):
            started = time.perf_counter()
            result = train_jit(jax.random.PRNGKey(0), runner_state)
            jax.block_until_ready(result["metrics"]["env_step"])
            timings.append(round(time.perf_counter() - started, 3))
            runner_state = result["runner_state"]
    steady = timings[1:] or timings
    seconds_per_update = float(np.median(steady)) / updates
    return {
        "layout": args.layout,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "num_minibatches": args.num_minibatches,
        "update_epochs": args.update_epochs,
        "gru_dim": args.gru_dim,
        "fc_dim": args.fc_dim,
        "actors": config["NUM_ACTORS"],
        "steps_per_update": steps_per_update,
        "updates_per_call": updates,
        "call_seconds": timings,
        "seconds_per_update_steady": round(seconds_per_update, 3),
        "env_steps_per_second_steady": round(steps_per_update / seconds_per_update, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    argv = parser.parse_args()
    results = []
    for name in argv.configs:
        record = measure(CANDIDATES[name], argv.updates, argv.repeats)
        record["name"] = name
        results.append(record)
        print(json.dumps(record), flush=True)
    out_path = HERE / "results" / "throughput.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"[throughput] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
