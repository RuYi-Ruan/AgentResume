"""Measure JaxMARL SMAX training throughput on this CPU for small maps.

Why: SMAX is the only JAX cooperative environment already installed here whose observation
space is tiny (no CNN), so it is the cheapest candidate for a 3-agent continuous-learning
main experiment. This script measures steady-state environment steps per second and the
compile cost; it says nothing about learnability.

Usage (absolute interpreter path is required on this machine):
  python smax_throughput_probe.py --maps 3m 2s3z --updates 3
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
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))

from baselines.IPPO.ippo_rnn_smax import make_train  # noqa: E402

CONFIG_PATH = JAXMARL_REF / "baselines" / "IPPO" / "config" / "ippo_rnn_smax.yaml"


def build_config(map_name: str, updates: int, num_envs: int, num_steps: int, seed: int) -> dict:
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_PATH))
    config.update(
        {
            "MAP_NAME": map_name,
            "NUM_ENVS": num_envs,
            "NUM_STEPS": num_steps,
            "TOTAL_TIMESTEPS": updates * num_envs * num_steps,
            "SEED": seed,
            "NUM_SEEDS": 1,
        }
    )
    return config


def measure(map_name: str, updates: int, num_envs: int, num_steps: int, repeats: int) -> dict:
    config = build_config(map_name, updates * (repeats + 1), num_envs, num_steps, seed=0)
    train = make_train(config)
    steps_per_update = config["NUM_ACTORS"] * num_steps  # env steps = actors x rollout length
    train_jit = jax.jit(train)
    timings = []
    with wandb.init(mode="disabled"):
        for call in range(repeats + 1):
            started = time.perf_counter()
            result = train_jit(jax.random.PRNGKey(0))
            jax.block_until_ready(result)
            timings.append(round(time.perf_counter() - started, 3))
    steady = float(np.median(timings[1:])) / config["NUM_UPDATES"] if len(timings) > 1 else timings[0]
    return {
        "map_name": map_name,
        "num_envs": num_envs,
        "num_steps": num_steps,
        "num_agents": config["NUM_ACTORS"] // num_envs,
        "actors": config["NUM_ACTORS"],
        "updates_per_call": config["NUM_UPDATES"] // (repeats + 1),
        "steps_per_call": steps_per_update * (config["NUM_UPDATES"] // (repeats + 1)),
        "call_seconds": timings,
        "seconds_per_update_steady": round(steady, 3),
        "env_steps_per_second_steady": round(steps_per_update / steady, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps", nargs="*", default=["3m", "2s3z"])
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    argv = parser.parse_args()

    records = []
    for map_name in argv.maps:
        record = measure(map_name, argv.updates, argv.num_envs, argv.num_steps, argv.repeats)
        records.append(record)
        print(json.dumps(record), flush=True)

    out_dir = HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "smax_throughput.json"
    out_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"[smax probe] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
