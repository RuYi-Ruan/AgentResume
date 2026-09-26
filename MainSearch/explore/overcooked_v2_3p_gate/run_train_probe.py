"""Tiny official IPPO training-loop cost probe on the adapted 3-player map.

This uses a shared policy and is NOT the independent-actor learnability baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import wandb
from omegaconf import OmegaConf

from jaxmarl.environments.overcooked_v2.layouts import Layout, grounded_coord_ring


HERE = Path(__file__).resolve().parent
JAXMARL_REF = HERE.parent / "benchmark_suitability_smax" / "jaxmarl_ref"
sys.path.insert(0, str(JAXMARL_REF))
from baselines.IPPO.ippo_rnn_overcooked_v2 import make_train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=512)
    args = parser.parse_args()
    if args.timesteps < 256 or args.timesteps % 256:
        parser.error("--timesteps must be a positive multiple of 256")
    config_path = JAXMARL_REF / "baselines" / "IPPO" / "config" / "ippo_rnn_overcooked_v2.yaml"
    config = OmegaConf.to_container(OmegaConf.load(config_path))
    layout_text = grounded_coord_ring.replace("W       W", "W   A   W", 1)
    layout = Layout.from_string(
        layout_text, possible_recipes=[[0, 0, 0], [1, 1, 1]]
    )
    assert len(layout.agent_positions) == 3
    config["ENV_KWARGS"]["layout"] = layout
    config.update(
        {
            "NUM_ENVS": 8,
            "NUM_STEPS": 32,
            "TOTAL_TIMESTEPS": args.timesteps,
            "NUM_MINIBATCHES": 4,
            "UPDATE_EPOCHS": 1,
            "GRU_HIDDEN_DIM": 32,
            "FC_DIM_SIZE": 32,
            "ANNEAL_LR": False,
            "REW_SHAPING_HORIZON": args.timesteps,
        }
    )
    print(f"Beginning {args.timesteps}-step shared-IPPO cost probe (8 environments x 32 steps)", flush=True)
    started = time.perf_counter()
    with wandb.init(mode="disabled"):
        train_jit = jax.jit(make_train(config))
        result = train_jit(jax.random.PRNGKey(0))
        jax.block_until_ready(result["metrics"]["env_step"])
    elapsed = time.perf_counter() - started
    output = {
        "kind": "three_player_shared_ippo_cost_probe",
        "not_an_independent_actor_baseline": True,
        "not_a_learnability_test": True,
        "environment_steps": args.timesteps,
        "parallel_environments": 8,
        "updates": args.timesteps // 256,
        "wall_seconds_including_compile": round(elapsed, 2),
        "jax_devices": [str(device) for device in jax.devices()],
        "metrics": {
            key: [float(v) for v in value]
            for key, value in result["metrics"].items()
        },
    }
    result_path = HERE / "results" / f"train_probe_{args.timesteps}.json"
    result_path.parent.mkdir(exist_ok=True)
    result_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Completed in {elapsed:.2f}s; wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
