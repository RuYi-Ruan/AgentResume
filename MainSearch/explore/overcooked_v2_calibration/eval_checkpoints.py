"""Evaluate every checkpoint of a calibration run with the frozen protocol.

Rationale: the in-run frozen evaluation uses the deterministic (argmax) policy. An
undertrained network can collapse to a constant action under argmax, which hides whatever
partial progress the sampled policy still shows. This script reports both, per checkpoint,
plus the argmax action histogram.

Usage:
  python eval_checkpoints.py --run-dir results/calib2p_seed0_1M_20260924 --samples 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from diagnose_eval import ACTION_NAMES, build_rollout, load_params  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--samples", type=int, default=3)
    argv = parser.parse_args()

    run_dir = Path(argv.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    config = run["config"]
    config["ENV_KWARGS"] = run["env_kwargs"]

    keys = jax.random.split(jax.random.PRNGKey(run["config"].get("EVAL_SEED", 1234)), argv.episodes)
    checkpoints = sorted(run_dir.glob("checkpoint_*.pkl"))
    rollouts = {mode: build_rollout(config, mode) for mode in ("net_argmax", "net_sample")}

    curve = {"run_dir": str(run_dir), "episodes": argv.episodes, "samples": argv.samples, "points": []}
    for path in checkpoints:
        params, meta = load_params(str(path))
        point = {"checkpoint": path.name, "env_steps": meta.get("env_steps"), "eval": {}}
        for mode in ("net_argmax", "net_sample"):
            repeats = argv.samples if mode == "net_sample" else 1
            runs = []
            for repeat in range(repeats):
                repeat_keys = jax.vmap(lambda k: jax.random.fold_in(k, repeat))(keys)
                deliveries, wrong, raw, shaped, hist = jax.device_get(
                    rollouts[mode](params, repeat_keys)
                )
                runs.append(
                    {
                        "correct_deliveries_mean": float(np.asarray(deliveries).mean()),
                        "wrong_deliveries_mean": float(np.asarray(wrong).mean()),
                        "raw_return_mean": float(np.asarray(raw).mean()),
                        "shaped_return_mean": float(np.asarray(shaped).mean()),
                        "episodes_with_delivery": int((np.asarray(deliveries) > 0).sum()),
                        "episodes_with_shaping": int((np.asarray(shaped) > 0).sum()),
                        "action_counts_agent0": np.asarray(hist[0]).sum(axis=0).astype(int).tolist(),
                    }
                )
            point["eval"][mode] = runs[0] if repeats == 1 else {
                "per_repeat": runs,
                "correct_deliveries_mean": float(np.mean([r["correct_deliveries_mean"] for r in runs])),
                "wrong_deliveries_mean": float(np.mean([r["wrong_deliveries_mean"] for r in runs])),
                "raw_return_mean": float(np.mean([r["raw_return_mean"] for r in runs])),
                "shaped_return_mean": float(np.mean([r["shaped_return_mean"] for r in runs])),
                "episodes_with_delivery": int(np.mean([r["episodes_with_delivery"] for r in runs])),
                "episodes_with_shaping": int(np.mean([r["episodes_with_shaping"] for r in runs])),
            }
        curve["points"].append(point)
        argmax = point["eval"]["net_argmax"]
        sampled = point["eval"]["net_sample"]
        print(
            f"{point['env_steps']:>9} steps | argmax deliveries {argmax['correct_deliveries_mean']:.2f} "
            f"shaped {argmax['shaped_return_mean']:.2f} acts {argmax['action_counts_agent0']} | "
            f"sampled deliveries {sampled['correct_deliveries_mean']:.2f} raw {sampled['raw_return_mean']:.2f}",
            flush=True,
        )

    out_path = run_dir / "checkpoint_evals.json"
    out_path.write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    print(f"[eval_checkpoints] wrote {out_path}", flush=True)
    print(f"[eval_checkpoints] action order: {ACTION_NAMES}", flush=True)


if __name__ == "__main__":
    main()
