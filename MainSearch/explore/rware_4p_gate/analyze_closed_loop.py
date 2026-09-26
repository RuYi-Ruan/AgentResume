"""Compare the three closed-loop arms (none / frozen_etm / online_etm) of the RWARE ETM gate.

Reads each arm's `run.json`, aligns on cumulative environment steps and reports:
  * frozen-evaluation deliveries (greedy and sampled) over training,
  * training-side deliveries per episode,
  * per-arm summary: final, mean of the last third, best, and steps to reach thresholds,
  * the two quantities the gate is about: online_etm vs none (does feeding partner predictions
    help at all?) and online_etm vs frozen_etm (does keeping the teammate model up to date help
    more than a stale one?).

Usage:
  python analyze_closed_loop.py --results results --run-prefix rware_etm --suffix seed0_8M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["none", "frozen_etm", "online_etm"]


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def threshold_steps(curve, threshold: float):
    for point in curve:
        if point["eval_sampled_delivered_mean"] >= threshold:
            return point["cumulative_env_steps"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--run-prefix", default="rware_etm")
    parser.add_argument("--suffix", default="seed0_8M")
    argv = parser.parse_args()

    results_dir = Path(argv.results)
    if not results_dir.is_absolute():
        results_dir = HERE / results_dir

    summary = {}
    print(f"{'arm':>12} {'steps':>10} {'final sampled':>14} {'final greedy':>13} "
          f"{'last-third sampled':>19} {'best sampled':>13} {'train deliv/ep final':>21}")
    for arm in ARMS:
        run_dir = results_dir / f"{argv.run_prefix}_{arm}_{argv.suffix}"
        if not run_dir.exists():
            print(f"{arm:>12}  (missing: {run_dir})")
            continue
        run = load(run_dir)
        segments = run["segments"]
        sampled = np.array([s["eval_sampled_delivered_mean"] for s in segments])
        greedy = np.array([s["eval_greedy_delivered_mean"] for s in segments])
        train = np.array([s["train_delivered_per_episode"] for s in segments])
        tail = max(1, len(segments) // 3)
        summary[arm] = {
            "run_dir": str(run_dir),
            "env_steps": segments[-1]["cumulative_env_steps"],
            "wall_seconds": run["wall_seconds_total"],
            "final_sampled": float(sampled[-1]),
            "final_greedy": float(greedy[-1]),
            "last_third_sampled_mean": float(sampled[-tail:].mean()),
            "last_third_greedy_mean": float(greedy[-tail:].mean()),
            "best_sampled": float(sampled.max()),
            "best_greedy": float(greedy.max()),
            "train_delivered_per_episode_final": float(train[-1]),
            "steps_to_5_sampled": threshold_steps(segments, 5.0),
            "steps_to_10_sampled": threshold_steps(segments, 10.0),
            "sampled_curve": sampled.tolist(),
            "greedy_curve": greedy.tolist(),
            "train_curve": train.tolist(),
        }
        print(f"{arm:>12} {segments[-1]['cumulative_env_steps']:>10,} {sampled[-1]:>14.2f} "
              f"{greedy[-1]:>13.2f} {sampled[-tail:].mean():>19.2f} {sampled.max():>13.2f} "
              f"{train[-1]:>21.2f}")

    out_path = results_dir / "closed_loop_comparison.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\n[wrote] {out_path}")
    if len(summary) == 3:
        print("\ngate read-out (frozen evaluation, 32 fixed episodes):")
        print(f"  online_etm - none        : sampled {summary['online_etm']['final_sampled'] - summary['none']['final_sampled']:+.2f} "
              f"(last third {summary['online_etm']['last_third_sampled_mean'] - summary['none']['last_third_sampled_mean']:+.2f})")
        print(f"  online_etm - frozen_etm  : sampled {summary['online_etm']['final_sampled'] - summary['frozen_etm']['final_sampled']:+.2f} "
              f"(last third {summary['online_etm']['last_third_sampled_mean'] - summary['frozen_etm']['last_third_sampled_mean']:+.2f})")
        print(f"  none - frozen_etm        : sampled {summary['none']['final_sampled'] - summary['frozen_etm']['final_sampled']:+.2f} "
              f"(last third {summary['none']['last_third_sampled_mean'] - summary['frozen_etm']['last_third_sampled_mean']:+.2f})")


if __name__ == "__main__":
    main()
