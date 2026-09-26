"""Quantify behaviour change across training and difference between agents.

Input: a run directory produced by `train_rware_iippo.py --dump-behaviour`, whose `run.json`
records, per segment and per agent, the action histogram of the deterministic policy on the
fixed evaluation set.

Outputs (printed and written next to run.json as `behaviour_analysis.json`):
  * per-agent action distribution at first / middle / final segment;
  * mean pairwise total-variation distance between agents at each segment (are the four
    policies distinct?);
  * total-variation distance, per agent, between consecutive segments and versus the final
    segment (is each policy still changing, i.e. are partners evolving?).

This is diagnostics for the ETM premise ("partner ability changes during training"), not an
ETM result, and it does not simulate partners by switching checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def tv_distance(p, q) -> float:
    """Total variation distance between two probability vectors."""
    return 0.5 * float(np.abs(np.asarray(p) - np.asarray(q)).sum())


def mean_pairwise(distributions) -> float:
    distances = [
        tv_distance(distributions[i], distributions[j])
        for i in range(len(distributions))
        for j in range(i + 1, len(distributions))
    ]
    return float(np.mean(distances)) if distances else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    argv = parser.parse_args()
    run_dir = Path(argv.run_dir)
    if not run_dir.is_absolute():
        run_dir = HERE / run_dir

    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    segments = [s for s in run["segments"] if "action_distributions" in s]
    if not segments:
        raise SystemExit(f"{run_dir} has no action_distributions; rerun with --dump-behaviour")

    print(f"run: {run_dir.name} | segments with behaviour dump: {len(segments)}")
    print(f"agents: {len(segments[0]['action_distributions'])} | steps: "
          f"{segments[-1]['cumulative_env_steps']:,}")

    first, mid, last = segments[0], segments[len(segments) // 2], segments[-1]
    for label, seg in (("first", first), ("middle", mid), ("final", last)):
        dists = seg["action_distributions"]
        print(f"\n{label} ({seg['cumulative_env_steps']:,} steps), per-agent action distribution:")
        for agent, dist in enumerate(dists):
            dominant = int(np.argmax(dist))
            print(f"  agent {agent}: {[round(x, 3) for x in dist]}  dominant action {dominant}")
        print(f"  mean pairwise TV distance between agents: {mean_pairwise(dists):.3f}")

    print("\nchange over training (per agent):")
    print(f"{'steps':>12} {'meanTV(agents)':>15} {'per-agent TV vs previous':>28} {'per-agent TV vs final':>24}")
    previous = None
    for seg in segments:
        dists = seg["action_distributions"]
        vs_prev = (
            [round(tv_distance(previous[i], dists[i]), 3) for i in range(len(dists))]
            if previous is not None
            else [0.0] * len(dists)
        )
        vs_final = [
            round(tv_distance(last["action_distributions"][i], dists[i]), 3)
            for i in range(len(dists))
        ]
        print(f"{seg['cumulative_env_steps']:>12,} {mean_pairwise(dists):>15.3f} {str(vs_prev):>28} {str(vs_final):>24}")
        previous = dists

    early_vs_final = [
        tv_distance(first["action_distributions"][i], last["action_distributions"][i])
        for i in range(len(last["action_distributions"]))
    ]
    summary = {
        "run_dir": str(run_dir),
        "segments": len(segments),
        "env_steps": last["cumulative_env_steps"],
        "first_action_distributions": first["action_distributions"],
        "final_action_distributions": last["action_distributions"],
        "mean_pairwise_tv_between_agents_final": mean_pairwise(last["action_distributions"]),
        "per_agent_tv_first_vs_final": early_vs_final,
        "mean_tv_consecutive_segments": float(
            np.mean(
                [
                    tv_distance(segments[i]["action_distributions"][a], segments[i + 1]["action_distributions"][a])
                    for i in range(len(segments) - 1)
                    for a in range(len(segments[0]["action_distributions"]))
                ]
            )
        ),
        "note": "deterministic (argmax) action histograms on the fixed 32-episode evaluation set",
    }
    out_path = run_dir / "behaviour_analysis.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nfirst-vs-final TV per agent: {[round(x, 3) for x in early_vs_final]}")
    print(f"mean consecutive-segment TV: {summary['mean_tv_consecutive_segments']:.3f}")
    print(f"[analyse] wrote {out_path}")


if __name__ == "__main__":
    main()
