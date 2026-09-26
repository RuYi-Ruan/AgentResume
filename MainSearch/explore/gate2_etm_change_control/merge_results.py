"""Merge independently executed Gate 2 seeds without recomputing trajectories."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_experiment import Config, aggregate, write_csv


ROOT = Path(__file__).parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("seed", "episode", "window", "label", "prediction", "correct", "nonempty", "gradient_updates"):
            if key in row:
                row[key] = int(row[key])
    return rows


def main():
    predictions, traces, updates = [], [], []
    for seed in range(5):
        seed_dir = ROOT / f"results_seed{seed}"
        predictions.extend(read_csv(seed_dir / "predictions.csv"))
        traces.extend(read_csv(seed_dir / "learning_trace.csv"))
        updates.extend(read_csv(seed_dir / "update_counts.csv"))

    config = Config(seeds=5)
    curve, summary = aggregate(predictions, traces, updates, config)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "predictions.csv", predictions)
    write_csv(output_dir / "learning_trace.csv", traces)
    write_csv(output_dir / "update_counts.csv", updates)
    write_csv(output_dir / "learning_curve.csv", curve)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
