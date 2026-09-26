"""Merge independently executed Gate 1 seed chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_experiment import Config, aggregate, write_csv


def read_predictions(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            for field in ("seed", "episode", "window", "label", "prediction", "correct", "nonempty"):
                row[field] = int(row[field])
            rows.append(row)
    return rows


def read_trace(path):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            row["seed"] = int(row["seed"])
            row["episode"] = int(row["episode"])
            row["soups"] = float(row["soups"])
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--calibration-episodes", type=int, default=120)
    parser.add_argument("--learning-episodes", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--window-episodes", type=int, default=60)
    args = parser.parse_args()
    rows, trace = [], []
    for chunk in args.chunks:
        rows.extend(read_predictions(chunk / "predictions.csv"))
        trace.extend(read_trace(chunk / "learning_trace.csv"))
    seed_ids = sorted({row["seed"] for row in rows})
    config = Config(
        seed_start=min(seed_ids),
        seeds=len(seed_ids),
        calibration_episodes=args.calibration_episodes,
        learning_episodes=args.learning_episodes,
        horizon=args.horizon,
        window_episodes=args.window_episodes,
    )
    curve, summary = aggregate(rows, trace, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "learning_trace.csv", trace)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
