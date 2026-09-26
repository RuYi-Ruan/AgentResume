"""Merge independently executed seed chunks without recomputing experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_experiment import (
    Config,
    aggregate,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=120)
    args = parser.parse_args()
    rows = []
    for chunk in args.chunks:
        with (chunk / "predictions.csv").open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "window": int(row["window"]),
                        "condition": row["condition"],
                        "pair": row["pair"],
                        "label": int(row["label"]),
                        "prediction": int(row["prediction"]),
                        "correct": int(row["correct"]),
                        "nonempty": int(row["nonempty"]),
                        "seed": int(row["seed"]),
                        "episode": int(row["episode"]),
                    }
                )
    seed_ids = sorted({row["seed"] for row in rows})
    config = Config(
        seed_start=min(seed_ids),
        seeds=len(seed_ids),
        episodes=args.episodes,
        horizon=args.horizon,
    )
    curve, summary = aggregate(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
