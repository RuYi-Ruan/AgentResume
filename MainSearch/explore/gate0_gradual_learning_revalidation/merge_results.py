"""Merge independently executed Gate 0 seed chunks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from run_experiment import Config, aggregate, write_csv


def read_rows(path, numeric_fields):
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            row = dict(raw)
            for field, caster in numeric_fields.items():
                if row[field] != "":
                    row[field] = caster(row[field])
                else:
                    row[field] = None
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("chunks", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=360)
    parser.add_argument("--horizon", type=int, default=180)
    parser.add_argument("--checkpoint-interval", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=12)
    args = parser.parse_args()
    training, evaluations = [], []
    eval_numeric = {
        "seed": int,
        "checkpoint": int,
        "eval_case": int,
        "learned_soups": float,
        "static_soups": float,
        "teacher_soups": float,
        **{f"agent_{i}_agrees": int for i in range(3)},
        **{f"agent_{i}_hybrid_soups": float for i in range(3)},
    }
    training_numeric = {
        "seed": int,
        "episode": int,
        "soups": float,
        "epsilon": float,
        **{f"agent_{i}_loss": float for i in range(3)},
    }
    for chunk in args.chunks:
        training.extend(read_rows(chunk / "training.csv", training_numeric))
        evaluations.extend(read_rows(chunk / "evaluations.csv", eval_numeric))
    seed_ids = sorted({row["seed"] for row in evaluations})
    config = Config(
        seed_start=min(seed_ids),
        seeds=len(seed_ids),
        episodes=args.episodes,
        horizon=args.horizon,
        checkpoint_interval=args.checkpoint_interval,
        eval_episodes=args.eval_episodes,
    )
    curve, summary = aggregate(evaluations, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training.csv", training)
    write_csv(args.output_dir / "evaluations.csv", evaluations)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
