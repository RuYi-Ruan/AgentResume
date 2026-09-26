"""Merge the corrected independently executed Gate 3 seeds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from run_experiment import Config, aggregate, write_csv


ROOT = Path(__file__).parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    integer_keys = {
        "seed", "checkpoint", "eval_offset", "focal", "base_role",
        "selected_role", "oracle_role", "decision_correct", "gradient_updates",
    }
    for row in rows:
        for key in integer_keys & row.keys():
            row[key] = int(row[key])
        if "soups" in row:
            row["soups"] = float(row["soups"])
    return rows


def main():
    rows, updates = [], []
    for seed in range(5):
        seed_dir = ROOT / f"results_v2_seed{seed}"
        rows.extend(read_csv(seed_dir / "evaluations.csv"))
        updates.extend(read_csv(seed_dir / "update_counts.csv"))
    config = Config(seeds=5)
    summary = aggregate(rows, updates, config)
    output_dir = ROOT / "results_v2"
    output_dir.mkdir(exist_ok=True)
    write_csv(output_dir / "evaluations.csv", rows)
    write_csv(output_dir / "update_counts.csv", updates)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
