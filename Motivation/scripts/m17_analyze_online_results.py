"""Paired analysis for the M17 fixed-event online score experiment."""
from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np


INPUT = pathlib.Path("data/m17_fixed_online_formal.json")
OUTPUT = pathlib.Path("data/m17_fixed_online_analysis.json")


def exact_sign_p(better, worse):
    non_ties = better + worse
    if non_ties == 0:
        return 1.0
    tail = sum(math.comb(non_ties, k) for k in range(min(better, worse) + 1)) / (2 ** non_ties)
    return min(1.0, 2.0 * tail)


def paired_comparison(left_rows, right_rows, bootstrap_seed=1701, samples=100_000):
    left = {row["event_id"]: row["deliveries"] for row in left_rows}
    right = {row["event_id"]: row["deliveries"] for row in right_rows}
    if set(left) != set(right):
        raise ValueError("paired conditions do not contain identical event IDs")
    ids = sorted(left)
    differences = np.asarray([left[key] - right[key] for key in ids], dtype=np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    means = differences[rng.integers(0, len(differences), size=(samples, len(differences)))].mean(axis=1)
    better = int(np.sum(differences > 0))
    worse = int(np.sum(differences < 0))
    return {
        "paired_events": len(ids),
        "mean_delivery_difference": float(differences.mean()),
        "mean_score_difference": float(20.0 * differences.mean()),
        "bootstrap_95pct_delivery_difference": [
            float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))
        ],
        "better": better,
        "equal": int(np.sum(differences == 0)),
        "worse": worse,
        "exact_two_sided_sign_test_p": exact_sign_p(better, worse),
        "differences": differences.astype(int).tolist(),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    conditions = data["conditions"]
    result = {
        "milestone": "M17 fixed-event online paired analysis",
        "input": str(args.input),
        "comparisons": {
            "new_minus_old": paired_comparison(
                conditions["new"]["per_event"], conditions["old"]["per_event"], 1701
            ),
            "new_minus_none": paired_comparison(
                conditions["new"]["per_event"], conditions["none"]["per_event"], 1702
            ),
            "none_minus_old": paired_comparison(
                conditions["none"]["per_event"], conditions["old"]["per_event"], 1703
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
