"""Paired statistical summary for the frozen M17 fixed-event result."""
from __future__ import annotations

import argparse
from collections import defaultdict
from math import comb
import json
import pathlib

import numpy as np


INPUT = pathlib.Path("data/m17_qwen_formal_test.json")
OUTPUT = pathlib.Path("data/m17_fixed_formal_analysis.json")


def exact_mcnemar_p(stale_only, updated_only):
    discordant = int(stale_only + updated_only)
    if discordant == 0:
        return 1.0
    lower = min(int(stale_only), int(updated_only))
    tail = sum(comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2.0 * tail)


def paired_interval(differences, seed=20260912, samples=20000):
    values = np.asarray(differences, dtype=np.float64)
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    grouped = defaultdict(dict)
    for row in data["rows"]:
        if int(row["repeat"]) != 0:
            continue
        parsed = row.get("response", {}).get("parsed")
        if parsed is None:
            raise ValueError(f"invalid formal response for {row['event_id']} {row['impression']}")
        grouped[row["event_id"]][row["impression"]] = parsed
    if len(grouped) != 80 or any(set(values) != {"pre", "post", "none"} for values in grouped.values()):
        raise ValueError("formal result is not a complete 80-event x 3-condition matrix")

    pairs = []
    for event_id, values in sorted(grouped.items()):
        stale_correct = values["pre"]["intent"] == "FETCH"
        updated_correct = values["post"]["intent"] == "FETCH"
        none_correct = values["none"]["intent"] == "FETCH"
        pairs.append({
            "event_id": event_id,
            "stale_correct": stale_correct,
            "updated_correct": updated_correct,
            "none_correct": none_correct,
            "difference": int(updated_correct) - int(stale_correct),
        })
    stale_only = sum(row["stale_correct"] and not row["updated_correct"] for row in pairs)
    updated_only = sum(row["updated_correct"] and not row["stale_correct"] for row in pairs)
    both = sum(row["stale_correct"] and row["updated_correct"] for row in pairs)
    neither = len(pairs) - stale_only - updated_only - both
    differences = [row["difference"] for row in pairs]
    result = {
        "milestone": "M17 frozen fixed-event paired analysis",
        "source": str(args.input),
        "events": len(pairs),
        "valid_responses": data["metrics"]["valid"],
        "post_alice": {
            "stale_impression_accuracy": sum(row["stale_correct"] for row in pairs) / len(pairs),
            "updated_impression_accuracy": sum(row["updated_correct"] for row in pairs) / len(pairs),
            "no_impression_accuracy": sum(row["none_correct"] for row in pairs) / len(pairs),
            "updated_minus_stale": float(np.mean(differences)),
            "paired_bootstrap_95_percent_interval": paired_interval(differences),
            "paired_table": {
                "both_correct": both,
                "stale_only_correct": stale_only,
                "updated_only_correct": updated_only,
                "neither_correct": neither,
            },
            "exact_mcnemar_two_sided_p": exact_mcnemar_p(stale_only, updated_only),
        },
        "full_factorial": {
            "correct_impression_accuracy": data["metrics"]["full_factorial_correct_impression_accuracy"],
            "mismatched_impression_accuracy": data["metrics"]["full_factorial_mismatched_impression_accuracy"],
        },
        "interpretation_boundary": (
            "This establishes an offline action-to-intent effect. Team-score consequences "
            "still require the separately planned online controller experiment."
        ),
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
