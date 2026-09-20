"""Check whether visible M17 histories actually distinguish pre from post Alice.

This is a small logistic-regression baseline implemented with NumPy so the
experiment does not add a scikit-learn dependency.  Paired pre/post records
from the same rollout seed always stay in the same cross-validation fold.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import pathlib

import numpy as np


INPUT = pathlib.Path("data/m17_paired_impressions.json")
OUTPUT = pathlib.Path("data/m17_impression_signal_gate.json")
OUTCOMES = (
    "seen_holding_onion",
    "seen_at_onion",
    "seen_waiting_at_pot",
    "remained_empty_without_fetch",
    "lost_visibility",
    "visible_timeout_no_clue",
    "episode_ended",
)
ACTIONS = ("move", "interact", "turn", "stay")


def features(episode):
    segments = episode["segments"]
    outcomes = {name: 0 for name in OUTCOMES}
    actions = {name: 0 for name in ACTIONS}
    visible_lengths = []
    for segment in segments:
        outcomes[segment["outcome"]] = outcomes.get(segment["outcome"], 0) + 1
        action = segment["frames"][0]["action_result"]["kind"]
        actions[action] = actions.get(action, 0) + 1
        visible_lengths.append(int(segment["visible_steps"]))
    count = max(1, len(segments))
    return np.asarray(
        [
            len(segments),
            sum(visible_lengths) / count,
            *(actions[name] / count for name in ACTIONS),
            *(outcomes[name] / count for name in OUTCOMES),
        ],
        dtype=np.float64,
    )


def paired_features(segment):
    outcome = segment["outcome"]
    first_action = segment["frames"][0]["action_result"]["kind"]
    frames = segment["frames"]
    action_counts = Counter(item["action_result"]["kind"] for item in frames)
    return np.asarray(
        [
            int(segment["visible_steps"]),
            *(1.0 if first_action == name else 0.0 for name in ACTIONS),
            *(action_counts[name] / max(1, len(frames)) for name in ACTIONS),
            *(1.0 if outcome == name else 0.0 for name in OUTCOMES),
        ],
        dtype=np.float64,
    )


def fit_logistic(x, y, steps=3000, learning_rate=0.08, l2=0.01):
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(steps):
        logits = np.clip(x @ weights, -30.0, 30.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = (x.T @ (probabilities - y)) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=pathlib.Path, default=INPUT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--minimum-accuracy", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if "history" in data["pre"]:
        by_type = {
            name: {int(row["seed"]): row for row in data[name]["history"]["per_seed"]}
            for name in ("pre", "post")
        }
        sample_keys = sorted(set(by_type["pre"]) & set(by_type["post"]))
        encode = features
        key_name = "seed"
    else:
        by_type = {
            name: {str(row["event_id"]): row for row in data[name]["segments"]}
            for name in ("pre", "post")
        }
        sample_keys = sorted(set(by_type["pre"]) & set(by_type["post"]))
        encode = paired_features
        key_name = "event_id"
    fold_rows = []
    all_truth = []
    all_prediction = []
    for fold in range(args.folds):
        test_keys = {key for index, key in enumerate(sample_keys) if index % args.folds == fold}
        train_x, train_y, test_x, test_y = [], [], [], []
        for label, name in enumerate(("pre", "post")):
            for key in sample_keys:
                target_x = test_x if key in test_keys else train_x
                target_y = test_y if key in test_keys else train_y
                target_x.append(encode(by_type[name][key]))
                target_y.append(label)
        train_x = np.stack(train_x)
        test_x = np.stack(test_x)
        train_y = np.asarray(train_y, dtype=np.float64)
        test_y = np.asarray(test_y, dtype=np.int64)
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        train_x = np.column_stack((np.ones(len(train_x)), (train_x - mean) / scale))
        test_x = np.column_stack((np.ones(len(test_x)), (test_x - mean) / scale))
        weights = fit_logistic(train_x, train_y)
        prediction = (test_x @ weights >= 0.0).astype(np.int64)
        accuracy = float(np.mean(prediction == test_y))
        fold_rows.append({
            "fold": fold,
            "test_keys": sorted(test_keys),
            "records": len(test_y),
            "accuracy": accuracy,
        })
        all_truth.extend(test_y.tolist())
        all_prediction.extend(prediction.tolist())

    total_accuracy = float(np.mean(np.asarray(all_truth) == np.asarray(all_prediction)))
    passed = total_accuracy >= args.minimum_accuracy
    result = {
        "milestone": "M17 observable-impression separability gate",
        "passed": passed,
        "qwen_calls": 0,
        "method": f"paired-{key_name} cross-validated logistic regression",
        "feature_source": "only recorded visible-segment counts, actions, outcomes and lengths",
        "forbidden_features": ["Alice intent", "Alice target", "checkpoint weights", "out-of-view behavior"],
        "paired_units": len(sample_keys),
        "records": len(all_truth),
        "minimum_accuracy": args.minimum_accuracy,
        "accuracy": total_accuracy,
        "folds": fold_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
