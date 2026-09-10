"""Offline MLP feasibility test for a structured stale/updated impression.

The historical deterministic L0/Lk trajectories are used only as a labeled
dataset.  A conditional MLP predicts Alice's macro intent from observable
context and Bob's believed capability type.  On the same held-out Lk events,
we swap only that type input: L0 is stale and Lk is updated.

This is a representation/causal-control pilot, not yet a claim that Alice's
capability was learned.  A later phase should replace the scripted source
policies with PPO checkpoints.
"""
from __future__ import annotations

import ast
import json
import pathlib
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, "src")
from ocres.mlp import MLPClassifier


ROOT = pathlib.Path("data/twopot_v1")
OUT = pathlib.Path("data/mlp_impression_results.json")
LEVELS = ("L0", "Lk")
INTENTS = ("FETCH", "PLACE", "COOK_START", "GET_DISH", "PICKUP", "DELIVER", "PRE", "HOLD")
HELD = (None, "onion", "dish", "soup")
POT = ("empty", "items1", "items2", "items3", "cooking", "ready")
INTENT_ID = {name: i for i, name in enumerate(INTENTS)}


def seed_of(path):
    return int(re.search(r"seed(\d+)", path.stem).group(1))


def held_value(value):
    return None if value in (None, "None") else str(value)


def context_key(row):
    return str(row["p0"]), str(held_value(row["held0"])), str(row["pot"])


def encode_context(row):
    # Same observable abstraction used by the historical Book baseline:
    # Alice position, held object and both pot states. No intent/action leak.
    x = np.zeros(35 + len(HELD) + 2 * len(POT), dtype=np.float64)
    px, py = (int(v) for v in row["p0"])
    x[py * 7 + px] = 1
    off = 35
    x[off + HELD.index(held_value(row["held0"]))] = 1
    off += len(HELD)
    pots = ast.literal_eval(str(row["pot"]))
    for j, (_, kind) in enumerate(sorted(pots)[:2]):
        x[off + j * len(POT) + POT.index(kind)] = 1
    return x


def load_events(level):
    events = []
    for path in sorted((ROOT / f"level_{level}").glob("ep_seed*.npz"), key=seed_of):
        z = np.load(path, allow_pickle=True)
        prev = None
        for i in range(len(z["t"])):
            intent = str(z["intent0"][i])
            marker = intent, str(z["target0"][i])
            if marker == prev:
                continue
            prev = marker
            row = {k: z[k][i] for k in z.files}
            events.append({"seed": seed_of(path), "level": level, "intent": intent,
                           "key": context_key(row), "x": encode_context(row)})
    return events


def with_impression(event, impression):
    belief = np.array([1.0, 0.0] if impression == "L0" else [0.0, 1.0])
    return np.concatenate([event["x"], belief])


def majority_map(events, level):
    counts = {}
    for event in events:
        if event["level"] == level:
            counts.setdefault(event["key"], Counter())[event["intent"]] += 1
    return {key: count.most_common(1)[0][0] for key, count in counts.items()}


def accuracy(pred, truth):
    return float(np.mean(np.asarray(pred) == np.asarray(truth))) if len(truth) else None


def run():
    events = [event for level in LEVELS for event in load_events(level)]
    seeds = sorted({event["seed"] for event in events})
    rows = []
    for test_seed in seeds:
        train = [event for event in events if event["seed"] != test_seed]
        test_l0 = [event for event in events if event["seed"] == test_seed and event["level"] == "L0"]
        test_lk = [event for event in events if event["seed"] == test_seed and event["level"] == "Lk"]
        x_train = np.stack([with_impression(e, e["level"]) for e in train])
        y_train = np.array([INTENT_ID[e["intent"]] for e in train])
        model = MLPClassifier(x_train.shape[1], len(INTENTS), hidden_dim=32, seed=100 + test_seed)
        model.fit(x_train, y_train)
        state_model = MLPClassifier(train[0]["x"].size, len(INTENTS), hidden_dim=32, seed=200 + test_seed)
        state_model.fit(np.stack([e["x"] for e in train]), y_train)

        truth_l0 = np.array([INTENT_ID[e["intent"]] for e in test_l0])
        truth_lk = np.array([INTENT_ID[e["intent"]] for e in test_lk])
        pred_l0 = model.predict(np.stack([with_impression(e, "L0") for e in test_l0]))
        pred_updated = model.predict(np.stack([with_impression(e, "Lk") for e in test_lk]))
        pred_stale = model.predict(np.stack([with_impression(e, "L0") for e in test_lk]))
        pred_state_only = state_model.predict(np.stack([e["x"] for e in test_lk]))

        maj0, majk = majority_map(train, "L0"), majority_map(train, "Lk")
        flip_idx = [i for i, e in enumerate(test_lk)
                    if e["key"] in maj0 and e["key"] in majk and maj0[e["key"]] != majk[e["key"]]]
        row = {
            "test_seed": test_seed,
            "n_l0_events": len(test_l0),
            "n_lk_events": len(test_lk),
            "n_reinterpretation_events": len(flip_idx),
            "l0_correct_impression_acc": accuracy(pred_l0, truth_l0),
            "lk_stale_impression_acc": accuracy(pred_stale, truth_lk),
            "lk_updated_impression_acc": accuracy(pred_updated, truth_lk),
            "lk_state_only_acc": accuracy(pred_state_only, truth_lk),
            "reinterpretation_stale_acc": accuracy(pred_stale[flip_idx], truth_lk[flip_idx]),
            "reinterpretation_updated_acc": accuracy(pred_updated[flip_idx], truth_lk[flip_idx]),
            "reinterpretation_state_only_acc": accuracy(pred_state_only[flip_idx], truth_lk[flip_idx]),
        }
        rows.append(row)
        print(f"seed{test_seed}: L0={row['l0_correct_impression_acc']:.3f} "
              f"Lk stale={row['lk_stale_impression_acc']:.3f} updated={row['lk_updated_impression_acc']:.3f} "
              f"state-only={row['lk_state_only_acc']:.3f} flip(n={len(flip_idx)}) "
              f"stale={row['reinterpretation_stale_acc']} updated={row['reinterpretation_updated_acc']} "
              f"state-only={row['reinterpretation_state_only_acc']}")

    keys = ("l0_correct_impression_acc", "lk_stale_impression_acc", "lk_updated_impression_acc",
            "lk_state_only_acc", "reinterpretation_stale_acc", "reinterpretation_updated_acc",
            "reinterpretation_state_only_acc")
    mean = {key: float(np.mean([row[key] for row in rows if row[key] is not None])) for key in keys}
    result = {
        "experiment": "conditional_mlp_impression_feasibility",
        "source_policy": "historical scripted CookAgent (not learned)",
        "split": "leave-one-seed-out",
        "event_definition": "intent or target transition",
        "features": "Alice position + held object + two pot states + impression one-hot",
        "seeds": seeds,
        "rows": rows,
        "mean": mean,
        "updated_minus_stale": mean["lk_updated_impression_acc"] - mean["lk_stale_impression_acc"],
        "reinterpretation_updated_minus_stale": mean["reinterpretation_updated_acc"] - mean["reinterpretation_stale_acc"],
        "limitations": [
            "Alice capability labels come from scripted policies, not PPO checkpoints.",
            "Trajectories are highly deterministic and contain few unique reinterpretation contexts.",
            "This pilot validates the impression representation and counterfactual control only.",
        ],
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("mean:", json.dumps(mean, ensure_ascii=False))
    print("updated-stale:", result["updated_minus_stale"])
    print("reinterpretation updated-stale:", result["reinterpretation_updated_minus_stale"])


if __name__ == "__main__":
    run()
