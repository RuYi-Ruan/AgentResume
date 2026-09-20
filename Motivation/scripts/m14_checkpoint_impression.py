"""M14: stale/updated impression test using genuinely trained checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys
from collections import Counter

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ocres.agents import CookAgent
from ocres.data import TWO_POT
from ocres.grid import World
from ocres.runner import run_episode
from ocres.trainable import (
    INTENT_ID,
    INTENTS,
    MacroIntentPolicy,
    TrainableMacroAgent,
    encode_log_observation,
    legal_intent_mask,
    load_policy_checkpoint,
    seed_everything,
    select_device,
)


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m11_bc_l0.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m13_ppo_post.pt")
OUTPUT = pathlib.Path("data/m14_checkpoint_impression_results.json")
DETAILS = pathlib.Path("data/m14_event_predictions.csv")


def randomize_start(world, seed):
    rng = random.Random(1000 + int(seed))
    first, second = rng.sample(sorted(world.grid.passable), 2)
    state = world.env.state
    state.players[0].update_pos_and_or(first, (1, 0))
    state.players[1].update_pos_and_or(second, (1, 0))
    state.timestep = 0


def context_key(row):
    return (
        tuple(row["p0"]),
        tuple(row["p1"]),
        None if row["held0"] in (None, "None") else str(row["held0"]),
        None if row["held1"] in (None, "None") else str(row["held1"]),
        str(row["pot"]),
    )


def collect_events(model, spec, device, seed, level, horizon):
    world = World.make(grid_rows=TWO_POT, horizon=horizon)
    randomize_start(world, seed)
    alice = TrainableMacroAgent(world.grid, 0, model, spec, device=device, horizon=horizon)
    partner = CookAgent(world.grid, 1, parallel_after_delay=None)
    logs, metrics = run_episode(world, [alice, partner], horizon=horizon)
    events = []
    deliveries = 0
    for row in logs:
        info = row["info0"]
        if info and info.get("decision_id"):
            events.append(
                {
                    "seed": int(seed),
                    "level": level,
                    "intent": row["intent0"],
                    "target": tuple(row["target0"]),
                    "t": int(row["t"]),
                    "decision_id": int(info["decision_id"]),
                    "x": encode_log_observation(row, deliveries, horizon, spec),
                    "mask": legal_intent_mask(row),
                    "key": context_key(row),
                }
            )
        if row["r"] > 0:
            deliveries += 1
    return events, metrics


def add_belief(events, level):
    belief = np.asarray([1.0, 0.0] if level == "pre" else [0.0, 1.0], dtype=np.float32)
    return np.stack([np.concatenate([event["x"], belief]) for event in events])


def train_model(x, y, masks, input_dim, hidden_dim, device, epochs, seed):
    seed_everything(seed)
    model = MacroIntentPolicy(input_dim, hidden_dim=hidden_dim).to(device)
    counts = np.bincount(y, minlength=len(INTENTS))
    weights = np.zeros(len(INTENTS), dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
        torch.from_numpy(masks.astype(np.bool_)),
    )
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y, batch_mask in loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            loss = criterion(model(batch_x, batch_mask), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, x, masks, device):
    model.eval()
    logits = model(torch.from_numpy(x.astype(np.float32)).to(device), torch.from_numpy(masks).to(device))
    return logits.argmax(dim=1).cpu().numpy()


def accuracy(prediction, truth, indices=None):
    if indices is not None:
        prediction, truth = prediction[indices], truth[indices]
    return float(np.mean(prediction == truth)) if len(truth) else None


def majority_map(events, level):
    counts = {}
    for event in events:
        if event["level"] == level:
            counts.setdefault(event["key"], Counter())[event["intent"]] += 1
    return {key: value.most_common(1)[0][0] for key, value in counts.items()}


def paired_bootstrap(rows, seed, updated_key="updated_acc", stale_key="stale_acc", samples=20000):
    differences = np.asarray(
        [row[updated_key] - row[stale_key] for row in rows if row[updated_key] is not None], dtype=float
    )
    if not len(differences):
        return None
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(samples)])
    return {
        "mean_updated_minus_stale": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "seeds_updated_better": int(np.sum(differences > 0)),
        "seed_count": len(rows),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--details", type=pathlib.Path, default=DETAILS)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(4001, 4025)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(5001, 5017)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    pre_policy, pre_spec, _ = load_policy_checkpoint(args.pre_checkpoint, map_location=device)
    post_policy, post_spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location=device)
    if pre_spec != post_spec:
        raise SystemExit("pre/post observation specs differ")

    train_events, test_events = [], []
    capability = {"train": {"pre": [], "post": []}, "test": {"pre": [], "post": []}}
    for split, seeds in (("train", args.train_seeds), ("test", args.test_seeds)):
        destination = train_events if split == "train" else test_events
        for seed in seeds:
            for level, policy in (("pre", pre_policy), ("post", post_policy)):
                events, metrics = collect_events(policy, pre_spec, device, seed, level, args.horizon)
                destination.extend(events)
                capability[split][level].append(metrics["deliveries"])
        print(f"collected {split}: {len(destination)} decision events")

    y_train = np.asarray([INTENT_ID[event["intent"]] for event in train_events])
    masks_train = np.stack([event["mask"] for event in train_events])
    x_conditional = np.concatenate(
        [add_belief([event], event["level"])[0][None, :] for event in train_events], axis=0
    )
    x_state = np.stack([event["x"] for event in train_events])
    conditional = train_model(
        x_conditional, y_train, masks_train, pre_spec.dim + 2, args.hidden_dim, device, args.epochs, args.seed
    )
    state_only = train_model(
        x_state, y_train, masks_train, pre_spec.dim, args.hidden_dim, device, args.epochs, args.seed + 1
    )

    test_pre = [event for event in test_events if event["level"] == "pre"]
    test_post = [event for event in test_events if event["level"] == "post"]
    y_pre = np.asarray([INTENT_ID[event["intent"]] for event in test_pre])
    y_post = np.asarray([INTENT_ID[event["intent"]] for event in test_post])
    masks_pre = np.stack([event["mask"] for event in test_pre])
    masks_post = np.stack([event["mask"] for event in test_post])
    pre_correct = predict(conditional, add_belief(test_pre, "pre"), masks_pre, device)
    post_updated = predict(conditional, add_belief(test_post, "post"), masks_post, device)
    post_stale = predict(conditional, add_belief(test_post, "pre"), masks_post, device)
    post_state = predict(state_only, np.stack([event["x"] for event in test_post]), masks_post, device)

    pre_map, post_map = majority_map(train_events, "pre"), majority_map(train_events, "post")
    reinterpretation = np.asarray(
        [
            index
            for index, event in enumerate(test_post)
            if event["key"] in pre_map
            and event["key"] in post_map
            and pre_map[event["key"]] != post_map[event["key"]]
        ],
        dtype=np.int64,
    )
    per_seed = []
    for seed in args.test_seeds:
        indices = np.asarray([i for i, event in enumerate(test_post) if event["seed"] == seed])
        flip_indices = np.intersect1d(indices, reinterpretation)
        per_seed.append(
            {
                "seed": seed,
                "events": len(indices),
                "stale_acc": accuracy(post_stale, y_post, indices),
                "updated_acc": accuracy(post_updated, y_post, indices),
                "state_only_acc": accuracy(post_state, y_post, indices),
                "reinterpretation_events": len(flip_indices),
                "reinterpretation_stale_acc": accuracy(post_stale, y_post, flip_indices),
                "reinterpretation_updated_acc": accuracy(post_updated, y_post, flip_indices),
                "reinterpretation_state_only_acc": accuracy(post_state, y_post, flip_indices),
            }
        )

    result = {
        "milestone": "M14 trained-checkpoint impression counterfactual",
        "pre_checkpoint": str(args.pre_checkpoint),
        "post_checkpoint": str(args.post_checkpoint),
        "device": str(device),
        "split": {"train_seeds": args.train_seeds, "test_seeds": args.test_seeds},
        "capability_deliveries": {
            split: {
                level: {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}
                for level, values in levels.items()
            }
            for split, levels in capability.items()
        },
        "event_counts": {"train": len(train_events), "test_pre": len(test_pre), "test_post": len(test_post)},
        "accuracy": {
            "pre_correct_impression": accuracy(pre_correct, y_pre),
            "post_stale_impression": accuracy(post_stale, y_post),
            "post_updated_impression": accuracy(post_updated, y_post),
            "post_state_only": accuracy(post_state, y_post),
        },
        "reinterpretation": {
            "events": len(reinterpretation),
            "post_stale_impression": accuracy(post_stale, y_post, reinterpretation),
            "post_updated_impression": accuracy(post_updated, y_post, reinterpretation),
            "post_state_only": accuracy(post_state, y_post, reinterpretation),
        },
        "per_test_seed": per_seed,
        "paired_bootstrap": paired_bootstrap(per_seed, args.seed),
        "reinterpretation_paired_bootstrap": paired_bootstrap(
            per_seed,
            args.seed + 1,
            updated_key="reinterpretation_updated_acc",
            stale_key="reinterpretation_stale_acc",
        ),
        "limitations": [
            "The impression is supplied as an oracle one-hot belief; online belief inference is not tested yet.",
            "Exact reinterpretation contexts are defined from training trajectories and may be repetitive.",
            "This phase evaluates intent inference on fixed trajectories, not downstream team reward.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    reinterpretation_set = set(reinterpretation.tolist())
    with args.details.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "seed",
                "t",
                "decision_id",
                "alice_true_intent",
                "alice_target",
                "stale_prediction",
                "updated_prediction",
                "state_only_prediction",
                "stale_correct",
                "updated_correct",
                "state_only_correct",
                "is_reinterpretation",
                "pre_majority_intent",
                "post_majority_intent",
                "alice_position",
                "partner_position",
                "alice_held",
                "partner_held",
                "pot_states",
            ),
        )
        writer.writeheader()
        for index, event in enumerate(test_post):
            key = event["key"]
            truth = event["intent"]
            writer.writerow(
                {
                    "seed": event["seed"],
                    "t": event["t"],
                    "decision_id": event["decision_id"],
                    "alice_true_intent": truth,
                    "alice_target": event["target"],
                    "stale_prediction": INTENTS[post_stale[index]],
                    "updated_prediction": INTENTS[post_updated[index]],
                    "state_only_prediction": INTENTS[post_state[index]],
                    "stale_correct": int(INTENTS[post_stale[index]] == truth),
                    "updated_correct": int(INTENTS[post_updated[index]] == truth),
                    "state_only_correct": int(INTENTS[post_state[index]] == truth),
                    "is_reinterpretation": int(index in reinterpretation_set),
                    "pre_majority_intent": pre_map.get(key),
                    "post_majority_intent": post_map.get(key),
                    "alice_position": key[0],
                    "partner_position": key[1],
                    "alice_held": key[2],
                    "partner_held": key[3],
                    "pot_states": key[4],
                }
            )
    print(json.dumps({"accuracy": result["accuracy"], "reinterpretation": result["reinterpretation"], "paired_bootstrap": result["paired_bootstrap"]}, ensure_ascii=False, indent=2))
    print("results:", args.output)
    print("event details:", args.details)


if __name__ == "__main__":
    main()
