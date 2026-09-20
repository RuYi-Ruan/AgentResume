"""M15: infer Alice's intent/facility after her first action, then cooperate online."""
from __future__ import annotations

import argparse
import csv
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ocres.agents import CookAgent
from ocres.data import TWO_POT
from ocres.grid import STAY, World
from ocres.impression_events import (
    FacilityVocabulary,
    action_observation_dim,
    encode_observed_action,
    extract_intent_change_events,
)
from ocres.partner_inference import PartnerInferenceNet, PredictionAwareBob
from ocres.recipes import agent_held, pot_kinds
from ocres.runner import run_episode
from ocres.trainable import INTENT_ID, INTENTS, TrainableMacroAgent, live_row, load_policy_checkpoint, seed_everything, select_device


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m11_bc_l0.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m13_ppo_post.pt")
OUTPUT = pathlib.Path("data/m15_online_action_impression_results.json")
DETAILS = pathlib.Path("data/m15_online_event_predictions.csv")
BOB_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m15_partner_inference.pt")


def randomize_start(world, seed):
    rng = random.Random(3000 + int(seed))
    first, second = rng.sample(sorted(world.grid.passable), 2)
    state = world.env.state
    state.players[0].update_pos_and_or(first, (1, 0))
    state.players[1].update_pos_and_or(second, (1, 0))
    state.timestep = 0


def belief(level):
    values = {"pre": (1.0, 0.0), "post": (0.0, 1.0), "none": (0.0, 0.0)}
    return np.asarray(values[level], dtype=np.float32)


def with_belief(feature, level):
    return np.concatenate((feature, belief(level))).astype(np.float32, copy=False)


def collect_episode_events(policy, spec, device, seed, level, horizon):
    world = World.make(grid_rows=TWO_POT, horizon=horizon)
    randomize_start(world, seed)
    alice = TrainableMacroAgent(world.grid, 0, policy, spec, device=device, horizon=horizon)
    neutral_partner = CookAgent(world.grid, 1, parallel_after_delay=None)
    logs, metrics = run_episode(world, [alice, neutral_partner], horizon=horizon)
    events, facilities = extract_intent_change_events(logs, world.grid, horizon=horizon, spec=spec)
    for event in events:
        event.update(seed=int(seed), level=level)
    return events, facilities, metrics


def class_weights(labels, classes, device):
    counts = np.bincount(labels, minlength=classes)
    weights = np.zeros(classes, dtype=np.float32)
    present = counts > 0
    weights[present] = len(labels) / (present.sum() * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_predictor(events, input_dim, facility_count, hidden_dim, epochs, device, seed):
    """Correct-impression samples teach conditioning; zero copies define control."""

    features, intents, facilities = [], [], []
    for event in events:
        for impression in (event["level"], "none"):
            features.append(with_belief(event["x"], impression))
            intents.append(INTENT_ID[event["intent"]])
            facilities.append(event["facility_id"])
    x = np.stack(features)
    yi = np.asarray(intents, dtype=np.int64)
    yf = np.asarray(facilities, dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(yi), torch.from_numpy(yf))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    model = PartnerInferenceNet(input_dim, len(INTENTS), facility_count, hidden_dim).to(device)
    intent_loss = torch.nn.CrossEntropyLoss(weight=class_weights(yi, len(INTENTS), device))
    facility_loss = torch.nn.CrossEntropyLoss(weight=class_weights(yf, facility_count, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for batch_x, batch_intent, batch_facility in loader:
            batch_x = batch_x.to(device)
            batch_intent = batch_intent.to(device)
            batch_facility = batch_facility.to(device)
            intent_logits, facility_logits = model(batch_x)
            loss = intent_loss(intent_logits, batch_intent) + facility_loss(facility_logits, batch_facility)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict(model, feature, impression, facility_names, device):
    model.eval()
    x = torch.from_numpy(with_belief(feature, impression)).unsqueeze(0).to(device)
    intent_logits, facility_logits = model(x)
    intent_id = int(intent_logits.argmax(dim=1).item())
    facility_id = int(facility_logits.argmax(dim=1).item())
    return INTENTS[intent_id], facility_names[facility_id]


def replay_metrics(model, events, impression, facilities, device):
    rows = []
    for event in events:
        guessed_intent, guessed_facility = predict(model, event["x"], impression, facilities.names, device)
        rows.append((guessed_intent == event["intent"], guessed_facility == event["facility"]))
    return {
        "events": len(rows),
        "intent_accuracy": float(np.mean([row[0] for row in rows])) if rows else None,
        "facility_accuracy": float(np.mean([row[1] for row in rows])) if rows else None,
        "joint_accuracy": float(np.mean([row[0] and row[1] for row in rows])) if rows else None,
    }


def public_row(state, grid, deliveries):
    row = live_row(state, grid, me=0)
    row["deliveries"] = deliveries
    return row


def movement_destination(position, action):
    if not isinstance(action, tuple) or action == STAY:
        return position
    return position[0] + action[0], position[1] + action[1]


def online_episode(policy, spec, model, device, seed, level, impression, horizon):
    world = World.make(grid_rows=TWO_POT, horizon=horizon)
    randomize_start(world, seed)
    facilities = FacilityVocabulary.from_grid(world.grid)
    alice = TrainableMacroAgent(world.grid, 0, policy, spec, device=device, horizon=horizon)
    bob = PredictionAwareBob(world.grid, 1, facilities)
    deliveries = 0
    total_reward = 0.0
    previous_intent = None
    events = []
    attempted_collisions = 0
    same_facility_ticks = 0

    while world.env.state.timestep < horizon and not world.env.is_done():
        state = world.env.state
        pre = public_row(state, world.grid, deliveries)
        alice_result = alice.action(state, deliveries)
        alice_action, alice_intent, alice_target = alice_result[:3]
        bob_action, bob_intent, bob_target = bob.action(state, deliveries)
        alice_action = alice_action if alice_action is not None else STAY
        bob_action = bob_action if bob_action is not None else STAY

        pos0, pos1 = tuple(state.players[0].position), tuple(state.players[1].position)
        dst0, dst1 = movement_destination(pos0, alice_action), movement_destination(pos1, bob_action)
        if (dst0 == dst1 and dst0 not in (pos0, pos1)) or (dst0 == pos1 and dst1 == pos0):
            attempted_collisions += 1
        if tuple(alice_target) in facilities.coordinate_to_id and tuple(bob_target) == tuple(alice_target):
            same_facility_ticks += 1

        _, reward, _, _ = world.env.step((alice_action, bob_action))
        total_reward += float(reward)
        if reward > 0:
            deliveries += 1
        post = public_row(world.env.state, world.grid, deliveries)

        changed = alice_intent != previous_intent
        previous_intent = alice_intent
        if changed:
            feature = encode_observed_action(
                pre, alice_action, post, deliveries=pre["deliveries"], horizon=horizon, spec=spec
            )
            counterfactual = {
                mode: predict(model, feature, mode, facilities.names, device)
                for mode in ("pre", "post", "none")
            }
            guessed_intent, guessed_facility = counterfactual[impression]
            true_facility_id = facilities.label(alice_target)
            true_facility = facilities.names[true_facility_id]
            bob.update_prediction(guessed_intent, guessed_facility)
            events.append(
                {
                    "condition": f"{level}_{impression}",
                    "seed": int(seed),
                    "t_action": int(pre["t"]),
                    "t_prediction": int(post["t"]),
                    "alice_first_action": str(alice_action),
                    "true_intent": alice_intent,
                    "guessed_intent": guessed_intent,
                    "intent_correct": int(guessed_intent == alice_intent),
                    "true_facility": true_facility,
                    "guessed_facility": guessed_facility,
                    "facility_correct": int(guessed_facility == true_facility),
                    "joint_correct": int(guessed_intent == alice_intent and guessed_facility == true_facility),
                    "cf_pre_intent": counterfactual["pre"][0],
                    "cf_post_intent": counterfactual["post"][0],
                    "cf_none_intent": counterfactual["none"][0],
                    "cf_pre_facility": counterfactual["pre"][1],
                    "cf_post_facility": counterfactual["post"][1],
                    "cf_none_facility": counterfactual["none"][1],
                    "cf_intent_changed_pre_to_post": int(counterfactual["pre"][0] != counterfactual["post"][0]),
                    "cf_facility_changed_pre_to_post": int(counterfactual["pre"][1] != counterfactual["post"][1]),
                    "bob_next_role": "serve" if guessed_intent in ("FETCH", "PLACE", "COOK_START") else "cook" if guessed_intent in ("GET_DISH", "PICKUP", "DELIVER") else "dynamic",
                }
            )

    count = len(events)
    metrics = {
        "condition": f"{level}_{impression}",
        "seed": int(seed),
        "events": count,
        "intent_accuracy": sum(row["intent_correct"] for row in events) / count if count else None,
        "facility_accuracy": sum(row["facility_correct"] for row in events) / count if count else None,
        "joint_accuracy": sum(row["joint_correct"] for row in events) / count if count else None,
        "deliveries": deliveries,
        "reward": total_reward,
        "attempted_collisions": attempted_collisions,
        "same_facility_ticks": same_facility_ticks,
        "facility_avoidances": bob.facility_avoidances,
        "route_yields": bob.route_yields,
        "blocked_yields": bob.blocked_yields,
        "cf_intent_switch_rate": sum(row["cf_intent_changed_pre_to_post"] for row in events) / count if count else None,
        "cf_facility_switch_rate": sum(row["cf_facility_changed_pre_to_post"] for row in events) / count if count else None,
    }
    return metrics, events


def summarize(rows):
    keys = (
        "events",
        "intent_accuracy",
        "facility_accuracy",
        "joint_accuracy",
        "deliveries",
        "reward",
        "attempted_collisions",
        "same_facility_ticks",
        "facility_avoidances",
        "route_yields",
        "blocked_yields",
        "cf_intent_switch_rate",
        "cf_facility_switch_rate",
    )
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def paired_bootstrap(rows, condition_a, condition_b, key, seed, samples=20000):
    left = {row["seed"]: row[key] for row in rows if row["condition"] == condition_a}
    right = {row["seed"]: row[key] for row in rows if row["condition"] == condition_b}
    seeds = sorted(set(left) & set(right))
    differences = np.asarray([right[value] - left[value] for value in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    means = np.asarray([rng.choice(differences, len(differences), replace=True).mean() for _ in range(samples)])
    return {
        "comparison": f"{condition_b} minus {condition_a}",
        "metric": key,
        "mean_difference": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(means, (0.025, 0.975))],
        "seeds_improved": int(np.sum(differences > 0)),
        "seed_count": len(seeds),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--details", type=pathlib.Path, default=DETAILS)
    parser.add_argument("--bob-checkpoint", type=pathlib.Path, default=BOB_CHECKPOINT)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(6001, 6025)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(7001, 7017)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
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
    facility_vocab = None
    collection_capability = {"train": {"pre": [], "post": []}, "test": {"pre": [], "post": []}}
    for split, seeds in (("train", args.train_seeds), ("test", args.test_seeds)):
        destination = train_events if split == "train" else test_events
        for seed in seeds:
            for level, policy in (("pre", pre_policy), ("post", post_policy)):
                events, facilities, metrics = collect_episode_events(
                    policy, pre_spec, device, seed, level, args.horizon
                )
                destination.extend(events)
                collection_capability[split][level].append(metrics["deliveries"])
                if facility_vocab is None:
                    facility_vocab = facilities
                elif facility_vocab.names != facilities.names:
                    raise RuntimeError("facility vocabulary changed across episodes")
        print(f"collected {split}: {len(destination)} intent-change events", flush=True)

    input_dim = action_observation_dim(pre_spec) + 2
    model = train_predictor(
        train_events, input_dim, len(facility_vocab.names), args.hidden_dim, args.epochs, device, args.seed
    )
    args.bob_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "agentresume-partner-inference-v1",
            "state_dict": model.state_dict(),
            "model": {
                "input_dim": input_dim,
                "intent_count": len(INTENTS),
                "facility_count": len(facility_vocab.names),
                "hidden_dim": args.hidden_dim,
            },
            "intents": INTENTS,
            "facilities": facility_vocab.names,
            "timing": "predict at t+1 after observing first action at t",
        },
        args.bob_checkpoint,
    )

    test_pre = [event for event in test_events if event["level"] == "pre"]
    test_post = [event for event in test_events if event["level"] == "post"]
    replay = {
        "pre_correct": replay_metrics(model, test_pre, "pre", facility_vocab, device),
        "post_stale": replay_metrics(model, test_post, "pre", facility_vocab, device),
        "post_updated": replay_metrics(model, test_post, "post", facility_vocab, device),
        "post_no_impression": replay_metrics(model, test_post, "none", facility_vocab, device),
    }
    print(json.dumps({"fixed_replay": replay}, ensure_ascii=False, indent=2), flush=True)

    conditions = (
        ("pre", "pre", pre_policy),
        ("post", "pre", post_policy),
        ("post", "post", post_policy),
        ("post", "none", post_policy),
    )
    episodes, event_rows = [], []
    for level, impression, policy in conditions:
        for seed in args.test_seeds:
            metrics, events = online_episode(
                policy, pre_spec, model, device, seed, level, impression, args.horizon
            )
            episodes.append(metrics)
            event_rows.extend(events)
        print(f"online {level}_{impression}: {summarize([r for r in episodes if r['condition'] == f'{level}_{impression}'])}", flush=True)

    conditions_summary = {
        name: summarize([row for row in episodes if row["condition"] == name])
        for name in ("pre_pre", "post_pre", "post_post", "post_none")
    }
    result = {
        "milestone": "M15 action-conditioned intent/facility inference with online cooperation",
        "approved_design": "docs/experiments/M15_EXPERIMENT_DESIGN.md",
        "device": str(device),
        "seed": args.seed,
        "horizon": args.horizon,
        "split": {"train_seeds": args.train_seeds, "test_seeds": args.test_seeds},
        "event_definition": "Alice macro intent differs from previous tick; one guess after first primitive action",
        "predictor_input": "public pre-state + Alice first primitive action + public post-state + impression",
        "facility_vocabulary": facility_vocab.names,
        "training_events": len(train_events),
        "test_events_fixed_replay": len(test_events),
        "fixed_replay": replay,
        "online_summary": conditions_summary,
        "online_per_seed": episodes,
        "paired_post_updated_minus_stale": {
            key: paired_bootstrap(episodes, "post_pre", "post_post", key, args.seed + index)
            for index, key in enumerate(("intent_accuracy", "facility_accuracy", "joint_accuracy", "reward", "deliveries", "attempted_collisions"))
        },
        "limitations": [
            "Impression is supplied as an oracle pre/post vector; automatic impression updating is not tested.",
            "Bob uses a fixed interpretable downstream controller, not a learned cooperation policy.",
            "Online trajectories diverge across conditions because predictions affect Bob's behavior; fixed replay is reported separately.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)
    print(json.dumps({"online_summary": conditions_summary, "paired": result["paired_post_updated_minus_stale"]}, ensure_ascii=False, indent=2), flush=True)
    print("results:", args.output)
    print("details:", args.details)


if __name__ == "__main__":
    main()
