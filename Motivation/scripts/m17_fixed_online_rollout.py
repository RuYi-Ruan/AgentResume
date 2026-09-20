"""Continue frozen M17 ambiguity events and measure downstream team score.

Each trial begins from an already-audited visible event.  Bob stays still for
Alice's first action, receives one intent prediction, and then both agents play
for a fixed evaluation window.  This isolates the causal effect of that one
prediction without changing Bob's 3x3 observation radius.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np

from ocres.generic_agents import LayoutAwareTrainableAgent, PredictionRoleBob
from ocres.grid import STAY, World
from ocres.impression_events import FacilityVocabulary
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.m17_protocol import realized_partner_action
from ocres.recipes import agent_held
from ocres.trainable import load_policy_checkpoint

try:
    from m17_prepare_fixed_events import controlled_world
except ModuleNotFoundError:
    from scripts.m17_prepare_fixed_events import controlled_world


SPLITS = pathlib.Path("data/m17_fixed_event_splits.json")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
FORMAL_RESPONSES = pathlib.Path("data/m17_qwen_formal_test.json")
OUTPUT = pathlib.Path("data/m17_fixed_online_gate.json")


def frame(state, index):
    player = state.players[index]
    return {
        "position": list(player.position),
        "orientation": list(player.orientation),
        "held": agent_held(state, index),
    }


def response_lookup(path):
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for row in data.get("rows", []):
        parsed = row.get("response", {}).get("parsed")
        if parsed is not None:
            lookup[(row["event_id"], row["impression"])] = parsed
    return lookup


def prediction_for(condition, event, responses):
    if condition == "oracle":
        return dict(event["scoring_only"]["post"]), "oracle"
    if condition == "inverted":
        return dict(event["scoring_only"]["pre"]), "inverted"
    impression = {"old": "pre", "new": "post", "none": "none"}[condition]
    return responses.get((event["event_id"], impression)), f"saved_qwen_{impression}"


def run_trial(event, condition, post_model, spec, responses, window):
    cooking_index = int(event["group"]["cooking_pot_index"])
    before = event["public_event"]["before"]
    alice_position = tuple(before["alice_visible_position"])
    bob_position = tuple(before["bob_position"])
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=700)
    cooking_pot = sorted(probe.grid.pot_locs)[cooking_index]
    world = controlled_world(cooking_pot, alice_position, bob_position, 700)
    alice = LayoutAwareTrainableAgent(
        world.grid, 0, post_model, spec, device="cpu", horizon=700, role_hint="cook"
    )
    bob = PredictionRoleBob(world.grid, 1, 700)

    # Observation tick: Bob is passive, exactly matching the frozen public event.
    before_alice = frame(world.env.state, 0)
    action, intent, target, info = alice.action(world.env.state, 0)
    world.env.step((action or STAY, STAY))
    observed = realized_partner_action(before_alice, frame(world.env.state, 0))
    if observed != event["public_event"]["realized_alice_action"]:
        raise ValueError(f"frozen event reconstruction mismatch: {event['event_id']}")
    vocabulary = FacilityVocabulary.from_grid(world.grid)
    actual_target = vocabulary.names[vocabulary.label(target)]
    if intent != "FETCH" or info is None or actual_target != event["scoring_only"]["post"]["target_facility"]:
        raise ValueError(f"post checkpoint label mismatch: {event['event_id']}")

    prediction, source = prediction_for(condition, event, responses)
    if prediction is None:
        raise ValueError(f"missing saved prediction for {event['event_id']} {condition}")
    accepted = bob.update_prediction(prediction["intent"], prediction["target_facility"])

    deliveries = 0
    reward_total = 0.0
    start_t = int(world.env.state.timestep)
    while (
        world.env.state.timestep < start_t + window
        and not world.env.is_done()
    ):
        state = world.env.state
        alice_action, _, _, _ = alice.action(state, deliveries)
        bob_action, _, _, _ = bob.action(state, deliveries)
        _, reward, _, _ = world.env.step((alice_action or STAY, bob_action or STAY))
        reward_total += float(reward)
        if reward > 0:
            deliveries += 1
    return {
        "event_id": event["event_id"],
        "condition": condition,
        "deliveries": deliveries,
        "reward": reward_total,
        "prediction": prediction,
        "prediction_source": source,
        "intent_correct": prediction["intent"] == "FETCH",
        "target_correct": prediction["target_facility"] == actual_target,
        "prediction_accepted": accepted,
        "completed_commitments": bob.completed_commitments,
        "route_yields": bob.route_yields,
    }


def summarize(rows):
    return {
        "trials": len(rows),
        "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "intent_accuracy": float(np.mean([row["intent_correct"] for row in rows])),
        "target_accuracy": float(np.mean([row["target_correct"] for row in rows])),
        "route_yields": sum(row["route_yields"] for row in rows),
        "delivery_distribution": dict(sorted(Counter(row["deliveries"] for row in rows).items())),
        "per_event": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=pathlib.Path, default=SPLITS)
    parser.add_argument("--split", choices=("development", "formal_test"), default="development")
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--responses", type=pathlib.Path, default=FORMAL_RESPONSES)
    parser.add_argument("--conditions", nargs="+", choices=("oracle", "inverted", "old", "new", "none"), default=("oracle", "inverted"))
    parser.add_argument("--events", type=int, default=20)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    split_data = json.loads(args.splits.read_text(encoding="utf-8"))
    events = split_data[args.split]["events"][:args.events]
    post_model, spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    responses = response_lookup(args.responses)
    result = {
        "milestone": "M17 fixed-event online score rollout",
        "split": args.split,
        "window_after_observed_action": args.window,
        "events": len(events),
        "conditions": {},
    }
    for condition in args.conditions:
        rows = []
        for index, event in enumerate(events, start=1):
            row = run_trial(event, condition, post_model, spec, responses, args.window)
            rows.append(row)
            print(f"{condition} {index}/{len(events)}: deliveries={row['deliveries']}")
        result["conditions"][condition] = summarize(rows)
    if "oracle" in result["conditions"] and "inverted" in result["conditions"]:
        left = {row["event_id"]: row["deliveries"] for row in result["conditions"]["oracle"]["per_event"]}
        right = {row["event_id"]: row["deliveries"] for row in result["conditions"]["inverted"]["per_event"]}
        differences = [left[event["event_id"]] - right[event["event_id"]] for event in events]
        result["controller_gate"] = {
            "paired_mean_oracle_minus_inverted": float(np.mean(differences)),
            "oracle_better": sum(value > 0 for value in differences),
            "equal": sum(value == 0 for value in differences),
            "inverted_better": sum(value < 0 for value in differences),
            "differences": differences,
            "passed": float(np.mean(differences)) > 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "conditions": {
            key: {name: value for name, value in summary.items() if name != "per_event"}
            for key, summary in result["conditions"].items()
        },
        "controller_gate": result.get("controller_gate"),
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
