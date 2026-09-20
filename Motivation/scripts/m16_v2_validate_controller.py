"""M16-V2 gate: does a correct intent prediction improve cooperation?"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np

from ocres.generic_agents import LayoutAwareTrainableAgent, PredictionRoleBob
from ocres.grid import STAY
from ocres.trainable import load_policy_checkpoint
from m16_v2_audit_checkpoints import is_key_state, randomized_world


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m16_v2_controller_gate.json")


def run_episode(seed, condition, pre_model, post_model, spec, horizon):
    world = randomized_world(seed, horizon)
    alice = LayoutAwareTrainableAgent(
        world.grid, 0, post_model, spec, device="cpu", horizon=horizon, role_hint="cook"
    )
    bob = PredictionRoleBob(world.grid, 1, horizon)
    deliveries = 0
    reward_total = 0.0
    previous_marker = None
    segment_is_key = False
    queried = False
    eligible_events = 0
    prediction_updates = 0

    while world.env.state.timestep < horizon and not world.env.is_done():
        state = world.env.state
        alice_action, alice_intent, alice_target, _ = alice.action(state, deliveries)
        bob_action, _, _, _ = bob.action(state, deliveries)
        marker = alice_intent, alice_target
        if marker != previous_marker:
            previous_marker = marker
            segment_is_key = is_key_state(state, world.grid) and alice_intent == "FETCH"
            queried = False
        visible = max(
            abs(state.players[0].position[0] - state.players[1].position[0]),
            abs(state.players[0].position[1] - state.players[1].position[1]),
        ) <= 1
        if segment_is_key and not queried and visible:
            queried = True
            pre_cf = LayoutAwareTrainableAgent(
                world.grid, 0, pre_model, spec, device="cpu", horizon=horizon, role_hint="cook"
            )
            pre_action, pre_intent, pre_target, _ = pre_cf.action(state.deepcopy(), deliveries)
            strict = (
                pre_intent == "PRE"
                and pre_action == alice_action
                and alice_action != STAY
                and pre_target != alice_target
            )
            if strict:
                eligible_events += 1
                predicted = "FETCH" if condition == "oracle" else "PRE"
                bob.update_prediction(predicted)
                prediction_updates += 1

        _, reward, _, _ = world.env.step((alice_action or STAY, bob_action or STAY))
        reward_total += float(reward)
        if reward > 0:
            deliveries += 1

    return {
        "condition": condition,
        "seed": int(seed),
        "deliveries": deliveries,
        "reward": reward_total,
        "eligible_events": eligible_events,
        "prediction_updates": prediction_updates,
    }


def summarize(rows):
    return {
        "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "min_deliveries": min(row["deliveries"] for row in rows),
        "max_deliveries": max(row["deliveries"] for row in rows),
        "mean_eligible_events": float(np.mean([row["eligible_events"] for row in rows])),
        "per_seed": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(9901, 9917)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, _ = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    conditions = {}
    for condition in ("oracle", "inverted"):
        conditions[condition] = [
            run_episode(seed, condition, pre_model, post_model, pre_spec, args.horizon)
            for seed in args.seeds
        ]
    oracle = {row["seed"]: row["deliveries"] for row in conditions["oracle"]}
    inverted = {row["seed"]: row["deliveries"] for row in conditions["inverted"]}
    differences = [oracle[seed] - inverted[seed] for seed in args.seeds]
    result = {
        "milestone": "M16-V2 Bob controller sensitivity gate",
        "prediction_effect": "one bounded task commitment",
        "seeds": args.seeds,
        "conditions": {name: summarize(rows) for name, rows in conditions.items()},
        "paired_oracle_minus_inverted": {
            "mean_difference": float(np.mean(differences)),
            "seeds_oracle_better": int(sum(value > 0 for value in differences)),
            "seeds_equal": int(sum(value == 0 for value in differences)),
            "differences": differences,
        },
        "passed": float(np.mean(differences)) > 0,
        "qwen_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("results:", args.output)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
