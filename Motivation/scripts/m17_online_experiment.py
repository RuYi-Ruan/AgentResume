"""M17 online cooperation experiment.

Alice always uses the post-growth checkpoint.  At a genuinely new, visible
macro decision Bob observes Alice's first *realized* action.  The selected
predictor then updates one bounded Bob commitment, which can affect only later
ticks.  Hidden pre/post labels are used solely for event eligibility, oracle
controls, and scoring; they never enter Qwen prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np

from ocres.generic_agents import LayoutAwareTrainableAgent, PredictionRoleBob
from ocres.grid import STAY
from ocres.impression_events import FacilityVocabulary
from ocres.m17_protocol import partner_visible, realized_partner_action, stable_event_id
from ocres.recipes import agent_held
from ocres.sensor import visible_state
from ocres.trainable import load_policy_checkpoint
try:  # direct script execution
    from m16_v2_audit_checkpoints import is_key_state, randomized_world
    from m17_qwen_fixed import build_messages, call_validated, prompt_hash
except ModuleNotFoundError:  # imported as scripts.m17_online_experiment in tests
    from scripts.m16_v2_audit_checkpoints import is_key_state, randomized_world
    from scripts.m17_qwen_fixed import build_messages, call_validated, prompt_hash


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
IMPRESSIONS = pathlib.Path("data/m17_paired_impressions.json")
OUTPUT = pathlib.Path("data/m17_online_gate.json")
CACHE = pathlib.Path("data/m17_online_qwen_cache.json")


def player_frame(state, index):
    player = state.players[index]
    return {
        "position": list(player.position),
        "orientation": list(player.orientation),
        "held": agent_held(state, index),
    }


def local_public(observation):
    return {
        "bob_position": list(observation["pos"]),
        "bob_orientation": list(observation["orient"]),
        "bob_held": observation["held"],
        "tiles": observation["tiles"],
        "alice_visible_position": (
            list(observation["partner_visible"])
            if observation["partner_visible"] is not None else None
        ),
        "remembered_pots": observation["seen_pots"],
    }


def strict_growth_candidate(state, grid, post_decision, pre_decision):
    """Eligibility check made before the simultaneous environment step."""

    post_action, post_intent, post_target, post_info = post_decision
    pre_action, pre_intent, pre_target, _ = pre_decision
    alice = state.players[0].position
    bob = state.players[1].position
    return bool(
        post_info is not None
        and partner_visible(alice, bob)
        and is_key_state(state, grid)
        and post_intent == "FETCH"
        and pre_intent == "PRE"
        and post_action == pre_action
        and post_action != STAY
        and post_target != pre_target
    )


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def qwen_prediction(event, impression, impression_data, vocabulary, cache, cache_path):
    messages, fact_ids = build_messages(event, impression, impression_data, vocabulary)
    key = prompt_hash(messages)
    if key in cache:
        saved = cache[key]
        return saved.get("parsed"), saved.get("attempts", []), True, key
    parsed, attempts = call_validated(messages, fact_ids, vocabulary.names)
    cache[key] = {"parsed": parsed, "attempts": attempts}
    save_json(cache_path, cache)
    return parsed, attempts, False, key


def choose_prediction(condition, event, impression_data, vocabulary, cache, cache_path):
    scoring = event["scoring_only"]
    if condition == "oracle":
        return dict(scoring["post"]), {"source": "oracle"}
    if condition == "inverted":
        return dict(scoring["pre"]), {"source": "inverted"}
    impression = {"old": "pre", "new": "post", "none": "none"}[condition]
    parsed, attempts, cached, key = qwen_prediction(
        event, impression, impression_data, vocabulary, cache, cache_path
    )
    return parsed, {
        "source": "qwen",
        "impression": impression,
        "prompt_hash": key,
        "cached": cached,
        "attempts": attempts,
    }


def run_episode(seed, condition, pre_model, post_model, spec, impression_data,
                cache, cache_path, horizon):
    world = randomized_world(seed, horizon)
    alice = LayoutAwareTrainableAgent(
        world.grid, 0, post_model, spec, device="cpu", horizon=horizon, role_hint="cook"
    )
    bob = PredictionRoleBob(world.grid, 1, horizon)
    vocabulary = FacilityVocabulary.from_grid(world.grid)
    deliveries = 0
    reward_total = 0.0
    memory = {}
    events = []
    trigger_funnel = Counter()

    while world.env.state.timestep < horizon and not world.env.is_done():
        state = world.env.state
        before_t = int(state.timestep)
        before_alice = player_frame(state, 0)
        before_obs = visible_state(state, world.grid, 1, memory)
        memory = before_obs["seen_pots"]

        post_decision = alice.action(state, deliveries)
        alice_action, post_intent, post_target, post_info = post_decision
        bob_action, _, _, _ = bob.action(state, deliveries)

        # A fresh counterfactual agent is intentional: only its decision at the
        # exact public state is a hidden scoring label, never part of the prompt.
        pre_cf = LayoutAwareTrainableAgent(
            world.grid, 0, pre_model, spec, device="cpu", horizon=horizon, role_hint="cook"
        )
        pre_decision = pre_cf.action(state.deepcopy(), deliveries)
        if post_info is not None:
            trigger_funnel["fresh_decision"] += 1
            if partner_visible(state.players[0].position, state.players[1].position):
                trigger_funnel["visible"] += 1
                if is_key_state(state, world.grid):
                    trigger_funnel["key_state"] += 1
                    if post_intent == "FETCH":
                        trigger_funnel["post_fetch"] += 1
                        if pre_decision[1] == "PRE":
                            trigger_funnel["pre_pre"] += 1
                            if pre_decision[0] == alice_action and alice_action != STAY:
                                trigger_funnel["same_nonstay_action"] += 1
                                if pre_decision[2] != post_target:
                                    trigger_funnel["different_target"] += 1
        eligible_before = strict_growth_candidate(
            state, world.grid, post_decision, pre_decision
        )

        _, reward, _, _ = world.env.step((alice_action or STAY, bob_action or STAY))
        reward_total += float(reward)
        if reward > 0:
            deliveries += 1

        if not eligible_before:
            continue
        after_state = world.env.state
        after_alice = player_frame(after_state, 0)
        realized = realized_partner_action(before_alice, after_alice)
        expected_delta = list(alice_action) if isinstance(alice_action, tuple) else None
        if realized.get("kind") != "move" or realized.get("delta") != expected_delta:
            continue
        trigger_funnel["realized_move"] += 1
        after_obs = visible_state(after_state, world.grid, 1, memory)
        memory = after_obs["seen_pots"]
        public_event = {
            "before": local_public(before_obs),
            "realized_alice_action": realized,
            "after": local_public(after_obs),
        }
        pre_target = pre_decision[2]
        event = {
            "seed": int(seed),
            "episode_id": f"online-{condition}-{seed}",
            "event_id": stable_event_id(public_event),
            "public_event": public_event,
            "scoring_only": {
                "pre": {
                    "intent": "PRE",
                    "target_facility": vocabulary.names[vocabulary.label(pre_target)],
                },
                "post": {
                    "intent": "FETCH",
                    "target_facility": vocabulary.names[vocabulary.label(post_target)],
                },
            },
        }
        prediction, provenance = choose_prediction(
            condition, event, impression_data, vocabulary, cache, cache_path
        )
        accepted = False
        if prediction is not None:
            accepted = bob.update_prediction(
                prediction["intent"], prediction["target_facility"]
            )
        events.append({
            "t": before_t,
            "event_id": event["event_id"],
            "true_intent": "FETCH",
            "true_target_facility": event["scoring_only"]["post"]["target_facility"],
            "prediction": prediction,
            "prediction_accepted_by_controller": accepted,
            "prediction_provenance": provenance,
        })

    valid = [row for row in events if row["prediction"] is not None]
    return {
        "condition": condition,
        "seed": int(seed),
        "deliveries": int(deliveries),
        "reward": reward_total,
        "eligible_events": len(events),
        "valid_predictions": len(valid),
        "correct_intents": sum(row["prediction"]["intent"] == "FETCH" for row in valid),
        "correct_targets": sum(
            row["prediction"]["target_facility"] == row["true_target_facility"]
            for row in valid
        ),
        "accepted_prediction_updates": bob.accepted_prediction_updates,
        "ignored_prediction_updates": bob.ignored_prediction_updates,
        "completed_commitments": bob.completed_commitments,
        "route_yields": bob.route_yields,
        "trigger_funnel": dict(trigger_funnel),
        "events": events,
    }


def summarize(rows):
    events = sum(row["eligible_events"] for row in rows)
    valid = sum(row["valid_predictions"] for row in rows)
    return {
        "episodes": len(rows),
        "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "mean_reward": float(np.mean([row["reward"] for row in rows])),
        "total_events": events,
        "valid_predictions": valid,
        "intent_accuracy": (
            sum(row["correct_intents"] for row in rows) / events if events else None
        ),
        "target_accuracy": (
            sum(row["correct_targets"] for row in rows) / events if events else None
        ),
        "accepted_updates": sum(row["accepted_prediction_updates"] for row in rows),
        "ignored_updates": sum(row["ignored_prediction_updates"] for row in rows),
        "route_yields": sum(row["route_yields"] for row in rows),
        "deliveries_distribution": dict(sorted(Counter(row["deliveries"] for row in rows).items())),
        "per_seed": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--impressions", type=pathlib.Path, default=IMPRESSIONS)
    parser.add_argument(
        "--conditions", nargs="+", choices=("oracle", "inverted", "old", "new", "none"),
        default=("oracle", "inverted"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10001, 10009)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--cache", type=pathlib.Path, default=CACHE)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, _ = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    impressions = load_json(args.impressions, {})
    cache = load_json(args.cache, {})
    result = {
        "milestone": "M17 online cooperation",
        "horizon": args.horizon,
        "seeds": args.seeds,
        "conditions": {},
    }
    for condition in args.conditions:
        rows = []
        for seed in args.seeds:
            row = run_episode(
                seed, condition, pre_model, post_model, pre_spec, impressions,
                cache, args.cache, args.horizon,
            )
            rows.append(row)
            result["conditions"][condition] = summarize(rows)
            save_json(args.output, result)
            print(
                f"{condition} seed={seed}: deliveries={row['deliveries']} "
                f"events={row['eligible_events']} valid={row['valid_predictions']}"
            )
    if "oracle" in result["conditions"] and "inverted" in result["conditions"]:
        oracle = {row["seed"]: row["deliveries"] for row in result["conditions"]["oracle"]["per_seed"]}
        inverted = {row["seed"]: row["deliveries"] for row in result["conditions"]["inverted"]["per_seed"]}
        differences = [oracle[seed] - inverted[seed] for seed in args.seeds]
        result["controller_gate"] = {
            "paired_mean_oracle_minus_inverted": float(np.mean(differences)),
            "oracle_better": sum(value > 0 for value in differences),
            "equal": sum(value == 0 for value in differences),
            "inverted_better": sum(value < 0 for value in differences),
            "differences": differences,
            "passed": float(np.mean(differences)) > 0,
        }
    save_json(args.output, result)
    print(json.dumps({
        "conditions": {
            name: {key: value for key, value in summary.items() if key != "per_seed"}
            for name, summary in result["conditions"].items()
        },
        "controller_gate": result.get("controller_gate"),
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
