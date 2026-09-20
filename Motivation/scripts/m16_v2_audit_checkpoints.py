"""Audit frozen M16-V2 Alice checkpoints before any Qwen calls."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import Counter

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from ocres.generic_agents import LayoutAwareTrainableAgent, ScriptedMacroAgent
from ocres.grid import STAY, World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.recipes import agent_held, pot_kinds
from ocres.runner import run_episode
from ocres.trainable import load_policy_checkpoint, parse_pots


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m16_v2_checkpoint_audit.json")


def randomized_world(seed, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    rng = random.Random(7000 + int(seed))
    alice, bob = rng.sample(sorted(world.grid.passable), 2)
    world.env.state.players[0].update_pos_and_or(alice, (1, 0))
    world.env.state.players[1].update_pos_and_or(bob, (1, 0))
    world.env.state.timestep = 0
    return world


def controlled_world(cooking_pot, alice_position, bob_position, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    state = world.env.state
    state.players[0].update_pos_and_or(alice_position, (1, 0))
    state.players[1].update_pos_and_or(bob_position, (1, 0))
    state.timestep = 0
    state.add_object(SoupState.get_soup(cooking_pot, num_onions=3, cooking_tick=5))
    return world


def policy_action(world, model, spec, horizon):
    agent = LayoutAwareTrainableAgent(
        world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
    )
    action, intent, target, _ = agent.action(world.env.state, 0)
    return action, intent, target


def strict_checkpoint_pairs(pre_model, post_model, spec, horizon, required):
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    available = []
    for cooking_index, cooking_pot in enumerate(sorted(probe.grid.pot_locs)):
        for alice in sorted(probe.grid.passable):
            for bob in sorted(probe.grid.passable):
                if alice == bob or max(abs(alice[0] - bob[0]), abs(alice[1] - bob[1])) > 1:
                    continue
                world = controlled_world(cooking_pot, alice, bob, horizon)
                pre_action, pre_intent, pre_target = policy_action(world, pre_model, spec, horizon)
                post_action, post_intent, post_target = policy_action(world, post_model, spec, horizon)
                if not (
                    pre_intent == "PRE"
                    and post_intent == "FETCH"
                    and pre_action == post_action
                    and pre_action != STAY
                    and pre_target != post_target
                ):
                    continue
                available.append(
                    {
                        "cooking_pot_index": cooking_index,
                        "alice_position": list(alice),
                        "bob_position": list(bob),
                        "alice_held": None,
                        "bob_held": None,
                        "first_action": list(pre_action),
                        "pre_intent": pre_intent,
                        "post_intent": post_intent,
                        "pre_internal_target_for_scoring_only": list(pre_target),
                        "post_internal_target_for_scoring_only": list(post_target),
                    }
                )

    # Deterministic round-robin selection avoids taking forty near-duplicate
    # states from the first lexicographic region of the map.
    groups = {}
    for row in available:
        key = row["cooking_pot_index"], tuple(row["first_action"])
        groups.setdefault(key, []).append(row)
    selected = []
    keys = sorted(groups)
    while keys and len(selected) < required:
        next_keys = []
        for key in keys:
            if groups[key] and len(selected) < required:
                selected.append(groups[key].pop(0))
            if groups[key]:
                next_keys.append(key)
        keys = next_keys

    action_counts = Counter(tuple(row["first_action"]) for row in selected)
    pot_counts = Counter(row["cooking_pot_index"] for row in selected)
    return {
        "passed": len(selected) >= required,
        "required_pairs": required,
        "available_pairs": len(available),
        "selected_pairs": len(selected),
        "event_records": 2 * len(selected),
        "label_counts": {"PRE": len(selected), "FETCH": len(selected)},
        "observation_only_best_possible_accuracy": 0.5 if selected else None,
        "first_action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "cooking_pot_counts": {str(key): value for key, value in sorted(pot_counts.items())},
        "pairs": selected,
    }


def is_key_context(row):
    kinds = [kind for _, kind in parse_pots(row["pot"])]
    running = any(kind in ("cooking", "ready") for kind in kinds)
    accepting = any(kind in ("empty", "items1", "items2") for kind in kinds)
    return row["held0"] is None and running and accepting


def is_key_state(state, grid):
    kinds = list(pot_kinds(state, grid).values())
    running = any(kind in ("cooking", "ready") for kind in kinds)
    accepting = any(kind in ("empty", "items1", "items2") for kind in kinds)
    return agent_held(state, 0) is None and running and accepting


def natural_visibility(model, spec, seeds, horizon):
    rows = []
    total_intents = Counter()
    total_actions = Counter()
    for seed in seeds:
        world = randomized_world(seed, horizon)
        alice = LayoutAwareTrainableAgent(
            world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
        )
        bob = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
        logs, metrics = run_episode(world, [alice, bob], horizon)
        visible_key = []
        previous_marker = None
        segment_is_key = False
        queried_this_segment = False
        for row in logs:
            target = tuple(row["target0"]) if isinstance(row["target0"], tuple) else row["target0"]
            marker = row["intent0"], target
            if marker != previous_marker:
                previous_marker = marker
                segment_is_key = is_key_context(row) and row["intent0"] in ("PRE", "FETCH")
                queried_this_segment = False
            visible = max(abs(row["p0"][0] - row["p1"][0]), abs(row["p0"][1] - row["p1"][1])) <= 1
            if segment_is_key and not queried_this_segment and visible:
                queried_this_segment = True
                action = tuple(row["a"][0]) if isinstance(row["a"][0], tuple) else row["a"][0]
                visible_key.append({"t": row["t"], "intent": row["intent0"], "action": action})
                total_intents[row["intent0"]] += 1
                total_actions[action] += 1
        rows.append(
            {
                "seed": int(seed),
                "deliveries": metrics["deliveries"],
                "visible_key_events": len(visible_key),
                "intent_counts": dict(Counter(item["intent"] for item in visible_key)),
            }
        )
    return {
        "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "mean_visible_key_events": float(np.mean([row["visible_key_events"] for row in rows])),
        "min_visible_key_events": min(row["visible_key_events"] for row in rows),
        "max_visible_key_events": max(row["visible_key_events"] for row in rows),
        "intent_counts": dict(total_intents),
        "action_counts": {str(key): value for key, value in total_actions.items()},
        "per_seed": rows,
    }


def natural_post_counterfactual(pre_model, post_model, spec, seeds, horizon):
    """Check natural post events against the pre policy at the exact same state."""

    per_seed = []
    action_counts = Counter()
    for seed in seeds:
        world = randomized_world(seed, horizon)
        post = LayoutAwareTrainableAgent(
            world.grid, 0, post_model, spec, device="cpu", horizon=horizon, role_hint="cook"
        )
        bob = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
        deliveries = 0
        previous_marker = None
        segment_is_key = False
        queried = False
        visible_events = 0
        matched = 0
        while world.env.state.timestep < horizon and not world.env.is_done():
            state = world.env.state
            post_action, post_intent, post_target, _ = post.action(state, deliveries)
            bob_action, _, _, _ = bob.action(state, deliveries)
            marker = post_intent, post_target
            if marker != previous_marker:
                previous_marker = marker
                segment_is_key = is_key_state(state, world.grid) and post_intent == "FETCH"
                queried = False
            visible = max(
                abs(state.players[0].position[0] - state.players[1].position[0]),
                abs(state.players[0].position[1] - state.players[1].position[1]),
            ) <= 1
            if segment_is_key and not queried and visible:
                queried = True
                visible_events += 1
                pre_cf = LayoutAwareTrainableAgent(
                    world.grid, 0, pre_model, spec, device="cpu", horizon=horizon, role_hint="cook"
                )
                pre_action, pre_intent, pre_target, _ = pre_cf.action(state.deepcopy(), deliveries)
                if (
                    pre_intent == "PRE"
                    and pre_action == post_action
                    and pre_action != STAY
                    and pre_target != post_target
                ):
                    matched += 1
                    action_counts[tuple(post_action)] += 1
            _, reward, _, _ = world.env.step((post_action or STAY, bob_action or STAY))
            if reward > 0:
                deliveries += 1
        per_seed.append(
            {
                "seed": int(seed),
                "deliveries": deliveries,
                "visible_post_key_events": visible_events,
                "strict_counterfactual_matches": matched,
            }
        )
    total_visible = sum(row["visible_post_key_events"] for row in per_seed)
    total_matched = sum(row["strict_counterfactual_matches"] for row in per_seed)
    return {
        "visible_post_key_events": total_visible,
        "strict_counterfactual_matches": total_matched,
        "match_rate": total_matched / total_visible if total_visible else 0.0,
        "mean_matches_per_seed": float(np.mean([row["strict_counterfactual_matches"] for row in per_seed])),
        "min_matches_per_seed": min(row["strict_counterfactual_matches"] for row in per_seed),
        "action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "per_seed": per_seed,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(9801, 9817)))
    parser.add_argument("--required-pairs", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, _ = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    result = {
        "milestone": "M16-V2 frozen checkpoint ambiguity and visibility audit",
        "seeds": args.seeds,
        "strict_pairs": strict_checkpoint_pairs(
            pre_model, post_model, pre_spec, args.horizon, args.required_pairs
        ),
        "natural_pre": natural_visibility(pre_model, pre_spec, args.seeds, args.horizon),
        "natural_post": natural_visibility(post_model, post_spec, args.seeds, args.horizon),
        "natural_post_counterfactual": natural_post_counterfactual(
            pre_model, post_model, pre_spec, args.seeds, args.horizon
        ),
        "qwen_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "strict_pairs"}, ensure_ascii=False, indent=2))
    print(json.dumps({key: value for key, value in result["strict_pairs"].items() if key != "pairs"}, ensure_ascii=False, indent=2))
    print("results:", args.output)
    if not result["strict_pairs"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
