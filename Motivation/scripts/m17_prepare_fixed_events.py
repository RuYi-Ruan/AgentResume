"""Freeze disjoint M17 development and formal paired intent events.

The public payload contains only Bob's local observations and Alice's realized
first action.  PRE/FETCH labels and target facilities live under a separate
``scoring_only`` key and must never be passed to an LLM prompt builder.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pathlib
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from ocres.generic_agents import LayoutAwareTrainableAgent
from ocres.grid import STAY, World
from ocres.impression_events import FacilityVocabulary
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.m17_protocol import (
    partner_visible,
    realized_partner_action,
    stable_event_id,
    validate_partition_provenance,
)
from ocres.recipes import agent_held
from ocres.sensor import visible_state
from ocres.trainable import load_policy_checkpoint


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OLD_AUDIT = pathlib.Path("data/m16_v2_checkpoint_audit.json")
OUTPUT = pathlib.Path("data/m17_fixed_event_splits.json")


def controlled_world(cooking_pot, alice_position, bob_position, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    state = world.env.state
    state.players[0].update_pos_and_or(alice_position, (1, 0))
    state.players[1].update_pos_and_or(bob_position, (1, 0))
    state.timestep = 0
    state.add_object(SoupState.get_soup(cooking_pot, num_onions=3, cooking_tick=5))
    return world


def policy_decision(world, model, spec, horizon):
    agent = LayoutAwareTrainableAgent(
        world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
    )
    action, intent, target, _ = agent.action(world.env.state, 0)
    return action, intent, target


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
            if observation["partner_visible"] is not None
            else None
        ),
        "remembered_pots": observation["seen_pots"],
    }


def candidate(cooking_index, cooking_pot, alice_pos, bob_pos, pre_model, post_model, spec, horizon):
    pre_world = controlled_world(cooking_pot, alice_pos, bob_pos, horizon)
    post_world = controlled_world(cooking_pot, alice_pos, bob_pos, horizon)
    pre_action, pre_intent, pre_target = policy_decision(pre_world, pre_model, spec, horizon)
    post_action, post_intent, post_target = policy_decision(post_world, post_model, spec, horizon)
    if not (
        pre_intent == "PRE"
        and post_intent == "FETCH"
        and pre_action == post_action
        and pre_action != STAY
        and pre_target != post_target
    ):
        return None

    pre_before_state = pre_world.env.state
    post_before_state = post_world.env.state
    pre_before_alice = player_frame(pre_before_state, 0)
    post_before_alice = player_frame(post_before_state, 0)
    pre_before_obs = visible_state(pre_before_state, pre_world.grid, 1, {})
    post_before_obs = visible_state(post_before_state, post_world.grid, 1, {})
    if local_public(pre_before_obs) != local_public(post_before_obs):
        return None

    pre_world.env.step((pre_action, STAY))
    post_world.env.step((post_action, STAY))
    pre_after_state = pre_world.env.state
    post_after_state = post_world.env.state
    pre_after_alice = player_frame(pre_after_state, 0)
    post_after_alice = player_frame(post_after_state, 0)
    pre_realized = realized_partner_action(pre_before_alice, pre_after_alice)
    post_realized = realized_partner_action(post_before_alice, post_after_alice)
    if pre_realized != post_realized or pre_realized["kind"] != "move":
        return None
    pre_after_obs = visible_state(
        pre_after_state, pre_world.grid, 1, pre_before_obs["seen_pots"]
    )
    post_after_obs = visible_state(
        post_after_state, post_world.grid, 1, post_before_obs["seen_pots"]
    )
    before_public = local_public(pre_before_obs)
    after_public = local_public(pre_after_obs)
    if before_public != local_public(post_before_obs) or after_public != local_public(post_after_obs):
        return None
    if not pre_after_state.time_independent_equal(post_after_state):
        return None

    vocabulary = FacilityVocabulary.from_grid(pre_world.grid)
    public_event = {
        "before": before_public,
        "realized_alice_action": pre_realized,
        "after": after_public,
    }
    event_id = stable_event_id(public_event)
    return {
        "seed": 0,
        "episode_id": "controlled-m17-v1",
        "event_id": event_id,
        "group": {
            "cooking_pot_index": int(cooking_index),
            "action_delta": pre_realized["delta"],
        },
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


def collect_candidates(pre_model, post_model, spec, horizon):
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    rows = []
    seen = set()
    for cooking_index, cooking_pot in enumerate(sorted(probe.grid.pot_locs)):
        for alice in sorted(probe.grid.passable):
            for bob in sorted(probe.grid.passable):
                if alice == bob or not partner_visible(alice, bob):
                    continue
                row = candidate(
                    cooking_index, cooking_pot, alice, bob,
                    pre_model, post_model, spec, horizon,
                )
                if row is not None and row["event_id"] not in seen:
                    seen.add(row["event_id"])
                    rows.append(row)
    return rows


def old_pilot_signatures(path):
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    selected = {}
    for row in data["strict_pairs"]["pairs"]:
        action = tuple(row["first_action"])
        selected.setdefault(action, row)
    return {
        (
            int(row["cooking_pot_index"]),
            tuple(row["alice_position"]),
            tuple(row["bob_position"]),
            tuple(row["first_action"]),
        )
        for row in selected.values()
    }


def row_signature(row):
    before = row["public_event"]["before"]
    return (
        int(row["group"]["cooking_pot_index"]),
        tuple(before["alice_visible_position"]),
        tuple(before["bob_position"]),
        tuple(row["group"]["action_delta"]),
    )


def balanced_take(rows, count):
    actions = sorted({tuple(row["group"]["action_delta"]) for row in rows})
    base, remainder = divmod(count, len(actions)) if actions else (0, 0)
    result = []
    for action_index, action in enumerate(actions):
        quota = base + (1 if action_index < remainder else 0)
        by_pot = {}
        for row in rows:
            if tuple(row["group"]["action_delta"]) == action:
                by_pot.setdefault(int(row["group"]["cooking_pot_index"]), []).append(row)
        for values in by_pot.values():
            values.sort(key=lambda item: item["event_id"])
        pots = sorted(by_pot)
        chosen = []
        while pots and len(chosen) < quota:
            remaining_pots = []
            for pot in pots:
                if by_pot[pot] and len(chosen) < quota:
                    chosen.append(by_pot[pot].pop(0))
                if by_pot[pot]:
                    remaining_pots.append(pot)
            pots = remaining_pots
        result.extend(chosen)
    chosen_ids = {row["event_id"] for row in result}
    while len(result) < count:
        remaining = [row for row in rows if row["event_id"] not in chosen_ids]
        if not remaining:
            break
        action_counts = Counter(tuple(row["group"]["action_delta"]) for row in result)
        pot_counts = Counter(int(row["group"]["cooking_pot_index"]) for row in result)
        row = min(
            remaining,
            key=lambda item: (
                action_counts[tuple(item["group"]["action_delta"])],
                pot_counts[int(item["group"]["cooking_pot_index"])],
                item["event_id"],
            ),
        )
        result.append(row)
        chosen_ids.add(row["event_id"])
    return result


def split_summary(rows):
    return {
        "events": len(rows),
        "pot_counts": dict(Counter(str(row["group"]["cooking_pot_index"]) for row in rows)),
        "action_counts": dict(Counter(str(tuple(row["group"]["action_delta"])) for row in rows)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--old-audit", type=pathlib.Path, default=OLD_AUDIT)
    parser.add_argument("--development-events", type=int, default=20)
    parser.add_argument("--history-events", type=int, default=40)
    parser.add_argument("--formal-events", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, pre_meta = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, post_meta = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    candidates = collect_candidates(pre_model, post_model, pre_spec, args.horizon)
    pilot_signatures = old_pilot_signatures(args.old_audit)
    old_pilot = [row for row in candidates if row_signature(row) in pilot_signatures]
    invalid_old_pilot = len(pilot_signatures) - len(old_pilot)
    remaining = [row for row in candidates if row not in old_pilot]
    extra_dev = balanced_take(remaining, args.development_events - len(old_pilot))
    development = old_pilot + extra_dev
    development_ids = {row["event_id"] for row in development}
    history_probe = balanced_take(
        [
            row for row in remaining
            if row["event_id"] not in development_ids
            and row["public_event"]["after"]["alice_visible_position"] is not None
        ],
        args.history_events,
    )
    history_ids = {row["event_id"] for row in history_probe}
    formal = balanced_take(
        [
            row for row in remaining
            if row["event_id"] not in development_ids and row["event_id"] not in history_ids
        ],
        args.formal_events,
    )
    if (
        len(development) < args.development_events
        or len(history_probe) < args.history_events
        or len(formal) < args.formal_events
    ):
        raise RuntimeError(
            "not enough disjoint strict events for requested splits: "
            f"candidates={len(candidates)}, development={len(development)}/"
            f"{args.development_events}, history={len(history_probe)}/"
            f"{args.history_events}, formal={len(formal)}/{args.formal_events}"
        )
    validate_partition_provenance({
        "development": development,
        "history_probe": history_probe,
        "formal_test": formal,
    })

    result = {
        "milestone": "M17 frozen paired-event split gate",
        "passed": True,
        "qwen_calls": 0,
        "available_candidates": len(candidates),
        "old_pilot_signatures_permanently_excluded": len(pilot_signatures),
        "old_pilot_events_valid_for_development_only": len(old_pilot),
        "old_pilot_events_not_matched": invalid_old_pilot,
        "action_observation_rule": (
            "Alice is visible immediately before acting; the realized displacement is observed "
            "even if that displacement takes her outside the following 3x3 snapshot."
        ),
        "public_after_state_matched": True,
        "realized_action_matched": True,
        "labels_are_outside_public_event": True,
        "history_selection_rule": (
            "public geometry only: Alice remains in Bob's 3x3 snapshot after the first move"
        ),
        "checkpoint_meta": {"pre": pre_meta, "post": post_meta},
        "development": {"summary": split_summary(development), "events": development},
        "history_probe": {"summary": split_summary(history_probe), "events": history_probe},
        "formal_test": {"summary": split_summary(formal), "events": formal},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": True,
        "qwen_calls": 0,
        "available_candidates": len(candidates),
        "old_pilot_signatures_permanently_excluded": len(pilot_signatures),
        "old_pilot_events_valid_for_development_only": len(old_pilot),
        "old_pilot_events_not_matched": invalid_old_pilot,
        "development": result["development"]["summary"],
        "history_probe": result["history_probe"]["summary"],
        "formal_test": result["formal_test"]["summary"],
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
