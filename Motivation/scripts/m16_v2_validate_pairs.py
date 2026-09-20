"""M16-V2 gate: build strict PRE/FETCH pairs before any LLM call.

Each pair starts from the same legal world state.  The conservative and
parallel reference policies must expose the same first primitive action while
committing to different physical macro targets.  Consequently an evaluator
that only sees the public state and first action receives identical features
for one PRE and one FETCH record.
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

from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from ocres.generic_agents import ScriptedMacroAgent
from ocres.grid import STAY, World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.recipes import agent_held, pot_kinds


OUTPUT = pathlib.Path("data/m16_v2_strict_pair_validation.json")


def action_json(action):
    return list(action) if isinstance(action, tuple) else action


def make_state(cooking_pot, alice_position, bob_position, cooking_tick, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    state = world.env.state
    state.players[0].update_pos_and_or(alice_position, (1, 0))
    state.players[1].update_pos_and_or(bob_position, (1, 0))
    state.timestep = 0
    state.add_object(
        SoupState.get_soup(cooking_pot, num_onions=3, cooking_tick=cooking_tick)
    )
    return world


def candidate_pair(cooking_pot, alice_position, bob_position, cooking_tick, horizon):
    world = make_state(cooking_pot, alice_position, bob_position, cooking_tick, horizon)
    state = world.env.state
    pre = ScriptedMacroAgent(world.grid, 0, "cook", False, horizon)
    post = ScriptedMacroAgent(world.grid, 0, "cook", True, horizon)
    pre_action, pre_intent, pre_target, _ = pre.action(state, 0)
    post_action, post_intent, post_target, _ = post.action(state, 0)
    if pre_intent != "PRE" or post_intent != "FETCH":
        return None
    if pre_action != post_action or pre_action == STAY:
        return None
    if pre_target == post_target:
        return None

    public = {
        "t": int(state.timestep),
        "alice_position": list(state.players[0].position),
        "alice_held": agent_held(state, 0),
        "bob_position": list(state.players[1].position),
        "bob_held": agent_held(state, 1),
        "pots": [
            {"facility": f"pot_{index}", "state": kind}
            for index, (_, kind) in enumerate(sorted(pot_kinds(state, world.grid).items()))
        ],
        "first_action": action_json(pre_action),
    }
    return {
        "public_observation": public,
        "pre_record": {
            "intent": pre_intent,
            "internal_target_for_scoring_only": list(pre_target),
        },
        "post_record": {
            "intent": post_intent,
            "internal_target_for_scoring_only": list(post_target),
        },
    }


def collect_pairs(horizon, cooking_tick):
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    pairs = []
    for cooking_pot in sorted(probe.grid.pot_locs):
        for alice in sorted(probe.grid.passable):
            visible_bob_positions = sorted(
                position
                for position in probe.grid.passable
                if position != alice
                and max(abs(position[0] - alice[0]), abs(position[1] - alice[1])) <= 1
            )
            for bob in visible_bob_positions:
                pair = candidate_pair(cooking_pot, alice, bob, cooking_tick, horizon)
                if pair is not None:
                    pairs.append(pair)
    return pairs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--required-pairs", type=int, default=40)
    parser.add_argument("--cooking-tick", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pairs = collect_pairs(args.horizon, args.cooking_tick)
    selected = pairs[: args.required_pairs]
    action_counts = Counter(tuple(pair["public_observation"]["first_action"]) for pair in selected)
    passed = len(selected) >= args.required_pairs
    result = {
        "milestone": "M16-V2 strict PRE/FETCH public-observation pair gate",
        "passed": passed,
        "required_pairs": args.required_pairs,
        "available_pairs": len(pairs),
        "selected_pairs": len(selected),
        "event_records": 2 * len(selected),
        "label_counts": {"PRE": len(selected), "FETCH": len(selected)},
        "observation_only_best_possible_accuracy_on_exact_pairs": 0.5 if selected else None,
        "first_action_counts": {str(key): value for key, value in sorted(action_counts.items())},
        "pair_definition": (
            "Within each pair the public observation and first primitive action are identical; "
            "only the reference policy's physical macro intent and target differ."
        ),
        "pairs": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "pairs"}, ensure_ascii=False, indent=2))
    print("results:", args.output)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
