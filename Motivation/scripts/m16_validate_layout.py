"""M16 gate 1: validate map playability, geometric ambiguity and visibility."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import Counter, defaultdict

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np

from ocres.generic_agents import ScriptedMacroAgent
from ocres.grid import World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.m16_observation import alice_visible_before_action
from ocres.runner import run_episode


OUTPUT = pathlib.Path("data/m16_v2_layout_validation.json")


def randomized_world(seed, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    rng = random.Random(int(seed))
    alice, bob = rng.sample(sorted(world.grid.passable), 2)
    world.env.state.players[0].update_pos_and_or(alice, (1, 0))
    world.env.state.players[1].update_pos_and_or(bob, (1, 0))
    world.env.state.timestep = 0
    return world


def geometric_ambiguity(world):
    groups = (
        ("FETCH", world.grid.onion_locs),
        ("GET_DISH", world.grid.dish_locs),
        ("POT_TASK", world.grid.pot_locs),
        ("DELIVER", world.grid.serve_locs),
    )
    total = ambiguous = 0
    for position in sorted(world.grid.passable):
        action_to_groups = defaultdict(set)
        for label, facilities in groups:
            for facility in facilities:
                paths = [
                    world.grid.bfs(position, stand, ())
                    for stand, _ in world.grid.interact_stand_cells(facility)
                ]
                paths = [path for path in paths if path is not None]
                if not paths:
                    continue
                path = min(paths, key=lambda value: (len(value), value))
                action = (
                    (path[0][0] - position[0], path[0][1] - position[1])
                    if path
                    else "interact"
                )
                action_to_groups[action].add(label)
        for labels in action_to_groups.values():
            total += 1
            ambiguous += len(labels) >= 2
    return {"action_groups": total, "ambiguous_action_groups": ambiguous, "rate": ambiguous / total}


def run_condition(seed, parallel, horizon):
    world = randomized_world(seed, horizon)
    alice = ScriptedMacroAgent(world.grid, 0, "cook", parallel, horizon)
    bob = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
    logs, metrics = run_episode(world, [alice, bob], horizon)
    previous_intent = None
    queried_this_segment = False
    segments = observed = 0
    observed_mix = Counter()
    for row in logs:
        if row["intent0"] != previous_intent:
            previous_intent = row["intent0"]
            queried_this_segment = False
            segments += 1
        if not queried_this_segment:
            visible = max(
                abs(row["p0"][0] - row["p1"][0]),
                abs(row["p0"][1] - row["p1"][1]),
            ) <= 1
            if visible:
                queried_this_segment = True
                observed += 1
                observed_mix[row["intent0"]] += 1
    return {
        "seed": int(seed),
        "deliveries": metrics["deliveries"],
        "reward": metrics["reward"],
        "intent_segments": segments,
        "observed_segments": observed,
        "coverage": observed / segments if segments else 0.0,
        "observed_intents": dict(observed_mix),
    }


def summary(rows):
    mix = Counter()
    for row in rows:
        mix.update(row["observed_intents"])
    return {
        "mean_deliveries": float(np.mean([row["deliveries"] for row in rows])),
        "min_deliveries": min(row["deliveries"] for row in rows),
        "max_deliveries": max(row["deliveries"] for row in rows),
        "mean_intent_segments": float(np.mean([row["intent_segments"] for row in rows])),
        "mean_observed_segments": float(np.mean([row["observed_segments"] for row in rows])),
        "mean_coverage": float(np.mean([row["coverage"] for row in rows])),
        "observed_intent_mix": dict(mix),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    probe = randomized_world(args.seeds[0], args.horizon)
    conditions = {
        "conservative_reference": [run_condition(seed, False, args.horizon) for seed in args.seeds],
        "parallel_reference": [run_condition(seed, True, args.horizon) for seed in args.seeds],
    }
    result = {
        "milestone": "M16-V2 active-wait map and visibility gate",
        "layout": "ambiguous_kitchen_v2",
        "grid_rows": AMBIGUOUS_KITCHEN_V2,
        "seeds": args.seeds,
        "horizon": args.horizon,
        "visibility": "3x3; at most one query at the first actually observed action in each intent segment",
        "features": {
            "passable": len(probe.grid.passable),
            "pots": len(probe.grid.pot_locs),
            "onions": len(probe.grid.onion_locs),
            "dishes": len(probe.grid.dish_locs),
            "serves": len(probe.grid.serve_locs),
        },
        "geometric_ambiguity": geometric_ambiguity(probe),
        "conditions": {
            name: {"summary": summary(rows), "per_seed": rows}
            for name, rows in conditions.items()
        },
        "interpretation": [
            "This is a scripted reference gate, not the formal trained Alice result.",
            "Visibility and empirical ambiguity must be measured again on frozen pre/post checkpoints.",
            "Unobserved intent segments never trigger a Bob query and are excluded from prediction accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "geometric_ambiguity": result["geometric_ambiguity"],
        "summaries": {name: value["summary"] for name, value in result["conditions"].items()},
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
