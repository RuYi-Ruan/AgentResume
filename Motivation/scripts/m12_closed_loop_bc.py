"""M12: compare the behavior-cloned Alice with its L0 expert in closed loop."""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, "src")
from ocres.trainable import TrainableMacroAgent, load_policy_checkpoint, select_device
from ocres.agents import CookAgent
from ocres.data import TWO_POT
from ocres.grid import World
from ocres.runner import run_episode


CHECKPOINT = pathlib.Path("artifacts/checkpoints/m11_bc_l0.pt")
OUTPUT = pathlib.Path("data/m12_closed_loop_results.json")


def randomize_start(world, seed):
    rng = random.Random(1000 + seed)
    first, second = rng.sample(sorted(world.grid.passable), 2)
    state = world.env.state
    state.players[0].update_pos_and_or(first, (1, 0))
    state.players[1].update_pos_and_or(second, (1, 0))
    state.timestep = 0


def run_condition(seed, condition, model, spec, device, horizon):
    world = World.make(grid_rows=TWO_POT, horizon=horizon)
    randomize_start(world, seed)
    if condition == "learned_bc":
        alice = TrainableMacroAgent(world.grid, 0, model, spec, device=device, horizon=horizon)
    elif condition == "expert_l0":
        alice = CookAgent(world.grid, 0, parallel_after_delay=None)
    else:
        raise ValueError(condition)
    partner = CookAgent(world.grid, 1, parallel_after_delay=None)
    logs, metrics = run_episode(world, [alice, partner], horizon=horizon)
    decisions = sum(
        bool(row["info0"] and row["info0"].get("decision_id")) for row in logs
    ) if condition == "learned_bc" else None
    return {
        "seed": seed,
        "condition": condition,
        **metrics,
        "policy_decisions": decisions,
    }


def summarize(rows, condition):
    selected = [row for row in rows if row["condition"] == condition]
    deliveries = [row["deliveries"] for row in selected]
    return {
        "episodes": len(selected),
        "delivery_mean": sum(deliveries) / len(deliveries),
        "delivery_min": min(deliveries),
        "delivery_max": max(deliveries),
        "zero_delivery_episodes": sum(value == 0 for value in deliveries),
        "mean_gap": sum(row["mean_gap"] for row in selected if row["mean_gap"] is not None)
        / max(sum(row["mean_gap"] is not None for row in selected), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 17)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    model, spec, checkpoint_meta = load_policy_checkpoint(args.checkpoint, map_location=device)
    rows = []
    for seed in args.seeds:
        for condition in ("expert_l0", "learned_bc"):
            row = run_condition(seed, condition, model, spec, device, args.horizon)
            rows.append(row)
            print(f"seed={seed:02d} {condition:10s} deliveries={row['deliveries']} gap={row['mean_gap']}")
    expert = summarize(rows, "expert_l0")
    learned = summarize(rows, "learned_bc")
    result = {
        "milestone": "M12 closed-loop behavior-cloning baseline",
        "claim_scope": "closed-loop reproduction of scripted L0 only; not learned capability improvement",
        "checkpoint": str(args.checkpoint),
        "checkpoint_test_accuracy": checkpoint_meta.get("test", {}).get("accuracy"),
        "device": str(device),
        "horizon": args.horizon,
        "training_seed_range": [1, 7],
        "evaluation_seeds": args.seeds,
        "rows": rows,
        "summary": {"expert_l0": expert, "learned_bc": learned},
        "learned_to_expert_delivery_ratio": learned["delivery_mean"] / expert["delivery_mean"] if expert["delivery_mean"] else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("learned/expert delivery ratio:", result["learned_to_expert_delivery_ratio"])
    print("results:", args.output)


if __name__ == "__main__":
    main()

