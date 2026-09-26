"""Structural suitability pilot for three-agent VMAS Transport."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import vmas


CONDITIONS = ("all_active", "two_active", "all_idle")


@dataclass(frozen=True)
class Config:
    seeds: int = 5
    parallel_worlds: int = 64
    steps: int = 200
    n_agents: int = 3


def actions_for(env, condition):
    package = env.scenario.packages[0].state.pos
    goal = env.world.landmarks[0].state.pos
    direction = torch.nn.functional.normalize(goal - package, dim=-1)
    side = torch.stack((-direction[:, 1], direction[:, 0]), dim=-1)
    actions = []
    for index, agent in enumerate(env.agents):
        if condition == "all_idle" or (condition == "two_active" and index == 2):
            actions.append(torch.zeros_like(agent.state.pos))
            continue
        lateral = (index - 1) * 0.07
        staging = package - 0.16 * direction + lateral * side
        distance = torch.linalg.norm(agent.state.pos - staging, dim=-1, keepdim=True)
        desired = torch.where(distance < 0.09, direction, staging - agent.state.pos)
        actions.append(torch.nn.functional.normalize(desired, dim=-1))
    return actions


def run_condition(seed, condition, config):
    env = vmas.make_env(
        scenario="transport",
        num_envs=config.parallel_worlds,
        device="cpu",
        continuous_actions=True,
        max_steps=config.steps,
        n_agents=config.n_agents,
    )
    env.reset(seed=seed)
    total = torch.zeros(config.parallel_worlds)
    for _ in range(config.steps):
        _, reward, done, _ = env.step(actions_for(env, condition))
        total += reward[0].detach().cpu()
    package = env.scenario.packages[0].state.pos.detach().cpu()
    goal = env.world.landmarks[0].state.pos.detach().cpu()
    distances = torch.linalg.norm(package - goal, dim=-1)
    return [
        {
            "seed": seed,
            "world": index,
            "condition": condition,
            "return": float(total[index]),
            "final_distance": float(distances[index]),
        }
        for index in range(config.parallel_worlds)
    ]


def summarize(rows, config):
    by_condition = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "worlds": len(selected),
            "mean_return": float(np.mean([row["return"] for row in selected])),
            "mean_final_distance": float(np.mean([row["final_distance"] for row in selected])),
        }
    lookup = {(row["seed"], row["world"], row["condition"]): row for row in rows}
    paired = {}
    for comparator in ("two_active", "all_idle"):
        differences = [
            lookup[(seed, world, "all_active")]["return"]
            - lookup[(seed, world, comparator)]["return"]
            for seed in range(config.seeds)
            for world in range(config.parallel_worlds)
        ]
        paired["all_active_minus_" + comparator] = {
            "mean_return_gain": float(np.mean(differences)),
            "positive_worlds": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "negative_worlds": sum(value < 0 for value in differences),
            "per_seed_mean": {
                str(seed): float(np.mean(differences[
                    seed * config.parallel_worlds:(seed + 1) * config.parallel_worlds
                ]))
                for seed in range(config.seeds)
            },
        }
    return {
        "status": "development_structural_suitability",
        "claim_boundary": "Fixed diagnostic controllers only; no online learning or ETM tested.",
        "config": asdict(config),
        "conditions": by_condition,
        "paired": paired,
    }


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--parallel-worlds", type=int, default=Config.parallel_worlds)
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(seeds=args.seeds, parallel_worlds=args.parallel_worlds, steps=args.steps)
    rows = []
    for seed in range(config.seeds):
        for condition in CONDITIONS:
            rows.extend(run_condition(seed, condition, config))
        print(f"seed {seed} complete", flush=True)
    summary = summarize(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
