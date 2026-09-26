"""Diagnostic only: privileged assignment controller checks task attainability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from run import make_env


def evaluate_one(seed: int):
    env = make_env()
    env.reset(seed=500_000 + seed)
    world = env.unwrapped.world
    agents = world.agents
    landmarks = world.landmarks
    agent_start = np.stack([a.state.p_pos for a in agents])
    targets = np.stack([l.state.p_pos for l in landmarks])
    row, col = linear_sum_assignment(np.linalg.norm(agent_start[:, None] - targets[None], axis=-1))
    assigned = dict(zip(row, col))
    team_return = 0.0
    for _ in range(25):
        actions = {}
        for idx, agent in enumerate(agents):
            delta = targets[assigned[idx]] - agent.state.p_pos
            velocity = agent.state.p_vel
            desired = 2.5 * delta - 0.9 * velocity
            axis = int(np.argmax(np.abs(desired)))
            if abs(desired[axis]) < 0.08:
                action = 0
            elif axis == 0:
                action = 2 if desired[axis] > 0 else 1
            else:
                action = 4 if desired[axis] > 0 else 3
            actions[agent.name] = action
        _, rewards, terminations, truncations, _ = env.step(actions)
        team_return += float(np.mean(list(rewards.values())))
        if all(terminations.get(a.name, False) or truncations.get(a.name, False) for a in agents):
            break
    final_positions = np.stack([a.state.p_pos for a in agents])
    nearest = np.linalg.norm(final_positions[:, None] - targets[None], axis=-1).min(axis=0)
    env.close()
    return team_return, float(nearest.mean()), int(np.all(nearest < 0.15))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = [evaluate_one(seed) for seed in range(args.episodes)]
    array = np.asarray(values)
    summary = {
        "scope": "Privileged world-state controller; environment diagnostic only, not a valid MARL baseline or ETM result",
        "episodes": args.episodes,
        "mean_return": float(array[:, 0].mean()),
        "mean_nearest_landmark_distance": float(array[:, 1].mean()),
        "full_cover_episodes": int(array[:, 2].sum()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
