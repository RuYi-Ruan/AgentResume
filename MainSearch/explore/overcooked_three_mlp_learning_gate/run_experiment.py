"""Development gate for gradual learning by three independent MLP policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import (
    GRID,
    MOVE,
    RoleAgent,
    make_env,
)


ACTIONS = (("cook", 0), ("cook", 1), ("serve", 0))


@dataclass(frozen=True)
class Config:
    seeds: int = 6
    episodes: int = 360
    horizon: int = 220
    window: int = 60
    learning_rate: float = 0.015
    epsilon_start: float = 0.90
    epsilon_end: float = 0.05
    eval_episodes: int = 12


class PolicyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 3))

    def forward(self, value):
        return self.net(value)


def observation(index, position):
    value = [position[0] / (len(GRID[0]) - 1), position[1] / (len(GRID) - 1)]
    value.extend(1.0 if index == other else 0.0 for other in range(3))
    return torch.tensor(value, dtype=torch.float32)


def ideal_action(position):
    if position[0] < 5:
        return 0
    if position[0] > 7:
        return 1
    return 2


def play_episode(mdp, env, action_ids, episode_seed, horizon):
    # The map is fixed, so preserve the cached motion planner across episodes.
    env.reset(regen_mdp=False)
    rng = random.Random(episode_seed)
    starts = list(mdp.start_player_positions)
    rng.shuffle(starts)
    for player, position in zip(env.state.players, starts):
        player.update_pos_and_or(position, rng.choice(MOVE))
    agents = [RoleAgent(mdp, i, *ACTIONS[action]) for i, action in enumerate(action_ids)]
    total_reward = 0
    for _ in range(horizon):
        actions = tuple(agent.action(env.state) for agent in agents)
        _, reward, done, _ = env.step(actions, joint_agent_action_info=[{}, {}, {}])
        total_reward += reward
        if done:
            break
    return total_reward / 20.0, starts


def epsilon_at(episode, config):
    fraction = min(episode / max(config.episodes - 1, 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def evaluate(seed, checkpoint, policies, mdp, env, config):
    soups = []
    correct = [0, 0, 0]
    for offset in range(config.eval_episodes):
        episode_seed = 90_000_000 + seed * 10_000 + checkpoint * 100 + offset
        rng = random.Random(episode_seed)
        starts = list(mdp.start_player_positions)
        rng.shuffle(starts)
        actions = []
        for index, policy in enumerate(policies):
            with torch.no_grad():
                action = int(policy(observation(index, starts[index])).argmax())
            actions.append(action)
            correct[index] += int(action == ideal_action(starts[index]))
        soup_count, _ = play_episode(mdp, env, actions, episode_seed, config.horizon)
        soups.append(soup_count)
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "mean_soups": float(np.mean(soups)),
        **{f"agent_{i}_role_accuracy": correct[i] / config.eval_episodes for i in range(3)},
    }


def run_seed(seed, config):
    torch.manual_seed(10_000 + seed)
    np.random.seed(20_000 + seed)
    rng = random.Random(30_000 + seed)
    policies = [PolicyMLP() for _ in range(3)]
    optimizers = [torch.optim.Adam(policy.parameters(), lr=config.learning_rate) for policy in policies]
    mdp, env = make_env(config.horizon)
    evaluations = [evaluate(seed, 0, policies, mdp, env, config)]
    training_rows = []
    for episode in range(config.episodes):
        episode_seed = seed * 1_000_000 + episode
        start_rng = random.Random(episode_seed)
        starts = list(mdp.start_player_positions)
        start_rng.shuffle(starts)
        epsilon = epsilon_at(episode, config)
        choices = []
        q_values = []
        for index, policy in enumerate(policies):
            q = policy(observation(index, starts[index]))
            action = rng.randrange(3) if rng.random() < epsilon else int(q.detach().argmax())
            choices.append(action)
            q_values.append(q[action])
        soups, actual_starts = play_episode(mdp, env, choices, episode_seed, config.horizon)
        target = torch.tensor(soups / 10.0, dtype=torch.float32)
        for optimizer, chosen_q in zip(optimizers, q_values):
            optimizer.zero_grad()
            (chosen_q - target).pow(2).backward()
            optimizer.step()
        training_rows.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": soups,
                "epsilon": epsilon,
                **{
                    f"agent_{i}_role_correct": int(choices[i] == ideal_action(actual_starts[i]))
                    for i in range(3)
                },
            }
        )
        if (episode + 1) % config.window == 0:
            evaluations.append(evaluate(seed, episode + 1, policies, mdp, env, config))
    return training_rows, evaluations


def aggregate(evaluations, config):
    checkpoints = sorted({row["checkpoint"] for row in evaluations})
    curve = []
    for checkpoint in checkpoints:
        values = [row for row in evaluations if row["checkpoint"] == checkpoint]
        curve.append(
            {
                "checkpoint": checkpoint,
                "evaluation_games": len(values) * config.eval_episodes,
                "mean_soups": float(np.mean([row["mean_soups"] for row in values])),
                **{
                    f"agent_{i}_role_accuracy": float(
                        np.mean([row[f"agent_{i}_role_accuracy"] for row in values])
                    )
                    for i in range(3)
                },
            }
        )
    first, last = curve[0], curve[-1]
    per_seed_gain = []
    for seed in range(config.seeds):
        start = next(row for row in evaluations if row["seed"] == seed and row["checkpoint"] == 0)
        end = next(
            row for row in evaluations
            if row["seed"] == seed and row["checkpoint"] == config.episodes
        )
        per_seed_gain.append(end["mean_soups"] - start["mean_soups"])
    se = (
        float(np.std(per_seed_gain, ddof=1) / math.sqrt(len(per_seed_gain)))
        if len(per_seed_gain) > 1
        else 0.0
    )
    summary = {
        "status": "development_gradual_learning_gate",
        "claim_boundary": "Three independent contextual MLP role policies in one scripted low-level Overcooked map; not a formal ETM result.",
        "config": asdict(config),
        "initial_mean_soups": first["mean_soups"],
        "final_mean_soups": last["mean_soups"],
        "paired_seed_gain_soups": float(np.mean(per_seed_gain)),
        "normal_approx_95ci": [
            float(np.mean(per_seed_gain) - 1.96 * se),
            float(np.mean(per_seed_gain) + 1.96 * se),
        ],
        "improved_seeds": sum(value > 0 for value in per_seed_gain),
        "final_agent_role_accuracy": [last[f"agent_{i}_role_accuracy"] for i in range(3)],
    }
    return curve, summary


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seeds=args.seeds,
        episodes=args.episodes,
        horizon=args.horizon,
        eval_episodes=args.eval_episodes,
    )
    training_rows, evaluations = [], []
    for seed in range(config.seeds):
        seed_training, seed_evaluations = run_seed(seed, config)
        training_rows.extend(seed_training)
        evaluations.extend(seed_evaluations)
        print(f"seed {seed + 1}/{config.seeds} complete", flush=True)
    curve, summary = aggregate(evaluations, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training.csv", training_rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
