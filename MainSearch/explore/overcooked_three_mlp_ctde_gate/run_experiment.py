"""Development CTDE gate with three independent decentralized MLP actors."""

from __future__ import annotations

import argparse
import csv
import itertools
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

from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import make_env
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import (
    ACTIONS,
    PolicyMLP,
    observation,
    play_episode,
)


JOINT_ACTIONS = list(itertools.product(range(len(ACTIONS)), repeat=3))


@dataclass(frozen=True)
class Config:
    seeds: int = 4
    episodes: int = 300
    horizon: int = 180
    window: int = 50
    actor_learning_rate: float = 0.01
    exploration_end: float = 0.10
    eval_episodes: int = 12


def context_key(starts, canonical_starts):
    index = {tuple(position): i for i, position in enumerate(canonical_starts)}
    return tuple(index[tuple(position)] for position in starts)


def joint_index(actions):
    return actions[0] * 9 + actions[1] * 3 + actions[2]


def actor_actions(policies, starts):
    output = []
    for index, policy in enumerate(policies):
        with torch.no_grad():
            output.append(int(policy(observation(index, starts[index])).argmax()))
    return tuple(output)


def best_seen_action(values, counts, context):
    seen = np.flatnonzero(counts[context] > 0)
    if len(seen) == 0:
        return None
    best_value = values[context, seen].max()
    best = seen[np.isclose(values[context, seen], best_value)]
    return int(best.min())


def train_actors(policies, optimizers, starts, target_actions):
    losses = []
    for index, (policy, optimizer) in enumerate(zip(policies, optimizers)):
        logits = policy(observation(index, starts[index])).unsqueeze(0)
        target = torch.tensor([target_actions[index]], dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def evaluate(seed, checkpoint, policies, mdp, env, config, values, counts):
    soups = []
    agreement = [0, 0, 0]
    measurable = 0
    canonical = list(mdp.start_player_positions)
    context_to_index = {
        value: index for index, value in enumerate(itertools.permutations(range(3)))
    }
    for offset in range(config.eval_episodes):
        episode_seed = 80_000_000 + seed * 10_000 + checkpoint * 100 + offset
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        actions = actor_actions(policies, starts)
        context = context_to_index[context_key(starts, canonical)]
        teacher_index = best_seen_action(values, counts, context)
        if teacher_index is not None:
            teacher = JOINT_ACTIONS[teacher_index]
            measurable += 1
            for i in range(3):
                agreement[i] += int(actions[i] == teacher[i])
        soup_count, _ = play_episode(mdp, env, actions, episode_seed, config.horizon)
        soups.append(soup_count)
    return {
        "seed": seed,
        "checkpoint": checkpoint,
        "mean_soups": float(np.mean(soups)),
        **{
            f"agent_{i}_teacher_agreement": agreement[i] / measurable if measurable else None
            for i in range(3)
        },
    }


def run_seed(seed, config):
    torch.manual_seed(1000 + seed)
    rng = random.Random(2000 + seed)
    policies = [PolicyMLP() for _ in range(3)]
    optimizers = [
        torch.optim.Adam(policy.parameters(), lr=config.actor_learning_rate)
        for policy in policies
    ]
    mdp, env = make_env(config.horizon)
    canonical = list(mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)
    evaluations = [evaluate(seed, 0, policies, mdp, env, config, values, counts)]
    training = []

    for episode in range(config.episodes):
        episode_seed = seed * 1_000_000 + episode
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        context = context_to_index[context_key(starts, canonical)]
        untried = np.flatnonzero(counts[context] == 0)
        fraction = episode / max(config.episodes - 1, 1)
        epsilon = 1.0 + fraction * (config.exploration_end - 1.0)
        if len(untried):
            selected = int(rng.choice(untried.tolist()))
        elif rng.random() < epsilon:
            selected = rng.randrange(len(JOINT_ACTIONS))
        else:
            selected = joint_index(actor_actions(policies, starts))
        actions = JOINT_ACTIONS[selected]
        soups, actual_starts = play_episode(mdp, env, actions, episode_seed, config.horizon)
        counts[context, selected] += 1
        n = counts[context, selected]
        values[context, selected] += (soups - values[context, selected]) / n

        teacher_index = best_seen_action(values, counts, context)
        teacher = JOINT_ACTIONS[teacher_index]
        losses = train_actors(policies, optimizers, actual_starts, teacher)
        training.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": soups,
                "epsilon": epsilon,
                **{f"agent_{i}_loss": losses[i] for i in range(3)},
                **{f"agent_{i}_agrees": int(actions[i] == teacher[i]) for i in range(3)},
            }
        )
        if (episode + 1) % config.window == 0:
            evaluations.append(
                evaluate(seed, episode + 1, policies, mdp, env, config, values, counts)
            )
    return training, evaluations


def aggregate(evaluations, config):
    curve = []
    for checkpoint in sorted({row["checkpoint"] for row in evaluations}):
        values = [row for row in evaluations if row["checkpoint"] == checkpoint]
        row = {
            "checkpoint": checkpoint,
            "evaluation_games": len(values) * config.eval_episodes,
            "mean_soups": float(np.mean([value["mean_soups"] for value in values])),
        }
        for agent in range(3):
            observed = [
                value[f"agent_{agent}_teacher_agreement"]
                for value in values
                if value[f"agent_{agent}_teacher_agreement"] is not None
            ]
            row[f"agent_{agent}_teacher_agreement"] = (
                float(np.mean(observed)) if observed else None
            )
        curve.append(row)

    gains = []
    for seed in range(config.seeds):
        first = next(row for row in evaluations if row["seed"] == seed and row["checkpoint"] == 0)
        last = next(
            row for row in evaluations
            if row["seed"] == seed and row["checkpoint"] == config.episodes
        )
        gains.append(last["mean_soups"] - first["mean_soups"])
    se = float(np.std(gains, ddof=1) / math.sqrt(len(gains))) if len(gains) > 1 else 0.0
    last = curve[-1]
    summary = {
        "status": "development_ctde_gate",
        "claim_boundary": "Central payoff memory trains three separate MLP actors; fixed map and scripted low-level control; not a formal ETM result.",
        "config": asdict(config),
        "initial_mean_soups": curve[0]["mean_soups"],
        "final_mean_soups": last["mean_soups"],
        "paired_seed_gain_soups": float(np.mean(gains)),
        "normal_approx_95ci": [
            float(np.mean(gains) - 1.96 * se),
            float(np.mean(gains) + 1.96 * se),
        ],
        "improved_seeds": sum(gain > 0 for gain in gains),
        "final_teacher_agreement": [last[f"agent_{i}_teacher_agreement"] for i in range(3)],
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
    training, evaluations = [], []
    for seed in range(config.seeds):
        seed_training, seed_evaluations = run_seed(seed, config)
        training.extend(seed_training)
        evaluations.extend(seed_evaluations)
        print(f"seed {seed + 1}/{config.seeds} complete", flush=True)
    curve, summary = aggregate(evaluations, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training.csv", training)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
