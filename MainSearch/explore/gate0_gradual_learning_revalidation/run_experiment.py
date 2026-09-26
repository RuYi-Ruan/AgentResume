"""Gate 0: revalidate gradual learning of three independent role MLPs."""

from __future__ import annotations

import argparse
import copy
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

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import make_env
from MainSearch.explore.overcooked_three_mlp_ctde_gate.run_experiment import (
    JOINT_ACTIONS,
    actor_actions,
    best_seen_action,
    context_key,
    joint_index,
    train_actors,
)
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import (
    PolicyMLP,
    play_episode,
)


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seeds: int = 5
    episodes: int = 360
    horizon: int = 180
    checkpoint_interval: int = 40
    eval_episodes: int = 12
    actor_learning_rate: float = 0.01
    exploration_end: float = 0.10


def evaluate(seed, checkpoint, actors, static_actors, values, counts, envs, config):
    learned_env, static_env, teacher_env, *hybrid_envs = envs
    canonical = list(learned_env[0].start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    rows = []
    for offset in range(config.eval_episodes):
        episode_seed = 90_000_000 + seed * 10_000 + offset
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        context = context_to_index[context_key(starts, canonical)]
        learned_roles = actor_actions(actors, starts)
        static_roles = actor_actions(static_actors, starts)
        learned_soups, _ = play_episode(
            *learned_env, learned_roles, episode_seed, config.horizon
        )
        static_soups, _ = play_episode(
            *static_env, static_roles, episode_seed, config.horizon
        )
        teacher_index = best_seen_action(values, counts, context)
        teacher_roles = JOINT_ACTIONS[teacher_index] if teacher_index is not None else None
        teacher_soups = None
        hybrid_soups = [None, None, None]
        agreements = [None, None, None]
        if teacher_roles is not None:
            teacher_soups, _ = play_episode(
                *teacher_env, teacher_roles, episode_seed, config.horizon
            )
            for agent in range(3):
                hybrid_roles = list(teacher_roles)
                hybrid_roles[agent] = learned_roles[agent]
                hybrid_soups[agent], _ = play_episode(
                    *hybrid_envs[agent],
                    tuple(hybrid_roles),
                    episode_seed,
                    config.horizon,
                )
                agreements[agent] = int(learned_roles[agent] == teacher_roles[agent])
        rows.append(
            {
                "seed": seed,
                "checkpoint": checkpoint,
                "eval_case": offset,
                "learned_soups": learned_soups,
                "static_soups": static_soups,
                "teacher_soups": teacher_soups,
                "learned_roles": "-".join(map(str, learned_roles)),
                "static_roles": "-".join(map(str, static_roles)),
                "teacher_roles": (
                    "-".join(map(str, teacher_roles)) if teacher_roles is not None else ""
                ),
                **{f"agent_{i}_agrees": agreements[i] for i in range(3)},
                **{f"agent_{i}_hybrid_soups": hybrid_soups[i] for i in range(3)},
            }
        )
    return rows


def run_seed(seed, config):
    torch.manual_seed(130_000 + seed)
    rng = random.Random(140_000 + seed)
    actors = [PolicyMLP() for _ in range(3)]
    static_actors = copy.deepcopy(actors)
    optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        for actor in actors
    ]
    train_mdp, train_env = make_env(config.horizon)
    eval_envs = [make_env(config.horizon) for _ in range(6)]
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    counts = np.zeros_like(values, dtype=np.int64)
    training = []
    evaluations = evaluate(
        seed, 0, actors, static_actors, values, counts, eval_envs, config
    )

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
            selected = joint_index(actor_actions(actors, starts))
        roles = JOINT_ACTIONS[selected]
        soups, actual_starts = play_episode(
            train_mdp, train_env, roles, episode_seed, config.horizon
        )
        counts[context, selected] += 1
        n = counts[context, selected]
        values[context, selected] += (soups - values[context, selected]) / n
        teacher_index = best_seen_action(values, counts, context)
        teacher_roles = JOINT_ACTIONS[teacher_index]
        losses = train_actors(actors, optimizers, actual_starts, teacher_roles)
        training.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": soups,
                "epsilon": epsilon,
                "sampled_roles": "-".join(map(str, roles)),
                "teacher_roles": "-".join(map(str, teacher_roles)),
                **{f"agent_{i}_loss": losses[i] for i in range(3)},
            }
        )
        if (episode + 1) % config.checkpoint_interval == 0:
            evaluations.extend(
                evaluate(
                    seed,
                    episode + 1,
                    actors,
                    static_actors,
                    values,
                    counts,
                    eval_envs,
                    config,
                )
            )
    return training, evaluations


def interval(values):
    center = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return center, [center - 1.96 * se, center + 1.96 * se]


def aggregate(evaluations, config):
    checkpoints = sorted({row["checkpoint"] for row in evaluations})
    curve = []
    for checkpoint in checkpoints:
        rows = [row for row in evaluations if row["checkpoint"] == checkpoint]
        available = [row for row in rows if row["teacher_soups"] is not None]
        item = {
            "checkpoint": checkpoint,
            "evaluation_games_per_condition": len(rows),
            "learned_mean_soups": float(np.mean([row["learned_soups"] for row in rows])),
            "static_mean_soups": float(np.mean([row["static_soups"] for row in rows])),
            "teacher_mean_soups": (
                float(np.mean([row["teacher_soups"] for row in available]))
                if available else None
            ),
        }
        for agent in range(3):
            item[f"agent_{agent}_agreement"] = (
                float(np.mean([row[f"agent_{agent}_agrees"] for row in available]))
                if available else None
            )
            item[f"agent_{agent}_hybrid_soups"] = (
                float(np.mean([row[f"agent_{agent}_hybrid_soups"] for row in available]))
                if available else None
            )
        curve.append(item)

    first, last = checkpoints[0], checkpoints[-1]
    seed_ids = sorted({row["seed"] for row in evaluations})
    gains, per_seed = [], {}
    for seed in seed_ids:
        initial = [
            row["learned_soups"] for row in evaluations
            if row["seed"] == seed and row["checkpoint"] == first
        ]
        final = [
            row["learned_soups"] for row in evaluations
            if row["seed"] == seed and row["checkpoint"] == last
        ]
        gain = float(np.mean(final) - np.mean(initial))
        gains.append(gain)
        per_seed[str(seed)] = gain
    gain, gain_ci = interval(gains)
    static_invariant = all(
        len({
            (row["static_roles"], row["static_soups"])
            for row in evaluations
            if row["seed"] == seed and row["eval_case"] == eval_case
        }) == 1
        for seed in seed_ids for eval_case in range(config.eval_episodes)
    )
    initial_map = {
        (row["seed"], row["eval_case"]): row["learned_roles"]
        for row in evaluations if row["checkpoint"] == first
    }
    final_rows = [row for row in evaluations if row["checkpoint"] == last]
    behavior_change_rate = float(np.mean([
        row["learned_roles"] != initial_map[(row["seed"], row["eval_case"])]
        for row in final_rows
    ]))
    agent_behavior_change_rates = []
    for agent in range(3):
        agent_behavior_change_rates.append(float(np.mean([
            row["learned_roles"].split("-")[agent]
            != initial_map[(row["seed"], row["eval_case"])].split("-")[agent]
            for row in final_rows
        ])))
    return curve, {
        "status": "development_gate0_gradual_learning_revalidation",
        "claim_boundary": "Continuous high-level role learning with a central payoff teacher and scripted low-level execution; no ETM and not a formal main result.",
        "config": asdict(config),
        "training_games": config.seeds * config.episodes,
        "evaluation_cases": len(evaluations),
        "static_control_invariant": static_invariant,
        "initial_learned_soups": curve[0]["learned_mean_soups"],
        "final_learned_soups": curve[-1]["learned_mean_soups"],
        "paired_seed_gain_soups": gain,
        "gain_normal_approx_95ci": gain_ci,
        "improved_seeds": sum(value > 0 for value in gains),
        "per_seed_gain_soups": per_seed,
        "final_agent_agreement": [curve[-1][f"agent_{i}_agreement"] for i in range(3)],
        "final_agent_hybrid_soups": [curve[-1][f"agent_{i}_hybrid_soups"] for i in range(3)],
        "final_teacher_soups": curve[-1]["teacher_mean_soups"],
        "initial_to_final_behavior_change_rate": behavior_change_rate,
        "initial_to_final_agent_behavior_change_rates": agent_behavior_change_rates,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=Config.seed_start)
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--checkpoint-interval", type=int, default=Config.checkpoint_interval)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        episodes=args.episodes,
        horizon=args.horizon,
        checkpoint_interval=args.checkpoint_interval,
        eval_episodes=args.eval_episodes,
    )
    training, evaluations = [], []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        seed_training, seed_evaluations = run_seed(seed, config)
        training.extend(seed_training)
        evaluations.extend(seed_evaluations)
        print(f"seed {seed} complete", flush=True)
    curve, summary = aggregate(evaluations, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training.csv", training)
    write_csv(args.output_dir / "evaluations.csv", evaluations)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
