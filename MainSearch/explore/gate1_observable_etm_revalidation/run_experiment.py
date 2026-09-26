"""Gate 1: causal observable ETM revalidation on the Gate 0 learning path."""

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
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MainSearch.explore.observable_etm_revalidation.run_experiment import (
    PAIRS,
    ObservableETM,
    rollout_windows,
)
from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import make_env
from MainSearch.explore.overcooked_three_mlp_ctde_gate.run_experiment import (
    JOINT_ACTIONS,
    actor_actions,
    best_seen_action,
    context_key,
    joint_index,
    train_actors,
)
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import PolicyMLP


CONDITIONS = ("frozen_etm", "dynamic_etm")


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seeds: int = 5
    calibration_episodes: int = 120
    learning_episodes: int = 360
    horizon: int = 180
    macro_steps: int = 10
    window_episodes: int = 60
    visibility_radius: int = 4
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.01
    exploration_end: float = 0.10


def episode_starts(canonical, episode_seed):
    starts = list(canonical)
    random.Random(episode_seed).shuffle(starts)
    return starts


def evaluate_and_update(
    models, optimizers, hidden, summaries, events, phase, seed, episode
):
    rows = []
    class_weights = torch.tensor([0.25, 1, 1, 1, 1, 1], dtype=torch.float32)
    for window in range(len(summaries) - 1):
        for condition in CONDITIONS:
            can_update = phase == "calibration" or condition == "dynamic_etm"
            for pair in PAIRS:
                value = torch.tensor(
                    summaries[window][pair], dtype=torch.float32
                ).unsqueeze(0)
                logits, next_hidden = models[condition][pair].step(
                    value, hidden[condition][pair], update_hidden=can_update
                )
                label = events[window + 1][pair[1]]
                prediction = int(logits.detach().argmax())
                rows.append(
                    {
                        "seed": seed,
                        "phase": phase,
                        "episode": episode,
                        "window": window,
                        "condition": condition,
                        "pair": f"{pair[0]}->{pair[1]}",
                        "label": label,
                        "prediction": prediction,
                        "correct": int(prediction == label),
                        "nonempty": int(label != 0),
                    }
                )
                if can_update:
                    loss = nn.functional.cross_entropy(
                        logits,
                        torch.tensor([label], dtype=torch.long),
                        reduction="none",
                    ) * class_weights[label]
                    optimizers[condition][pair].zero_grad()
                    loss.mean().backward()
                    optimizers[condition][pair].step()
                    hidden[condition][pair] = next_hidden.detach()
    return rows


def run_seed(seed, config):
    torch.manual_seed(130_000 + seed)
    rng = random.Random(140_000 + seed)
    actors = [PolicyMLP() for _ in range(3)]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        for actor in actors
    ]

    base_models = {pair: ObservableETM() for pair in PAIRS}
    models = {
        condition: {pair: copy.deepcopy(base_models[pair]) for pair in PAIRS}
        for condition in CONDITIONS
    }
    optimizers = {
        condition: {
            pair: torch.optim.Adam(model.parameters(), lr=config.etm_learning_rate)
            for pair, model in models[condition].items()
        }
        for condition in CONDITIONS
    }
    hidden = {
        condition: {pair: torch.zeros(1, 32) for pair in PAIRS}
        for condition in CONDITIONS
    }

    train_mdp, train_env = make_env(config.horizon)
    observe_mdp, observe_env = make_env(config.horizon)
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    predictions, learning_trace = [], []

    initial_roles_by_context = {
        context: actor_actions(actors, [canonical[index] for index in context])
        for context in contexts
    }
    for calibration_episode in range(config.calibration_episodes):
        episode_seed = 70_000_000 + seed * 100_000 + calibration_episode
        starts = episode_starts(canonical, episode_seed)
        context = context_key(starts, canonical)
        roles = initial_roles_by_context[context]
        _, _, summaries, events = rollout_windows(
            observe_mdp, observe_env, roles, episode_seed, config
        )
        predictions.extend(
            evaluate_and_update(
                models,
                optimizers,
                hidden,
                summaries,
                events,
                "calibration",
                seed,
                calibration_episode,
            )
        )

    for episode in range(config.learning_episodes):
        episode_seed = seed * 1_000_000 + episode
        starts = episode_starts(canonical, episode_seed)
        context = context_to_index[context_key(starts, canonical)]
        proposals = actor_actions(actors, starts)
        untried = np.flatnonzero(payoff_counts[context] == 0)
        fraction = episode / max(config.learning_episodes - 1, 1)
        epsilon = 1.0 + fraction * (config.exploration_end - 1.0)
        if len(untried):
            selected = int(rng.choice(untried.tolist()))
        elif rng.random() < epsilon:
            selected = rng.randrange(len(JOINT_ACTIONS))
        else:
            selected = joint_index(proposals)
        sampled_roles = JOINT_ACTIONS[selected]
        training_soups, training_starts, _, _ = rollout_windows(
            train_mdp, train_env, sampled_roles, episode_seed, config
        )
        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            training_soups - payoff_values[context, selected]
        ) / count
        teacher_roles = JOINT_ACTIONS[
            best_seen_action(payoff_values, payoff_counts, context)
        ]
        train_actors(actors, actor_optimizers, training_starts, teacher_roles)
        observed_roles = actor_actions(actors, starts)
        _, _, summaries, events = rollout_windows(
            observe_mdp, observe_env, observed_roles, episode_seed, config
        )
        predictions.extend(
            evaluate_and_update(
                models,
                optimizers,
                hidden,
                summaries,
                events,
                "learning",
                seed,
                episode,
            )
        )
        learning_trace.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": training_soups,
                "sampled_roles": "-".join(map(str, sampled_roles)),
                "teacher_roles": "-".join(map(str, teacher_roles)),
                "observed_roles": "-".join(map(str, observed_roles)),
            }
        )
    return predictions, learning_trace


def metric_rows(rows, start, end, phase="learning"):
    output = []
    for condition in CONDITIONS:
        values = [
            row for row in rows
            if row["phase"] == phase and row["condition"] == condition
            and start <= row["episode"] < end
        ]
        nonempty = [row for row in values if row["nonempty"]]
        output.append(
            {
                "window_start": start,
                "window_end": end - 1,
                "condition": condition,
                "predictions": len(values),
                "nonempty_predictions": len(nonempty),
                "overall_accuracy": float(np.mean([row["correct"] for row in values])),
                "nonempty_accuracy": (
                    float(np.mean([row["correct"] for row in nonempty]))
                    if nonempty else 0.0
                ),
            }
        )
    return output


def aggregate(rows, trace, config):
    curve = []
    for start in range(0, config.learning_episodes, config.window_episodes):
        end = min(start + config.window_episodes, config.learning_episodes)
        curve.extend(metric_rows(rows, start, end))

    seed_ids = sorted({row["seed"] for row in rows})
    late_start = config.learning_episodes - config.window_episodes
    gains = []
    for seed in seed_ids:
        accuracy = {}
        for condition in CONDITIONS:
            values = [
                row for row in rows
                if row["seed"] == seed and row["phase"] == "learning"
                and row["condition"] == condition and row["episode"] >= late_start
                and row["nonempty"]
            ]
            accuracy[condition] = float(np.mean([row["correct"] for row in values]))
        gains.append(accuracy["dynamic_etm"] - accuracy["frozen_etm"])
    center = float(np.mean(gains))
    se = float(np.std(gains, ddof=1) / math.sqrt(len(gains))) if len(gains) > 1 else 0.0
    late = {
        row["condition"]: row for row in curve if row["window_start"] == late_start
    }

    calibration_equal = True
    for seed in seed_ids:
        frozen = [
            (r["prediction"], r["correct"]) for r in rows
            if r["seed"] == seed and r["phase"] == "calibration"
            and r["condition"] == "frozen_etm"
        ]
        dynamic = [
            (r["prediction"], r["correct"]) for r in rows
            if r["seed"] == seed and r["phase"] == "calibration"
            and r["condition"] == "dynamic_etm"
        ]
        calibration_equal &= frozen == dynamic

    pair_late = {}
    for pair in sorted({row["pair"] for row in rows}):
        pair_late[pair] = {}
        for condition in CONDITIONS:
            values = [
                row for row in rows
                if row["phase"] == "learning" and row["pair"] == pair
                and row["condition"] == condition and row["episode"] >= late_start
                and row["nonempty"]
            ]
            pair_late[pair][condition] = float(
                np.mean([row["correct"] for row in values]) if values else 0.0
            )

    return curve, {
        "status": "development_gate1_observable_etm_revalidation",
        "claim_boundary": "Cognition-only prediction on an exogenous continuous learning path; six independent directional ETMs, scripted low-level execution, not a formal result.",
        "config": asdict(config),
        "prediction_events": len(rows),
        "learning_trace_episodes": len(trace),
        "calibration_conditions_exact_match": calibration_equal,
        "late_metrics": late,
        "paired_seed_nonempty_accuracy_gain": center,
        "gain_normal_approx_95ci": [center - 1.96 * se, center + 1.96 * se],
        "dynamic_better_seeds": sum(gain > 0 for gain in gains),
        "per_seed_nonempty_accuracy_gain": {
            str(seed): gain for seed, gain in zip(seed_ids, gains)
        },
        "late_pair_nonempty_accuracy": pair_late,
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
    parser.add_argument("--calibration-episodes", type=int, default=Config.calibration_episodes)
    parser.add_argument("--learning-episodes", type=int, default=Config.learning_episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--window-episodes", type=int, default=Config.window_episodes)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        calibration_episodes=args.calibration_episodes,
        learning_episodes=args.learning_episodes,
        horizon=args.horizon,
        window_episodes=args.window_episodes,
    )
    rows, trace = [], []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        seed_rows, seed_trace = run_seed(seed, config)
        rows.extend(seed_rows)
        trace.extend(seed_trace)
        print(f"seed {seed} complete", flush=True)
    curve, summary = aggregate(rows, trace, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "learning_trace.csv", trace)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
