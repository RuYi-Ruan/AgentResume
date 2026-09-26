"""Gate 2: separate adaptation to teammate change from additional training."""

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

from MainSearch.explore.gate1_observable_etm_revalidation.run_experiment import episode_starts
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


MODEL_NAMES = ("frozen", "evolving_dynamic", "stale_budget", "static_dynamic")
EVAL_CONDITIONS = (
    "evolving_frozen",
    "evolving_dynamic",
    "evolving_stale_budget",
    "static_frozen",
    "static_dynamic",
)


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
    hidden_mode: str = "episodic"


def process_sequence(
    model_set,
    optimizer_set,
    hidden_set,
    summaries,
    events,
    update_weights,
    seed,
    phase,
    episode,
    condition,
    hidden_mode,
    record=True,
):
    rows = []
    updates = 0
    working_hidden = (
        {pair: torch.zeros_like(hidden_set[pair]) for pair in PAIRS}
        if hidden_mode == "episodic"
        else hidden_set
    )
    class_weights = torch.tensor([0.25, 1, 1, 1, 1, 1], dtype=torch.float32)
    for window in range(len(summaries) - 1):
        for pair in PAIRS:
            value = torch.tensor(summaries[window][pair], dtype=torch.float32).unsqueeze(0)
            logits, next_hidden = model_set[pair].step(
                value,
                working_hidden[pair],
                update_hidden=(hidden_mode == "episodic" or update_weights),
            )
            label = events[window + 1][pair[1]]
            prediction = int(logits.detach().argmax())
            if record:
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
            if hidden_mode == "episodic" or update_weights:
                working_hidden[pair] = next_hidden.detach()
            if update_weights:
                loss = nn.functional.cross_entropy(
                    logits,
                    torch.tensor([label], dtype=torch.long),
                    reduction="none",
                ) * class_weights[label]
                optimizer_set[pair].zero_grad()
                loss.mean().backward()
                optimizer_set[pair].step()
                updates += 1
    return rows, updates


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
        name: {pair: copy.deepcopy(base_models[pair]) for pair in PAIRS}
        for name in MODEL_NAMES
    }
    optimizers = {
        name: {
            pair: torch.optim.Adam(model.parameters(), lr=config.etm_learning_rate)
            for pair, model in models[name].items()
        }
        for name in MODEL_NAMES
    }
    hidden = {
        name: {pair: torch.zeros(1, 32) for pair in PAIRS}
        for name in MODEL_NAMES
    }
    update_counts = {name: 0 for name in MODEL_NAMES}

    train_mdp, train_env = make_env(config.horizon)
    evolving_mdp, evolving_env = make_env(config.horizon)
    static_mdp, static_env = make_env(config.horizon)
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    initial_roles = {
        context: actor_actions(actors, [canonical[index] for index in context])
        for context in contexts
    }
    rows, trace = [], []

    for episode in range(config.calibration_episodes):
        episode_seed = 70_000_000 + seed * 100_000 + episode
        starts = episode_starts(canonical, episode_seed)
        roles = initial_roles[context_key(starts, canonical)]
        _, _, summaries, events = rollout_windows(
            static_mdp, static_env, roles, episode_seed, config
        )
        for name in MODEL_NAMES:
            condition = f"calibration_{name}"
            new_rows, updates = process_sequence(
                models[name], optimizers[name], hidden[name], summaries, events,
                True, seed, "calibration", episode, condition, config.hidden_mode
            )
            rows.extend(new_rows)
            update_counts[name] += updates

    for episode in range(config.learning_episodes):
        episode_seed = seed * 1_000_000 + episode
        starts = episode_starts(canonical, episode_seed)
        context_tuple = context_key(starts, canonical)
        context = context_to_index[context_tuple]
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
        soups, actual_starts, _, _ = rollout_windows(
            train_mdp, train_env, sampled_roles, episode_seed, config
        )
        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            soups - payoff_values[context, selected]
        ) / count
        teacher_roles = JOINT_ACTIONS[
            best_seen_action(payoff_values, payoff_counts, context)
        ]
        train_actors(actors, actor_optimizers, actual_starts, teacher_roles)
        evolving_roles = actor_actions(actors, starts)
        static_roles = initial_roles[context_tuple]
        _, _, evolving_summaries, evolving_events = rollout_windows(
            evolving_mdp, evolving_env, evolving_roles, episode_seed, config
        )
        _, _, static_summaries, static_events = rollout_windows(
            static_mdp, static_env, static_roles, episode_seed, config
        )

        specs = (
            ("frozen", evolving_summaries, evolving_events, False, "evolving_frozen"),
            ("evolving_dynamic", evolving_summaries, evolving_events, True, "evolving_dynamic"),
            ("stale_budget", evolving_summaries, evolving_events, False, "evolving_stale_budget"),
            ("frozen", static_summaries, static_events, False, "static_frozen"),
            ("static_dynamic", static_summaries, static_events, True, "static_dynamic"),
        )
        for name, summaries, events, update, condition in specs:
            new_rows, updates = process_sequence(
                models[name], optimizers[name], hidden[name], summaries, events,
                update, seed, "learning", episode, condition, config.hidden_mode
            )
            rows.extend(new_rows)
            update_counts[name] += updates

        # Same update count as evolving_dynamic, but evidence remains on the old policy.
        _, updates = process_sequence(
            models["stale_budget"],
            optimizers["stale_budget"],
            hidden["stale_budget"],
            static_summaries,
            static_events,
            True,
            seed,
            "budget_training",
            episode,
            "stale_budget_training",
            config.hidden_mode,
            record=False,
        )
        update_counts["stale_budget"] += updates
        trace.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": soups,
                "sampled_roles": "-".join(map(str, sampled_roles)),
                "teacher_roles": "-".join(map(str, teacher_roles)),
                "evolving_roles": "-".join(map(str, evolving_roles)),
                "static_roles": "-".join(map(str, static_roles)),
            }
        )
    update_rows = [
        {"seed": seed, "model": name, "gradient_updates": count}
        for name, count in update_counts.items()
    ]
    return rows, trace, update_rows


def summarize_values(rows, condition, start, end):
    values = [
        row for row in rows
        if row["phase"] == "learning" and row["condition"] == condition
        and start <= row["episode"] < end
    ]
    nonempty = [row for row in values if row["nonempty"]]
    return {
        "window_start": start,
        "window_end": end - 1,
        "condition": condition,
        "predictions": len(values),
        "nonempty_predictions": len(nonempty),
        "overall_accuracy": float(np.mean([row["correct"] for row in values])),
        "nonempty_accuracy": float(
            np.mean([row["correct"] for row in nonempty]) if nonempty else 0.0
        ),
    }


def interval(values):
    center = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return center, [center - 1.96 * se, center + 1.96 * se]


def aggregate(rows, trace, update_rows, config):
    curve = []
    for start in range(0, config.learning_episodes, config.window_episodes):
        end = min(start + config.window_episodes, config.learning_episodes)
        for condition in EVAL_CONDITIONS:
            curve.append(summarize_values(rows, condition, start, end))
    late_start = config.learning_episodes - config.window_episodes
    late = {
        row["condition"]: row for row in curve if row["window_start"] == late_start
    }
    seed_ids = sorted({row["seed"] for row in rows})
    contrasts = {
        "evolving_dynamic_minus_frozen": ("evolving_dynamic", "evolving_frozen"),
        "evolving_dynamic_minus_stale_budget": ("evolving_dynamic", "evolving_stale_budget"),
        "static_dynamic_minus_frozen": ("static_dynamic", "static_frozen"),
    }
    contrast_results = {}
    for label, (left, right) in contrasts.items():
        gains = []
        for seed in seed_ids:
            accuracies = {}
            for condition in (left, right):
                values = [
                    row for row in rows
                    if row["seed"] == seed and row["phase"] == "learning"
                    and row["condition"] == condition and row["episode"] >= late_start
                    and row["nonempty"]
                ]
                accuracies[condition] = float(np.mean([row["correct"] for row in values]))
            gains.append(accuracies[left] - accuracies[right])
        center, ci = interval(gains)
        contrast_results[label] = {
            "mean_gain": center,
            "normal_approx_95ci": ci,
            "positive_seeds": sum(value > 0 for value in gains),
            "per_seed": {str(seed): gain for seed, gain in zip(seed_ids, gains)},
        }

    change_specific_gains = []
    for seed in seed_ids:
        accuracy = {}
        for condition in (
            "evolving_dynamic", "evolving_frozen", "static_dynamic", "static_frozen"
        ):
            values = [
                row for row in rows
                if row["seed"] == seed and row["phase"] == "learning"
                and row["condition"] == condition and row["episode"] >= late_start
                and row["nonempty"]
            ]
            accuracy[condition] = float(np.mean([row["correct"] for row in values]))
        change_specific_gains.append(
            (accuracy["evolving_dynamic"] - accuracy["evolving_frozen"])
            - (accuracy["static_dynamic"] - accuracy["static_frozen"])
        )
    center, ci = interval(change_specific_gains)
    contrast_results["change_specific_difference_in_differences"] = {
        "mean_gain": center,
        "normal_approx_95ci": ci,
        "positive_seeds": sum(value > 0 for value in change_specific_gains),
        "per_seed": {
            str(seed): gain for seed, gain in zip(seed_ids, change_specific_gains)
        },
    }

    calibration_equal = True
    for seed in seed_ids:
        sequences = []
        for name in MODEL_NAMES:
            sequences.append([
                (r["prediction"], r["correct"]) for r in rows
                if r["seed"] == seed and r["phase"] == "calibration"
                and r["condition"] == f"calibration_{name}"
            ])
        calibration_equal &= all(sequence == sequences[0] for sequence in sequences[1:])
    update_equal = all(
        next(r["gradient_updates"] for r in update_rows if r["seed"] == seed and r["model"] == "evolving_dynamic")
        == next(r["gradient_updates"] for r in update_rows if r["seed"] == seed and r["model"] == "stale_budget")
        == next(r["gradient_updates"] for r in update_rows if r["seed"] == seed and r["model"] == "static_dynamic")
        for seed in seed_ids
    )
    return curve, {
        "status": "development_gate2_change_control",
        "claim_boundary": "Cognition-only change-vs-training-budget control with scripted low-level execution; exploratory only.",
        "config": asdict(config),
        "prediction_events": len(rows),
        "learning_trace_episodes": len(trace),
        "calibration_all_models_exact_match": calibration_equal,
        "post_calibration_update_budgets_equal": update_equal,
        "late_metrics": late,
        "contrasts": contrast_results,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    # Tiny recurrent models are faster and more reproducible without BLAS thread
    # oversubscription when seeds are launched as separate processes.
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=Config.seed_start)
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--calibration-episodes", type=int, default=Config.calibration_episodes)
    parser.add_argument("--learning-episodes", type=int, default=Config.learning_episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--window-episodes", type=int, default=Config.window_episodes)
    parser.add_argument(
        "--hidden-mode",
        choices=("episodic", "persistent"),
        default=Config.hidden_mode,
        help="episodic gives every model normal within-episode inference; persistent reproduces the original diagnostic.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        calibration_episodes=args.calibration_episodes,
        learning_episodes=args.learning_episodes,
        horizon=args.horizon,
        window_episodes=args.window_episodes,
        hidden_mode=args.hidden_mode,
    )
    rows, trace, updates = [], [], []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        seed_rows, seed_trace, seed_updates = run_seed(seed, config)
        rows.extend(seed_rows)
        trace.extend(seed_trace)
        updates.extend(seed_updates)
        print(f"seed {seed} complete", flush=True)
    curve, summary = aggregate(rows, trace, updates, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "learning_trace.csv", trace)
    write_csv(args.output_dir / "update_counts.csv", updates)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
