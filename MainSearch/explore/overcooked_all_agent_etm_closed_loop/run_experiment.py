"""Development gate: do six dynamic ETMs improve real Overcooked throughput?"""

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

from MainSearch.explore.overcooked_directional_etm_gate.run_experiment import (
    PAIRS,
    play_with_visibility,
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
    seeds: int = 3
    episodes: int = 200
    horizon: int = 140
    window: int = 40
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.02
    frozen_after: int = 50
    visibility_radius: int = 4


class ClosedLoopETM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 12), nn.Tanh(), nn.Linear(12, 3))

    def forward(self, value):
        return self.net(value)


def etm_input(position, target_is_responder):
    return torch.tensor(
        [position[0] / 12.0, position[1] / 6.0, float(target_is_responder)],
        dtype=torch.float32,
    )


def predictions(models, starts, responder):
    return {
        pair: int(
            model(etm_input(starts[pair[1]], pair[1] == responder)).detach().argmax()
        )
        for pair, model in models.items()
    }


def update_etm(model, optimizer, position, target_is_responder, label):
    logits = model(etm_input(position, target_is_responder)).unsqueeze(0)
    loss = nn.functional.cross_entropy(logits, torch.tensor([label], dtype=torch.long))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def coordinated_roles(proposals, responder, predicted):
    others = [index for index in range(3) if index != responder]
    predicted_roles = [predicted[(responder, target)] for target in others]
    candidates = []
    for role in range(3):
        coverage = len(set(predicted_roles + [role]))
        commitment = int(role == proposals[responder])
        candidates.append((coverage, commitment, -role, role))
    selected = max(candidates)[-1]
    final = list(proposals)
    final[responder] = selected
    return tuple(final)


def run_seed(seed, config):
    torch.manual_seed(50_000 + seed)
    rng = random.Random(60_000 + seed)
    actors = [PolicyMLP() for _ in range(3)]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate) for actor in actors
    ]
    base_models = {pair: ClosedLoopETM() for pair in PAIRS}
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

    train_mdp, train_env = make_env(config.horizon)
    condition_envs = {condition: make_env(config.horizon) for condition in CONDITIONS}
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    rows = []

    for episode in range(config.episodes):
        episode_seed = seed * 1_000_000 + episode
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        responder = episode % 3
        proposals = actor_actions(actors, starts)

        # One shared training rollout updates the underlying capability trajectory.
        context = context_to_index[context_key(starts, canonical)]
        untried = np.flatnonzero(payoff_counts[context] == 0)
        fraction = episode / max(config.episodes - 1, 1)
        epsilon = 1.0 + fraction * (0.10 - 1.0)
        if len(untried):
            selected = int(rng.choice(untried.tolist()))
        elif rng.random() < epsilon:
            selected = rng.randrange(len(JOINT_ACTIONS))
        else:
            selected = joint_index(proposals)
        training_roles = JOINT_ACTIONS[selected]
        training_soups, training_starts, _ = play_with_visibility(
            train_mdp,
            train_env,
            training_roles,
            episode_seed,
            config.horizon,
            config.visibility_radius,
        )
        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            training_soups - payoff_values[context, selected]
        ) / count
        teacher_index = best_seen_action(payoff_values, payoff_counts, context)
        train_actors(
            actors,
            actor_optimizers,
            training_starts,
            JOINT_ACTIONS[teacher_index],
        )

        for condition in CONDITIONS:
            predicted = predictions(models[condition], starts, responder)
            final_roles = coordinated_roles(proposals, responder, predicted)
            mdp, env = condition_envs[condition]
            soups, actual_starts, visible = play_with_visibility(
                mdp,
                env,
                final_roles,
                episode_seed,
                config.horizon,
                config.visibility_radius,
            )
            correct = sum(predicted[pair] == final_roles[pair[1]] for pair in PAIRS)
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "condition": condition,
                    "soups": soups,
                    "intent_correct": correct,
                    "prediction_events": len(PAIRS),
                    "role_changed": int(final_roles[responder] != proposals[responder]),
                }
            )
            should_update = condition == "dynamic_etm" or episode < config.frozen_after
            if should_update:
                for pair in PAIRS:
                    if visible[pair]:
                        update_etm(
                            models[condition][pair],
                            optimizers[condition][pair],
                            actual_starts[pair[1]],
                            pair[1] == responder,
                            final_roles[pair[1]],
                        )
    return rows


def aggregate(rows, config):
    curve = []
    for start in range(0, config.episodes, config.window):
        for condition in CONDITIONS:
            values = [
                row
                for row in rows
                if row["condition"] == condition
                and start <= row["episode"] < start + config.window
            ]
            events = sum(row["prediction_events"] for row in values)
            curve.append(
                {
                    "window_start": start,
                    "window_end": min(start + config.window, config.episodes) - 1,
                    "condition": condition,
                    "games": len(values),
                    "mean_soups": float(np.mean([row["soups"] for row in values])),
                    "intent_accuracy": sum(row["intent_correct"] for row in values) / events,
                    "role_change_rate": float(
                        np.mean([row["role_changed"] for row in values])
                    ),
                }
            )

    late_start = config.episodes - config.window
    soup_gains, intent_gains = [], []
    for seed in range(config.seeds):
        by_condition = {}
        for condition in CONDITIONS:
            values = [
                row for row in rows
                if row["seed"] == seed
                and row["condition"] == condition
                and row["episode"] >= late_start
            ]
            by_condition[condition] = {
                "soups": float(np.mean([row["soups"] for row in values])),
                "intent": sum(row["intent_correct"] for row in values)
                / sum(row["prediction_events"] for row in values),
            }
        soup_gains.append(
            by_condition["dynamic_etm"]["soups"] - by_condition["frozen_etm"]["soups"]
        )
        intent_gains.append(
            by_condition["dynamic_etm"]["intent"]
            - by_condition["frozen_etm"]["intent"]
        )

    def interval(values):
        se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        center = float(np.mean(values))
        return center, [center - 1.96 * se, center + 1.96 * se]

    soup_gain, soup_ci = interval(soup_gains)
    intent_gain, intent_ci = interval(intent_gains)
    late = {
        row["condition"]: row
        for row in curve
        if row["window_start"] == late_start
    }
    summary = {
        "status": "development_all_agent_closed_loop_gate",
        "claim_boundary": "All agents rotate through an ETM-based responder role; policy learning trajectory is paired and exogenous to ETM condition; not a formal result.",
        "config": asdict(config),
        "condition_games": len(rows),
        "late_metrics": late,
        "paired_seed_soup_gain": soup_gain,
        "soup_gain_normal_approx_95ci": soup_ci,
        "paired_seed_intent_gain": intent_gain,
        "intent_gain_normal_approx_95ci": intent_ci,
        "dynamic_better_soup_seeds": sum(value > 0 for value in soup_gains),
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(seeds=args.seeds, episodes=args.episodes, horizon=args.horizon)
    rows = []
    for seed in range(config.seeds):
        rows.extend(run_seed(seed, config))
        print(f"seed {seed + 1}/{config.seeds} complete", flush=True)
    curve, summary = aggregate(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
