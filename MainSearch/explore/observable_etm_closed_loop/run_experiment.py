"""Closed-loop gate for observable, label-free evolving teammate models."""

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
    EVENTS,
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


CONDITIONS = ("no_etm", "frozen_etm", "dynamic_etm")
ETM_CONDITIONS = ("frozen_etm", "dynamic_etm")


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seeds: int = 5
    episodes: int = 240
    horizon: int = 120
    macro_steps: int = 10
    window_episodes: int = 40
    freeze_after: int = 60
    visibility_radius: int = 4
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.01
    controller_learning_rate: float = 0.01
    controller_mode: str = "continuous"


class CoordinationMLP(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = 2 + 3 + 2 * len(EVENTS)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 3)
        )

    def forward(self, value):
        return self.net(value)


def coordination_input(agent, starts, proposal, beliefs, condition):
    own_position = torch.tensor(
        [starts[agent][0] / 12.0, starts[agent][1] / 6.0], dtype=torch.float32
    )
    own_proposal = torch.nn.functional.one_hot(
        torch.tensor(proposal), num_classes=3
    ).float()
    partner_values = []
    for target in range(3):
        if target == agent:
            continue
        if condition == "no_etm":
            partner_values.extend([0.0] * len(EVENTS))
        else:
            partner_values.extend(beliefs[(agent, target)].tolist())
    return torch.cat(
        [own_position, own_proposal, torch.tensor(partner_values, dtype=torch.float32)]
    )


def choose_roles(controllers, starts, proposals, beliefs, condition):
    inputs, roles = [], []
    for agent in range(3):
        value = coordination_input(agent, starts, proposals[agent], beliefs, condition)
        inputs.append(value)
        roles.append(int(controllers[agent](value).detach().argmax()))
    return tuple(roles), inputs


def train_controllers(controllers, optimizers, inputs, targets):
    for agent in range(3):
        logits = controllers[agent](inputs[agent]).unsqueeze(0)
        loss = nn.functional.cross_entropy(
            logits, torch.tensor([targets[agent]], dtype=torch.long)
        )
        optimizers[agent].zero_grad()
        loss.backward()
        optimizers[agent].step()


def update_etm(models, optimizers, hidden, summaries, events, can_update):
    rows = []
    last_beliefs = {pair: np.full(len(EVENTS), 1 / len(EVENTS)) for pair in PAIRS}
    class_weights = torch.tensor([0.25, 1, 1, 1, 1, 1], dtype=torch.float32)
    for window in range(len(summaries) - 1):
        logits_by_pair, next_hidden = {}, {}
        for pair in PAIRS:
            value = torch.tensor(summaries[window][pair], dtype=torch.float32).unsqueeze(0)
            logits, state = models.step(value, hidden[pair], update_hidden=can_update)
            label = events[window + 1][pair[1]]
            prediction = int(logits.detach().argmax())
            logits_by_pair[pair] = logits
            next_hidden[pair] = state
            last_beliefs[pair] = torch.softmax(logits.detach(), dim=-1).squeeze(0).numpy()
            rows.append(
                {
                    "window": window,
                    "pair": f"{pair[0]}->{pair[1]}",
                    "label": label,
                    "prediction": prediction,
                    "correct": int(prediction == label),
                    "nonempty": int(label != 0),
                }
            )
        if can_update:
            losses = []
            for pair in PAIRS:
                label = events[window + 1][pair[1]]
                loss = nn.functional.cross_entropy(
                    logits_by_pair[pair],
                    torch.tensor([label], dtype=torch.long),
                    reduction="none",
                ) * class_weights[label]
                losses.append(loss)
            loss = torch.stack(losses).mean()
            optimizers.zero_grad()
            loss.backward()
            optimizers.step()
            hidden.update({pair: next_hidden[pair].detach() for pair in PAIRS})
    return rows, last_beliefs


def run_seed(seed, config):
    torch.manual_seed(90_000 + seed)
    rng = random.Random(100_000 + seed)

    actors = [PolicyMLP() for _ in range(3)]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate)
        for actor in actors
    ]
    base_controllers = [CoordinationMLP() for _ in range(3)]
    controllers = {
        condition: [copy.deepcopy(model) for model in base_controllers]
        for condition in CONDITIONS
    }
    controller_optimizers = {
        condition: [
            torch.optim.Adam(model.parameters(), lr=config.controller_learning_rate)
            for model in controllers[condition]
        ]
        for condition in CONDITIONS
    }
    base_etm = ObservableETM()
    etms = {condition: copy.deepcopy(base_etm) for condition in ETM_CONDITIONS}
    etm_optimizers = {
        condition: torch.optim.Adam(etms[condition].parameters(), lr=config.etm_learning_rate)
        for condition in ETM_CONDITIONS
    }
    hidden = {
        condition: {pair: torch.zeros(1, 32) for pair in PAIRS}
        for condition in ETM_CONDITIONS
    }
    beliefs = {
        condition: {pair: np.full(len(EVENTS), 1 / len(EVENTS)) for pair in PAIRS}
        for condition in ETM_CONDITIONS
    }

    train_mdp, train_env = make_env(config.horizon)
    condition_envs = {condition: make_env(config.horizon) for condition in CONDITIONS}
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    episode_rows, prediction_rows = [], []

    for episode in range(config.episodes):
        episode_seed = seed * 1_000_000 + episode
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        context = context_to_index[context_key(starts, canonical)]

        proposals = actor_actions(actors, starts)
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
        training_soups, training_starts, _, _ = rollout_windows(
            train_mdp, train_env, training_roles, episode_seed, config
        )
        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            training_soups - payoff_values[context, selected]
        ) / count
        teacher_roles = JOINT_ACTIONS[best_seen_action(payoff_values, payoff_counts, context)]
        train_actors(actors, actor_optimizers, training_starts, teacher_roles)
        proposals = actor_actions(actors, starts)

        for condition in CONDITIONS:
            current_beliefs = beliefs.get(condition, {})
            chosen_roles, controller_inputs = choose_roles(
                controllers[condition], starts, proposals, current_beliefs, condition
            )
            mdp, env = condition_envs[condition]
            soups, _, summaries, events = rollout_windows(
                mdp, env, chosen_roles, episode_seed, config
            )
            episode_rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "condition": condition,
                    "soups": soups,
                    "teacher_role_correct": sum(
                        int(chosen_roles[i] == teacher_roles[i]) for i in range(3)
                    ),
                    "chosen_roles": "-".join(map(str, chosen_roles)),
                    "teacher_roles": "-".join(map(str, teacher_roles)),
                    "proposals": "-".join(map(str, proposals)),
                }
            )
            if config.controller_mode == "continuous" or episode < config.freeze_after:
                train_controllers(
                    controllers[condition],
                    controller_optimizers[condition],
                    controller_inputs,
                    teacher_roles,
                )
            if condition in ETM_CONDITIONS:
                can_update = condition == "dynamic_etm" or episode < config.freeze_after
                rows, next_beliefs = update_etm(
                    etms[condition],
                    etm_optimizers[condition],
                    hidden[condition],
                    summaries,
                    events,
                    can_update,
                )
                beliefs[condition] = next_beliefs
                for row in rows:
                    row.update({"seed": seed, "episode": episode, "condition": condition})
                prediction_rows.extend(rows)
    return episode_rows, prediction_rows


def interval(values):
    center = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return center, [center - 1.96 * se, center + 1.96 * se]


def aggregate(episode_rows, prediction_rows, config):
    curve = []
    for start in range(0, config.episodes, config.window_episodes):
        for condition in CONDITIONS:
            values = [
                row for row in episode_rows
                if row["condition"] == condition
                and start <= row["episode"] < start + config.window_episodes
            ]
            prediction_values = [
                row for row in prediction_rows
                if row["condition"] == condition
                and start <= row["episode"] < start + config.window_episodes
                and row["nonempty"]
            ]
            curve.append(
                {
                    "window_start": start,
                    "window_end": min(start + config.window_episodes, config.episodes) - 1,
                    "condition": condition,
                    "games": len(values),
                    "mean_soups": float(np.mean([row["soups"] for row in values])),
                    "teacher_role_accuracy": sum(
                        row["teacher_role_correct"] for row in values
                    ) / (3 * len(values)),
                    "nonempty_etm_events": len(prediction_values),
                    "nonempty_etm_accuracy": (
                        float(np.mean([row["correct"] for row in prediction_values]))
                        if prediction_values else None
                    ),
                }
            )

    seed_ids = sorted({row["seed"] for row in episode_rows})
    late_start = config.episodes - config.window_episodes
    per_seed = {}
    dynamic_frozen, dynamic_no_etm = [], []
    for seed in seed_ids:
        means = {}
        for condition in CONDITIONS:
            values = [
                row["soups"] for row in episode_rows
                if row["seed"] == seed and row["condition"] == condition
                and row["episode"] >= late_start
            ]
            means[condition] = float(np.mean(values))
        dynamic_frozen.append(means["dynamic_etm"] - means["frozen_etm"])
        dynamic_no_etm.append(means["dynamic_etm"] - means["no_etm"])
        per_seed[str(seed)] = means
    df_gain, df_ci = interval(dynamic_frozen)
    dn_gain, dn_ci = interval(dynamic_no_etm)
    late = {}
    for condition in CONDITIONS:
        values = [
            row for row in episode_rows
            if row["condition"] == condition and row["episode"] >= late_start
        ]
        prediction_values = [
            row for row in prediction_rows
            if row["condition"] == condition and row["episode"] >= late_start
            and row["nonempty"]
        ]
        late[condition] = {
            "window_start": late_start,
            "window_end": config.episodes - 1,
            "condition": condition,
            "games": len(values),
            "mean_soups": float(np.mean([row["soups"] for row in values])),
            "teacher_role_accuracy": sum(
                row["teacher_role_correct"] for row in values
            ) / (3 * len(values)),
            "nonempty_etm_events": len(prediction_values),
            "nonempty_etm_accuracy": (
                float(np.mean([row["correct"] for row in prediction_values]))
                if prediction_values else None
            ),
        }
    prefreeze_equal = all(
        a["soups"] == b["soups"] and a["chosen_roles"] == b["chosen_roles"]
        for a, b in zip(
            [row for row in episode_rows if row["condition"] == "frozen_etm" and row["episode"] < config.freeze_after],
            [row for row in episode_rows if row["condition"] == "dynamic_etm" and row["episode"] < config.freeze_after],
        )
    )
    return curve, {
        "status": "development_observable_etm_closed_loop_gate",
        "claim_boundary": "Learned high-level coordination with scripted low-level execution; exploratory, not a formal result.",
        "config": asdict(config),
        "condition_games": len(episode_rows),
        "prediction_events": len(prediction_rows),
        "prefreeze_dynamic_frozen_exact_match": prefreeze_equal,
        "late_metrics": late,
        "late_soups_by_seed": per_seed,
        "dynamic_minus_frozen_soups": df_gain,
        "dynamic_minus_frozen_normal_approx_95ci": df_ci,
        "dynamic_better_frozen_seeds": sum(value > 0 for value in dynamic_frozen),
        "dynamic_minus_no_etm_soups": dn_gain,
        "dynamic_minus_no_etm_normal_approx_95ci": dn_ci,
        "dynamic_better_no_etm_seeds": sum(value > 0 for value in dynamic_no_etm),
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
    parser.add_argument(
        "--controller-mode",
        choices=("continuous", "frozen"),
        default=Config.controller_mode,
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        episodes=args.episodes,
        horizon=args.horizon,
        controller_mode=args.controller_mode,
    )
    episode_rows, prediction_rows = [], []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        episodes, predictions = run_seed(seed, config)
        episode_rows.extend(episodes)
        prediction_rows.extend(predictions)
        print(f"seed {seed} complete", flush=True)
    curve, summary = aggregate(episode_rows, prediction_rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "episodes.csv", episode_rows)
    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
