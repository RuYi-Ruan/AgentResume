"""Development gate for six independently evolving directed teammate models."""

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

from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import (
    MOVE,
    RoleAgent,
    make_env,
)
from MainSearch.explore.overcooked_three_mlp_ctde_gate.run_experiment import (
    JOINT_ACTIONS,
    actor_actions,
    best_seen_action,
    context_key,
    joint_index,
    train_actors,
)
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import (
    ACTIONS,
    PolicyMLP,
    observation,
)


PAIRS = tuple((observer, target) for observer in range(3) for target in range(3) if observer != target)


@dataclass(frozen=True)
class Config:
    seeds: int = 3
    episodes: int = 240
    horizon: int = 160
    window: int = 40
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.02
    frozen_after: int = 60
    visibility_radius: int = 4


class TeammateMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 12), nn.Tanh(), nn.Linear(12, 3))

    def forward(self, value):
        return self.net(value)


def teammate_input(position):
    return torch.tensor([position[0] / 12.0, position[1] / 6.0], dtype=torch.float32)


def play_with_visibility(mdp, env, action_ids, episode_seed, horizon, radius):
    env.reset(regen_mdp=False)
    rng = random.Random(episode_seed)
    starts = list(mdp.start_player_positions)
    rng.shuffle(starts)
    for player, position in zip(env.state.players, starts):
        player.update_pos_and_or(position, rng.choice(MOVE))
    agents = [RoleAgent(mdp, i, *ACTIONS[action]) for i, action in enumerate(action_ids)]
    visible = {pair: False for pair in PAIRS}
    total_reward = 0
    for _ in range(horizon):
        atomic = tuple(agent.action(env.state) for agent in agents)
        positions = [tuple(player.position) for player in env.state.players]
        for observer, target in PAIRS:
            distance = abs(positions[observer][0] - positions[target][0]) + abs(
                positions[observer][1] - positions[target][1]
            )
            if atomic[target] == "interact" and distance <= radius:
                visible[(observer, target)] = True
        _, reward, done, _ = env.step(atomic, joint_agent_action_info=[{}, {}, {}])
        total_reward += reward
        if done:
            break
    return total_reward / 20.0, starts, visible


def predict(models, starts):
    return {
        pair: int(model(teammate_input(starts[pair[1]])).detach().argmax())
        for pair, model in models.items()
    }


def update_model(model, optimizer, position, label):
    logits = model(teammate_input(position)).unsqueeze(0)
    target = torch.tensor([label], dtype=torch.long)
    loss = nn.functional.cross_entropy(logits, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def observer_disagreement(predictions):
    differences = []
    for target in range(3):
        observers = [observer for observer in range(3) if observer != target]
        differences.append(
            int(predictions[(observers[0], target)] != predictions[(observers[1], target)])
        )
    return float(np.mean(differences))


def run_seed(seed, config):
    torch.manual_seed(10_000 + seed)
    actor_rng = random.Random(20_000 + seed)
    actors = [PolicyMLP() for _ in range(3)]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate) for actor in actors
    ]
    dynamic_models = {pair: TeammateMLP() for pair in PAIRS}
    frozen_models = {pair: copy.deepcopy(dynamic_models[pair]) for pair in PAIRS}
    dynamic_optimizers = {
        pair: torch.optim.Adam(model.parameters(), lr=config.etm_learning_rate)
        for pair, model in dynamic_models.items()
    }
    frozen_optimizers = {
        pair: torch.optim.Adam(model.parameters(), lr=config.etm_learning_rate)
        for pair, model in frozen_models.items()
    }
    update_counts = {pair: 0 for pair in PAIRS}

    mdp, env = make_env(config.horizon)
    canonical = list(mdp.start_player_positions)
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
        context = context_to_index[context_key(starts, canonical)]
        untried = np.flatnonzero(payoff_counts[context] == 0)
        fraction = episode / max(config.episodes - 1, 1)
        epsilon = 1.0 + fraction * (0.10 - 1.0)
        if len(untried):
            selected = int(actor_rng.choice(untried.tolist()))
        elif actor_rng.random() < epsilon:
            selected = actor_rng.randrange(len(JOINT_ACTIONS))
        else:
            selected = joint_index(actor_actions(actors, starts))
        intentions = JOINT_ACTIONS[selected]

        dynamic_predictions = predict(dynamic_models, starts)
        frozen_predictions = predict(frozen_models, starts)
        soups, actual_starts, visible = play_with_visibility(
            mdp, env, intentions, episode_seed, config.horizon, config.visibility_radius
        )

        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            soups - payoff_values[context, selected]
        ) / count
        teacher_index = best_seen_action(payoff_values, payoff_counts, context)
        train_actors(
            actors, actor_optimizers, actual_starts, JOINT_ACTIONS[teacher_index]
        )

        for pair in PAIRS:
            if visible[pair]:
                label = intentions[pair[1]]
                update_model(
                    dynamic_models[pair],
                    dynamic_optimizers[pair],
                    actual_starts[pair[1]],
                    label,
                )
                update_counts[pair] += 1
                if episode < config.frozen_after:
                    update_model(
                        frozen_models[pair],
                        frozen_optimizers[pair],
                        actual_starts[pair[1]],
                        label,
                    )

        rows.append(
            {
                "seed": seed,
                "episode": episode,
                "soups": soups,
                "dynamic_correct": sum(
                    dynamic_predictions[pair] == intentions[pair[1]] for pair in PAIRS
                ),
                "frozen_correct": sum(
                    frozen_predictions[pair] == intentions[pair[1]] for pair in PAIRS
                ),
                "prediction_events": len(PAIRS),
                "dynamic_disagreement": observer_disagreement(dynamic_predictions),
                "frozen_disagreement": observer_disagreement(frozen_predictions),
                "observed_pairs": sum(visible.values()),
            }
        )
    return rows, update_counts


def aggregate(rows, counts_by_seed, config):
    curve = []
    for start in range(0, config.episodes, config.window):
        values = [row for row in rows if start <= row["episode"] < start + config.window]
        events = sum(row["prediction_events"] for row in values)
        curve.append(
            {
                "window_start": start,
                "window_end": min(start + config.window, config.episodes) - 1,
                "prediction_events": events,
                "dynamic_accuracy": sum(row["dynamic_correct"] for row in values) / events,
                "frozen_accuracy": sum(row["frozen_correct"] for row in values) / events,
                "dynamic_disagreement": float(
                    np.mean([row["dynamic_disagreement"] for row in values])
                ),
                "mean_observed_pairs": float(np.mean([row["observed_pairs"] for row in values])),
            }
        )

    late_start = config.episodes - config.window
    gains = []
    for seed in range(config.seeds):
        values = [row for row in rows if row["seed"] == seed and row["episode"] >= late_start]
        events = sum(row["prediction_events"] for row in values)
        gains.append(
            (
                sum(row["dynamic_correct"] for row in values)
                - sum(row["frozen_correct"] for row in values)
            )
            / events
        )
    se = float(np.std(gains, ddof=1) / math.sqrt(len(gains))) if len(gains) > 1 else 0.0
    all_counts = [count for counts in counts_by_seed for count in counts.values()]
    summary = {
        "status": "development_directional_etm_gate",
        "claim_boundary": "ETMs predict high-level role intentions but do not yet control policies; fixed map and scripted low-level execution.",
        "config": asdict(config),
        "prediction_events": len(rows) * len(PAIRS),
        "late_dynamic_accuracy": curve[-1]["dynamic_accuracy"],
        "late_frozen_accuracy": curve[-1]["frozen_accuracy"],
        "paired_seed_accuracy_gain": float(np.mean(gains)),
        "normal_approx_95ci": [
            float(np.mean(gains) - 1.96 * se),
            float(np.mean(gains) + 1.96 * se),
        ],
        "late_observer_disagreement": curve[-1]["dynamic_disagreement"],
        "directed_update_count_range": [min(all_counts), max(all_counts)],
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
    rows, counts = [], []
    for seed in range(config.seeds):
        seed_rows, seed_counts = run_seed(seed, config)
        rows.extend(seed_rows)
        counts.append(seed_counts)
        print(f"seed {seed + 1}/{config.seeds} complete", flush=True)
    curve, summary = aggregate(rows, counts, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "events.csv", rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
