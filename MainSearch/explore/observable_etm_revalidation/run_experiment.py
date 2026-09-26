"""Revalidate dynamic ETM using only causal, observer-visible trajectory evidence."""

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

from MainSearch.explore.overcooked_directional_etm_gate.run_experiment import PAIRS
from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import MOVE, RoleAgent, make_env
from MainSearch.explore.overcooked_three_mlp_ctde_gate.run_experiment import (
    JOINT_ACTIONS,
    actor_actions,
    best_seen_action,
    context_key,
    joint_index,
    train_actors,
)
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import ACTIONS, PolicyMLP


EVENTS = ("none", "onion", "pot_left", "pot_right", "dish", "serve")
HELD = (None, "onion", "dish", "soup")
ATOMIC = ((0, 0), (0, -1), (0, 1), (1, 0), (-1, 0), "interact")
CONDITIONS = ("frozen_etm", "dynamic_etm")


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seeds: int = 5
    episodes: int = 240
    horizon: int = 120
    macro_steps: int = 10
    window_episodes: int = 40
    frozen_after: int = 60
    visibility_radius: int = 4
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.01


class ObservableETM(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, 32), nn.ReLU(), nn.Linear(32, len(EVENTS))
        )

    def step(self, summary, hidden, update_hidden=True):
        candidate = self.gru(summary, hidden)
        state = candidate if update_hidden else hidden
        return self.head(torch.cat([state, summary], dim=-1)), state


def held_kind(player):
    return None if not player.has_object() else player.get_object().name


def interaction_event(mdp, state, target, atomic_action):
    if atomic_action != "interact":
        return 0
    player = state.players[target]
    facing = (
        player.position[0] + player.orientation[0],
        player.position[1] + player.orientation[1],
    )
    if facing in mdp.get_onion_dispenser_locations():
        return 1
    pots = sorted(mdp.get_pot_locations())
    if facing in pots:
        return 2 + pots.index(facing)
    if facing in mdp.get_dish_dispenser_locations():
        return 4
    if facing in mdp.get_serving_locations():
        return 5
    return 0


def empty_accumulator():
    return {
        pair: {
            "visible": 0,
            "dx": 0.0,
            "dy": 0.0,
            "held": np.zeros(len(HELD), dtype=np.float32),
            "actions": np.zeros(len(ATOMIC), dtype=np.float32),
        }
        for pair in PAIRS
    }


def summarize(accumulator, macro_steps):
    output = {}
    for pair, value in accumulator.items():
        visible = value["visible"]
        denom = max(visible, 1)
        output[pair] = np.concatenate(
            [
                np.array(
                    [visible / macro_steps, value["dx"] / denom, value["dy"] / denom],
                    dtype=np.float32,
                ),
                value["held"] / denom,
                value["actions"] / denom,
            ]
        )
    return output


def rollout_windows(mdp, env, roles, episode_seed, config):
    env.reset(regen_mdp=False)
    rng = random.Random(episode_seed)
    starts = list(mdp.start_player_positions)
    rng.shuffle(starts)
    for player, position in zip(env.state.players, starts):
        player.update_pos_and_or(position, rng.choice(MOVE))
    agents = [RoleAgent(mdp, i, *ACTIONS[role]) for i, role in enumerate(roles)]
    summaries, events = [], []
    accumulator = empty_accumulator()
    window_events = {target: 0 for target in range(3)}
    total_reward = 0

    for tick in range(config.horizon):
        state = env.state
        atomic = tuple(agent.action(state) for agent in agents)
        positions = [tuple(player.position) for player in state.players]
        for observer, target in PAIRS:
            dx = positions[target][0] - positions[observer][0]
            dy = positions[target][1] - positions[observer][1]
            if abs(dx) + abs(dy) <= config.visibility_radius:
                value = accumulator[(observer, target)]
                value["visible"] += 1
                value["dx"] += dx / 12.0
                value["dy"] += dy / 6.0
                value["held"][HELD.index(held_kind(state.players[target]))] += 1
                value["actions"][ATOMIC.index(atomic[target])] += 1
        for target in range(3):
            event = interaction_event(mdp, state, target, atomic[target])
            if window_events[target] == 0 and event != 0:
                window_events[target] = event
        _, reward, done, _ = env.step(atomic, joint_agent_action_info=[{}, {}, {}])
        total_reward += reward
        if (tick + 1) % config.macro_steps == 0:
            summaries.append(summarize(accumulator, config.macro_steps))
            events.append(dict(window_events))
            accumulator = empty_accumulator()
            window_events = {target: 0 for target in range(3)}
        if done:
            break
    return total_reward / 20.0, starts, summaries, events


def evaluate_and_update(models, optimizers, hidden, summaries, events, episode, config):
    rows = []
    class_weights = torch.tensor([0.25, 1, 1, 1, 1, 1], dtype=torch.float32)
    for window in range(len(summaries) - 1):
        labels = {pair: events[window + 1][pair[1]] for pair in PAIRS}
        for condition in CONDITIONS:
            can_update = condition == "dynamic_etm" or episode < config.frozen_after
            logits_by_pair, next_hidden = {}, {}
            for pair in PAIRS:
                value = torch.tensor(summaries[window][pair], dtype=torch.float32).unsqueeze(0)
                logits, state = models[condition].step(
                    value, hidden[condition][pair], update_hidden=can_update
                )
                logits_by_pair[pair] = logits
                next_hidden[pair] = state
                prediction = int(logits.detach().argmax())
                label = labels[pair]
                rows.append(
                    {
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
                losses = [
                    nn.functional.cross_entropy(
                        logits_by_pair[pair],
                        torch.tensor([labels[pair]], dtype=torch.long),
                        reduction="none",
                    )
                    * class_weights[labels[pair]]
                    for pair in PAIRS
                ]
                loss = torch.stack(losses).mean()
                optimizers[condition].zero_grad()
                loss.backward()
                optimizers[condition].step()
                hidden[condition] = {pair: next_hidden[pair].detach() for pair in PAIRS}
    return rows


def run_seed(seed, config):
    torch.manual_seed(70_000 + seed)
    rng = random.Random(80_000 + seed)
    actors = [PolicyMLP() for _ in range(3)]
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=config.actor_learning_rate) for actor in actors
    ]
    base_model = ObservableETM()
    models = {condition: copy.deepcopy(base_model) for condition in CONDITIONS}
    optimizers = {
        condition: torch.optim.Adam(models[condition].parameters(), lr=config.etm_learning_rate)
        for condition in CONDITIONS
    }
    hidden = {
        condition: {pair: torch.zeros(1, 32) for pair in PAIRS} for condition in CONDITIONS
    }

    train_mdp, train_env = make_env(config.horizon)
    observe_mdp, observe_env = make_env(config.horizon)
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    prediction_rows = []

    for episode in range(config.episodes):
        episode_seed = seed * 1_000_000 + episode
        start_rng = random.Random(episode_seed)
        starts = canonical.copy()
        start_rng.shuffle(starts)
        context = context_to_index[context_key(starts, canonical)]

        # Separate exploratory rollout supplies real experience for gradual policy learning.
        untried = np.flatnonzero(payoff_counts[context] == 0)
        fraction = episode / max(config.episodes - 1, 1)
        epsilon = 1.0 + fraction * (0.10 - 1.0)
        if len(untried):
            selected = int(rng.choice(untried.tolist()))
        elif rng.random() < epsilon:
            selected = rng.randrange(len(JOINT_ACTIONS))
        else:
            selected = joint_index(actor_actions(actors, starts))
        training_roles = JOINT_ACTIONS[selected]
        training_soups, training_starts, _, _ = rollout_windows(
            train_mdp, train_env, training_roles, episode_seed, config
        )
        payoff_counts[context, selected] += 1
        count = payoff_counts[context, selected]
        payoff_values[context, selected] += (
            training_soups - payoff_values[context, selected]
        ) / count
        teacher_index = best_seen_action(payoff_values, payoff_counts, context)
        train_actors(
            actors, actor_optimizers, training_starts, JOINT_ACTIONS[teacher_index]
        )

        # ETMs observe a fresh greedy-policy rollout; no internal role is passed to them.
        observed_roles = actor_actions(actors, starts)
        _, _, summaries, events = rollout_windows(
            observe_mdp, observe_env, observed_roles, episode_seed, config
        )
        rows = evaluate_and_update(
            models, optimizers, hidden, summaries, events, episode, config
        )
        for row in rows:
            row.update({"seed": seed, "episode": episode})
        prediction_rows.extend(rows)
    return prediction_rows


def aggregate(rows, config):
    curve = []
    for start in range(0, config.episodes, config.window_episodes):
        for condition in CONDITIONS:
            values = [
                row
                for row in rows
                if row["condition"] == condition
                and start <= row["episode"] < start + config.window_episodes
            ]
            nonempty = [row for row in values if row["nonempty"]]
            curve.append(
                {
                    "window_start": start,
                    "window_end": min(start + config.window_episodes, config.episodes) - 1,
                    "condition": condition,
                    "predictions": len(values),
                    "nonempty_predictions": len(nonempty),
                    "overall_accuracy": float(np.mean([row["correct"] for row in values])),
                    "nonempty_accuracy": float(
                        np.mean([row["correct"] for row in nonempty]) if nonempty else 0.0
                    ),
                }
            )

    late_start = config.episodes - config.window_episodes
    gains = []
    seed_ids = sorted({row["seed"] for row in rows})
    for seed in seed_ids:
        accuracy = {}
        for condition in CONDITIONS:
            values = [
                row
                for row in rows
                if row["seed"] == seed
                and row["condition"] == condition
                and row["episode"] >= late_start
                and row["nonempty"]
            ]
            accuracy[condition] = float(np.mean([row["correct"] for row in values]))
        gains.append(accuracy["dynamic_etm"] - accuracy["frozen_etm"])
    se = float(np.std(gains, ddof=1) / math.sqrt(len(gains))) if len(gains) > 1 else 0.0
    center = float(np.mean(gains))
    late = {
        row["condition"]: row
        for row in curve
        if row["window_start"] == late_start
    }
    return curve, {
        "status": "development_observable_etm_revalidation",
        "claim_boundary": "Cognition-only revalidation with scripted low-level execution; ETM sees only causal local trajectory summaries and predicts future observable interaction events.",
        "config": asdict(config),
        "prediction_events": len(rows),
        "late_metrics": late,
        "paired_seed_nonempty_accuracy_gain": center,
        "paired_seed_nonempty_accuracy_gains": {
            str(seed): gain for seed, gain in zip(seed_ids, gains)
        },
        "normal_approx_95ci": [center - 1.96 * se, center + 1.96 * se],
        "dynamic_better_seeds": sum(gain > 0 for gain in gains),
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        episodes=args.episodes,
        horizon=args.horizon,
    )
    rows = []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        rows.extend(run_seed(seed, config))
        print(
            f"seed {seed} complete ({seed - config.seed_start + 1}/{config.seeds})",
            flush=True,
        )
    curve, summary = aggregate(rows, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "predictions.csv", rows)
    write_csv(args.output_dir / "learning_curve.csv", curve)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
