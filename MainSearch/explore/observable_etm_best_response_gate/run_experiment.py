"""Rotating best-response gate for observable evolving teammate models."""

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

from MainSearch.explore.observable_etm_closed_loop.run_experiment import (
    CONDITIONS,
    ETM_CONDITIONS,
    CoordinationMLP,
    coordination_input,
    interval,
    update_etm,
    write_csv,
)
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


def choose_response(controller, responder, starts, proposals, beliefs, condition):
    value = coordination_input(
        responder, starts, proposals[responder], beliefs, condition
    )
    return int(controller(value).detach().argmax()), value


def train_response(controller, optimizer, value, target):
    logits = controller(value).unsqueeze(0)
    loss = torch.nn.functional.cross_entropy(
        logits, torch.tensor([target], dtype=torch.long)
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def oracle_best_response(envs, proposals, responder, episode_seed, config):
    outcomes = []
    for role, (mdp, env) in enumerate(envs):
        candidate = list(proposals)
        candidate[responder] = role
        soups, _, _, _ = rollout_windows(
            mdp, env, tuple(candidate), episode_seed, config
        )
        outcomes.append(soups)
    best = max(outcomes)
    candidates = [role for role, soups in enumerate(outcomes) if soups == best]
    target = proposals[responder] if proposals[responder] in candidates else min(candidates)
    return target, best, outcomes


def run_seed(seed, config):
    torch.manual_seed(110_000 + seed)
    rng = random.Random(120_000 + seed)
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
    oracle_envs = [make_env(config.horizon) for _ in range(3)]
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
        actor_teacher = JOINT_ACTIONS[
            best_seen_action(payoff_values, payoff_counts, context)
        ]
        train_actors(actors, actor_optimizers, training_starts, actor_teacher)
        proposals = actor_actions(actors, starts)

        responder = episode % 3
        response_target, oracle_soups, candidate_soups = oracle_best_response(
            oracle_envs, proposals, responder, episode_seed, config
        )
        for condition in CONDITIONS:
            current_beliefs = beliefs.get(condition, {})
            response, controller_input = choose_response(
                controllers[condition][responder],
                responder,
                starts,
                proposals,
                current_beliefs,
                condition,
            )
            chosen_roles = list(proposals)
            chosen_roles[responder] = response
            mdp, env = condition_envs[condition]
            soups, _, summaries, events = rollout_windows(
                mdp, env, tuple(chosen_roles), episode_seed, config
            )
            episode_rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "condition": condition,
                    "responder": responder,
                    "soups": soups,
                    "oracle_soups": oracle_soups,
                    "regret": oracle_soups - soups,
                    "response_correct": int(response == response_target),
                    "response": response,
                    "response_target": response_target,
                    "proposals": "-".join(map(str, proposals)),
                    "candidate_soups": "-".join(map(str, candidate_soups)),
                }
            )
            train_response(
                controllers[condition][responder],
                controller_optimizers[condition][responder],
                controller_input,
                response_target,
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


def aggregate(episode_rows, prediction_rows, config):
    curve = []
    for start in range(0, config.episodes, config.window_episodes):
        for condition in CONDITIONS:
            values = [
                row for row in episode_rows
                if row["condition"] == condition
                and start <= row["episode"] < start + config.window_episodes
            ]
            predictions = [
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
                    "mean_regret": float(np.mean([row["regret"] for row in values])),
                    "response_accuracy": float(np.mean([row["response_correct"] for row in values])),
                    "nonempty_etm_events": len(predictions),
                    "nonempty_etm_accuracy": (
                        float(np.mean([row["correct"] for row in predictions]))
                        if predictions else None
                    ),
                }
            )

    seed_ids = sorted({row["seed"] for row in episode_rows})
    late_start = config.episodes - config.window_episodes
    per_seed, dynamic_frozen, dynamic_no_etm = {}, [], []
    for seed in seed_ids:
        means = {}
        for condition in CONDITIONS:
            values = [
                row["soups"] for row in episode_rows
                if row["seed"] == seed and row["condition"] == condition
                and row["episode"] >= late_start
            ]
            means[condition] = float(np.mean(values))
        per_seed[str(seed)] = means
        dynamic_frozen.append(means["dynamic_etm"] - means["frozen_etm"])
        dynamic_no_etm.append(means["dynamic_etm"] - means["no_etm"])
    df_gain, df_ci = interval(dynamic_frozen)
    dn_gain, dn_ci = interval(dynamic_no_etm)
    late = {
        row["condition"]: row for row in curve if row["window_start"] == late_start
    }
    prefreeze_equal = all(
        a["soups"] == b["soups"] and a["response"] == b["response"]
        for a, b in zip(
            [row for row in episode_rows if row["condition"] == "frozen_etm" and row["episode"] < config.freeze_after],
            [row for row in episode_rows if row["condition"] == "dynamic_etm" and row["episode"] < config.freeze_after],
        )
    )
    return curve, {
        "status": "development_observable_etm_best_response_gate",
        "claim_boundary": "Rotating learned best response with environment-derived training targets and scripted low-level execution; exploratory only.",
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
