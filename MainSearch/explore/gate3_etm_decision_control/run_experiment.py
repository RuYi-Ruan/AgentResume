"""Gate 3: test whether observable ETM updates reduce real coordination loss."""

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
)


MODEL_NAMES = ("frozen", "dynamic", "stale_budget")
CONDITIONS = ("base", "frozen", "dynamic", "stale_budget", "oracle")


@dataclass(frozen=True)
class Config:
    seed_start: int = 0
    seeds: int = 5
    calibration_episodes: int = 120
    learning_episodes: int = 360
    horizon: int = 180
    macro_steps: int = 10
    probe_windows: int = 3
    eval_interval: int = 60
    eval_episodes: int = 6
    table_repeats: int = 2
    visibility_radius: int = 4
    actor_learning_rate: float = 0.01
    etm_learning_rate: float = 0.01
    exploration_end: float = 0.10


def reset_to_seed(mdp, env, episode_seed):
    env.reset(regen_mdp=False)
    rng = random.Random(episode_seed)
    starts = list(mdp.start_player_positions)
    rng.shuffle(starts)
    for player, position in zip(env.state.players, starts):
        player.update_pos_and_or(position, rng.choice(MOVE))
    return starts


def rollout_intervention(
    mdp, env, base_roles, focal, selected_role, episode_seed, horizon, switch_step
):
    reset_to_seed(mdp, env, episode_seed)
    base_agents = [RoleAgent(mdp, i, *ACTIONS[role]) for i, role in enumerate(base_roles)]
    replacement = RoleAgent(mdp, focal, *ACTIONS[selected_role])
    total_reward = 0
    for tick in range(horizon):
        actions = []
        for index, agent in enumerate(base_agents):
            active = replacement if index == focal and tick >= switch_step else agent
            actions.append(active.action(env.state))
        _, reward, done, _ = env.step(tuple(actions), joint_agent_action_info=[{}, {}, {}])
        total_reward += reward
        if done:
            break
    return total_reward / 20.0


def rollout_intervention_fixed_context(
    mdp,
    env,
    base_roles,
    focal,
    selected_role,
    context,
    orientation_seed,
    horizon,
    switch_step,
):
    env.reset(regen_mdp=False)
    canonical = list(mdp.start_player_positions)
    rng = random.Random(orientation_seed)
    for player, start_index in zip(env.state.players, context):
        player.update_pos_and_or(canonical[start_index], rng.choice(MOVE))
    agents = [RoleAgent(mdp, i, *ACTIONS[role]) for i, role in enumerate(base_roles)]
    replacement = RoleAgent(mdp, focal, *ACTIONS[selected_role])
    total_reward = 0
    for tick in range(horizon):
        actions = []
        for index, agent in enumerate(agents):
            active = replacement if index == focal and tick >= switch_step else agent
            actions.append(active.action(env.state))
        _, reward, done, _ = env.step(tuple(actions), joint_agent_action_info=[{}, {}, {}])
        total_reward += reward
        if done:
            break
    return total_reward / 20.0


def build_best_response_table(mdp, env, config):
    table = {}
    contexts = list(itertools.permutations(range(3)))
    switch_step = config.probe_windows * config.macro_steps
    for context_index, context in enumerate(contexts):
        for focal in range(3):
            targets = [index for index in range(3) if index != focal]
            for base_focal_role in range(3):
                for partner_roles in itertools.product(range(3), repeat=2):
                    means = []
                    base_roles = [0, 0, 0]
                    base_roles[focal] = base_focal_role
                    for target, role in zip(targets, partner_roles):
                        base_roles[target] = role
                    for candidate in range(3):
                        scores = [
                            rollout_intervention_fixed_context(
                                mdp,
                                env,
                                base_roles,
                                focal,
                                candidate,
                                context,
                                91_000_000 + context_index * 100_000 + focal * 10_000
                                + base_focal_role * 1_000 + partner_roles[0] * 100
                                + partner_roles[1] * 10 + repeat,
                                config.horizon,
                                switch_step,
                            )
                            for repeat in range(config.table_repeats)
                        ]
                        means.append(float(np.mean(scores)))
                    table[(context, focal, base_focal_role, *partner_roles)] = int(
                        np.flatnonzero(np.isclose(means, max(means)))[0]
                    )
    return table


def update_model_set(models, optimizers, summaries, events, update_weights):
    hidden = {pair: torch.zeros(1, 32) for pair in PAIRS}
    class_weights = torch.tensor([0.25, 1, 1, 1, 1, 1], dtype=torch.float32)
    updates = 0
    for window in range(len(summaries) - 1):
        for pair in PAIRS:
            value = torch.tensor(summaries[window][pair], dtype=torch.float32).unsqueeze(0)
            logits, next_hidden = models[pair].step(value, hidden[pair], update_hidden=True)
            hidden[pair] = next_hidden.detach()
            if update_weights:
                label = events[window + 1][pair[1]]
                loss = nn.functional.cross_entropy(
                    logits,
                    torch.tensor([label], dtype=torch.long),
                    reduction="none",
                ) * class_weights[label]
                optimizers[pair].zero_grad()
                loss.mean().backward()
                optimizers[pair].step()
                updates += 1
    return updates


def event_distribution_to_role(probabilities):
    # Observable semantic events: none, onion, left pot, right pot, dish, serve.
    scores = torch.stack(
        (
            probabilities[1] + 2.0 * probabilities[2],
            probabilities[1] + 2.0 * probabilities[3],
            probabilities[4] + probabilities[5]
            + 0.5 * (probabilities[2] + probabilities[3]),
        )
    )
    return int(scores.argmax())


def infer_partner_roles(models, summaries, focal, probe_windows):
    targets = [index for index in range(3) if index != focal]
    inferred = []
    for target in targets:
        pair = (focal, target)
        hidden = torch.zeros(1, 32)
        logits = None
        for summary in summaries[:probe_windows]:
            value = torch.tensor(summary[pair], dtype=torch.float32).unsqueeze(0)
            logits, hidden = models[pair].step(value, hidden, update_hidden=True)
            hidden = hidden.detach()
        probabilities = torch.softmax(logits.detach().squeeze(0), dim=-1)
        inferred.append(event_distribution_to_role(probabilities))
    return tuple(inferred)


def evaluate_checkpoint(
    seed, checkpoint, actors, models, table, mdp, observe_env, score_env, canonical, config
):
    rows = []
    switch_step = config.probe_windows * config.macro_steps
    for offset in range(config.eval_episodes):
        episode_seed = 83_000_000 + seed * 100_000 + checkpoint * 100 + offset
        starts = episode_starts(canonical, episode_seed)
        context = context_key(starts, canonical)
        base_roles = actor_actions(actors, starts)
        _, _, summaries, _ = rollout_windows(
            mdp, observe_env, base_roles, episode_seed, config
        )
        for focal in range(3):
            targets = [index for index in range(3) if index != focal]
            true_partner_roles = tuple(base_roles[index] for index in targets)
            oracle_role = table[(
                context, focal, base_roles[focal], *true_partner_roles
            )]
            selected = {"base": base_roles[focal], "oracle": oracle_role}
            inferred_by_condition = {}
            for condition in MODEL_NAMES:
                inferred = infer_partner_roles(
                    models[condition], summaries, focal, config.probe_windows
                )
                inferred_by_condition[condition] = inferred
                selected[condition] = table[(
                    context, focal, base_roles[focal], *inferred
                )]
            for condition in CONDITIONS:
                soups = rollout_intervention(
                    mdp,
                    score_env,
                    base_roles,
                    focal,
                    selected[condition],
                    episode_seed,
                    config.horizon,
                    switch_step,
                )
                inferred = inferred_by_condition.get(condition)
                rows.append(
                    {
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "eval_offset": offset,
                        "focal": focal,
                        "condition": condition,
                        "soups": soups,
                        "base_role": base_roles[focal],
                        "selected_role": selected[condition],
                        "oracle_role": oracle_role,
                        "decision_correct": int(selected[condition] == oracle_role),
                        "true_partner_roles": "-".join(map(str, true_partner_roles)),
                        "inferred_partner_roles": "" if inferred is None else "-".join(map(str, inferred)),
                    }
                )
    return rows


def run_seed(seed, config, table):
    torch.manual_seed(150_000 + seed)
    rng = random.Random(160_000 + seed)
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
    update_counts = {name: 0 for name in MODEL_NAMES}

    train_mdp, train_env = make_env(config.horizon)
    observe_mdp, observe_env = make_env(config.horizon)
    _, static_env = make_env(config.horizon)
    _, eval_observe_env = make_env(config.horizon)
    _, eval_score_env = make_env(config.horizon)
    canonical = list(train_mdp.start_player_positions)
    contexts = list(itertools.permutations(range(3)))
    context_to_index = {value: index for index, value in enumerate(contexts)}
    payoff_values = np.zeros((len(contexts), len(JOINT_ACTIONS)), dtype=np.float64)
    payoff_counts = np.zeros_like(payoff_values, dtype=np.int64)
    initial_roles = {
        context: actor_actions(actors, [canonical[index] for index in context])
        for context in contexts
    }

    for episode in range(config.calibration_episodes):
        episode_seed = 70_000_000 + seed * 100_000 + episode
        starts = episode_starts(canonical, episode_seed)
        roles = initial_roles[context_key(starts, canonical)]
        _, _, summaries, events = rollout_windows(
            observe_mdp, static_env, roles, episode_seed, config
        )
        for name in MODEL_NAMES:
            update_counts[name] += update_model_set(
                models[name], optimizers[name], summaries, events, True
            )

    rows = []
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
            selected_index = int(rng.choice(untried.tolist()))
        elif rng.random() < epsilon:
            selected_index = rng.randrange(len(JOINT_ACTIONS))
        else:
            selected_index = joint_index(proposals)
        sampled_roles = JOINT_ACTIONS[selected_index]
        soups, actual_starts, _, _ = rollout_windows(
            train_mdp, train_env, sampled_roles, episode_seed, config
        )
        payoff_counts[context, selected_index] += 1
        count = payoff_counts[context, selected_index]
        payoff_values[context, selected_index] += (
            soups - payoff_values[context, selected_index]
        ) / count
        teacher_roles = JOINT_ACTIONS[best_seen_action(payoff_values, payoff_counts, context)]
        train_actors(actors, actor_optimizers, actual_starts, teacher_roles)

        evolving_roles = actor_actions(actors, starts)
        static_roles = initial_roles[context_tuple]
        _, _, evolving_summaries, evolving_events = rollout_windows(
            observe_mdp, observe_env, evolving_roles, episode_seed, config
        )
        _, _, static_summaries, static_events = rollout_windows(
            observe_mdp, static_env, static_roles, episode_seed, config
        )
        update_counts["dynamic"] += update_model_set(
            models["dynamic"], optimizers["dynamic"],
            evolving_summaries, evolving_events, True
        )
        update_counts["stale_budget"] += update_model_set(
            models["stale_budget"], optimizers["stale_budget"],
            static_summaries, static_events, True
        )

        if (episode + 1) % config.eval_interval == 0:
            rows.extend(
                evaluate_checkpoint(
                    seed,
                    episode + 1,
                    actors,
                    models,
                    table,
                    observe_mdp,
                    eval_observe_env,
                    eval_score_env,
                    canonical,
                    config,
                )
            )
    return rows, [
        {"seed": seed, "model": name, "gradient_updates": count}
        for name, count in update_counts.items()
    ]


def interval(values):
    center = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return center, [center - 1.96 * se, center + 1.96 * se]


def aggregate(rows, updates, config):
    late = [row for row in rows if row["checkpoint"] == config.learning_episodes]
    condition_metrics = {}
    for condition in CONDITIONS:
        values = [row for row in late if row["condition"] == condition]
        condition_metrics[condition] = {
            "decisions": len(values),
            "mean_soups": float(np.mean([row["soups"] for row in values])),
            "oracle_decision_agreement": float(np.mean([row["decision_correct"] for row in values])),
        }
    all_condition_metrics = {}
    for condition in CONDITIONS:
        values = [row for row in rows if row["condition"] == condition]
        all_condition_metrics[condition] = {
            "decisions": len(values),
            "mean_soups": float(np.mean([row["soups"] for row in values])),
            "oracle_decision_agreement": float(
                np.mean([row["decision_correct"] for row in values])
            ),
        }
    contrasts = {}
    for label, left, right in (
        ("dynamic_minus_frozen", "dynamic", "frozen"),
        ("dynamic_minus_stale_budget", "dynamic", "stale_budget"),
        ("dynamic_minus_base", "dynamic", "base"),
    ):
        gains = []
        for seed in range(config.seed_start, config.seed_start + config.seeds):
            left_values = [r["soups"] for r in late if r["seed"] == seed and r["condition"] == left]
            right_values = [r["soups"] for r in late if r["seed"] == seed and r["condition"] == right]
            gains.append(float(np.mean(left_values) - np.mean(right_values)))
        center, ci = interval(gains)
        contrasts[label] = {
            "mean_soup_gain": center,
            "normal_approx_95ci": ci,
            "positive_seeds": sum(value > 0 for value in gains),
            "ties": sum(value == 0 for value in gains),
            "per_seed": {
                str(seed): gain
                for seed, gain in zip(
                    range(config.seed_start, config.seed_start + config.seeds), gains
                )
            },
        }
    all_checkpoint_contrasts = {}
    for label, left, right in (
        ("dynamic_minus_frozen", "dynamic", "frozen"),
        ("dynamic_minus_stale_budget", "dynamic", "stale_budget"),
        ("dynamic_minus_base", "dynamic", "base"),
    ):
        gains = []
        for seed in range(config.seed_start, config.seed_start + config.seeds):
            left_values = [r["soups"] for r in rows if r["seed"] == seed and r["condition"] == left]
            right_values = [r["soups"] for r in rows if r["seed"] == seed and r["condition"] == right]
            gains.append(float(np.mean(left_values) - np.mean(right_values)))
        center, ci = interval(gains)
        all_checkpoint_contrasts[label] = {
            "mean_soup_gain": center,
            "normal_approx_95ci": ci,
            "positive_seeds": sum(value > 0 for value in gains),
            "ties": sum(value == 0 for value in gains),
            "per_seed": {
                str(seed): gain
                for seed, gain in zip(
                    range(config.seed_start, config.seed_start + config.seeds), gains
                )
            },
        }
    return {
        "status": "development_gate3_decision_control",
        "claim_boundary": "Controlled ETM-to-decision test with scripted low-level execution; not a formal main result.",
        "config": asdict(config),
        "evaluation_rows": len(rows),
        "all_checkpoint_condition_metrics": all_condition_metrics,
        "all_checkpoint_contrasts": all_checkpoint_contrasts,
        "late_condition_metrics": condition_metrics,
        "late_contrasts": contrasts,
        "update_counts": updates,
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=Config.seed_start)
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--calibration-episodes", type=int, default=Config.calibration_episodes)
    parser.add_argument("--learning-episodes", type=int, default=Config.learning_episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--probe-windows", type=int, default=Config.probe_windows)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--table-repeats", type=int, default=Config.table_repeats)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        seed_start=args.seed_start,
        seeds=args.seeds,
        calibration_episodes=args.calibration_episodes,
        learning_episodes=args.learning_episodes,
        horizon=args.horizon,
        probe_windows=args.probe_windows,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        table_repeats=args.table_repeats,
    )
    table_mdp, table_env = make_env(config.horizon)
    table = build_best_response_table(table_mdp, table_env, config)
    rows, updates = [], []
    for seed in range(config.seed_start, config.seed_start + config.seeds):
        seed_rows, seed_updates = run_seed(seed, config, table)
        rows.extend(seed_rows)
        updates.extend(seed_updates)
        print(f"seed {seed} complete", flush=True)
    summary = aggregate(rows, updates, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "evaluations.csv", rows)
    write_csv(args.output_dir / "update_counts.csv", updates)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
