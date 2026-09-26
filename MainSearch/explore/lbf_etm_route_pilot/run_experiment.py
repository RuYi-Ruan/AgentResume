"""Minimal dynamic-vs-frozen ETM route test in a real three-agent LBF env."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import gymnasium as gym
import lbforaging  # noqa: F401
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MainSearch.explore.lbf_gradual_learning_gate.run_experiment import (
    MOVE_ACTIONS,
    adjacent_slots,
    food_positions,
    shortest_path,
)


AGENTS = ("A", "B", "C")
CONDITIONS = ("no_model", "frozen_etm", "dynamic_etm")
N_AGENTS = 3
N_TARGETS = 3
FOOD_REQUIREMENTS = np.array([1, 2, 3], dtype=np.int64)


@dataclass(frozen=True)
class Config:
    seeds: int = 30
    episodes: int = 900
    window: int = 100
    warmup: int = 150
    learning_rate: float = 0.025
    epsilon_start: float = 0.85
    epsilon_end: float = 0.08
    epsilon_decay_episodes: int = 800
    etm_decay: float = 0.99
    etm_prior: float = 1.0
    skill_weight: float = 0.30
    commitment_weight: float = 0.25
    environment_id: str = "Foraging-8x8-3p-3f-v3"
    permutation_samples: int = 20000


def softmax(values: np.ndarray, temperature: float = 0.5) -> np.ndarray:
    logits = values / temperature
    probs = np.exp(logits - logits.max())
    return probs / probs.sum()


def epsilon_at(episode: int, config: Config) -> float:
    fraction = min(episode / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def choose_proposal(
    q_values: np.ndarray, epsilon: float, explore_u: float, action_u: float
) -> int:
    if explore_u < epsilon:
        return min(int(action_u * N_TARGETS), N_TARGETS - 1)
    best = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(best[min(int(action_u * len(best)), len(best) - 1)])


class ETM:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.counts = {
            (observer, teammate): np.full(N_TARGETS, config.etm_prior)
            for observer in AGENTS
            for teammate in AGENTS
            if observer != teammate
        }

    def distribution(self, observer: str, teammate: str) -> np.ndarray:
        values = self.counts[(observer, teammate)]
        return values / values.sum()

    def update(self, observer: str, teammate: str, target: int) -> None:
        values = self.counts[(observer, teammate)]
        values *= self.config.etm_decay
        values += (1.0 - self.config.etm_decay) * self.config.etm_prior
        values[target] += 1.0


def expected_collection_value(
    observer_index: int,
    own_target: int,
    teammate_distributions: Sequence[np.ndarray],
) -> float:
    teammates = [index for index in range(N_AGENTS) if index != observer_index]
    expected = 0.0
    for first_target in range(N_TARGETS):
        for second_target in range(N_TARGETS):
            targets = [None] * N_AGENTS
            targets[observer_index] = own_target
            targets[teammates[0]] = first_target
            targets[teammates[1]] = second_target
            value = sum(
                requirement
                for target, requirement in enumerate(FOOD_REQUIREMENTS)
                if targets.count(target) >= requirement
            ) / float(FOOD_REQUIREMENTS.sum())
            expected += (
                teammate_distributions[0][first_target]
                * teammate_distributions[1][second_target]
                * value
            )
    return expected


def select_final_targets(
    condition: str,
    q_values: np.ndarray,
    proposals: Sequence[int],
    model: ETM,
    tie_values: Sequence[float],
    config: Config,
) -> Tuple[List[int], float]:
    final_targets = []
    correct = 0
    total = 0
    for observer_index, observer in enumerate(AGENTS):
        distributions = []
        for teammate_index, teammate in enumerate(AGENTS):
            if observer == teammate:
                continue
            if condition == "no_model":
                distribution = np.full(N_TARGETS, 1.0 / N_TARGETS)
            else:
                distribution = model.distribution(observer, teammate)
            distributions.append(distribution)
            correct += int(np.argmax(distribution) == proposals[teammate_index])
            total += 1
        own_skill = softmax(q_values[observer_index])
        utilities = np.array(
            [
                expected_collection_value(observer_index, target, distributions)
                + config.skill_weight * own_skill[target]
                + config.commitment_weight * float(target == proposals[observer_index])
                for target in range(N_TARGETS)
            ]
        )
        best = np.flatnonzero(np.isclose(utilities, utilities.max()))
        final_targets.append(
            int(best[min(int(tie_values[observer_index] * len(best)), len(best) - 1)])
        )
    return final_targets, correct / total


def configure_episode(env: gym.Env) -> List[Tuple[int, int]]:
    unwrapped = env.unwrapped
    foods = food_positions(unwrapped.field)
    for player in unwrapped.players:
        player.level = 1
    for rank, position in enumerate(foods):
        unwrapped.field[position] = int(FOOD_REQUIREMENTS[rank])
    return foods


def plan_subset(
    starts: Sequence[Tuple[int, int]],
    agent_indices: Sequence[int],
    target: Tuple[int, int],
    field: np.ndarray,
) -> Optional[List[Tuple[int, ...]]]:
    slots = adjacent_slots(target, field)
    if len(slots) < len(agent_indices):
        return None
    for assigned_slots in itertools.permutations(slots, len(agent_indices)):
        for order in itertools.permutations(range(len(agent_indices))):
            positions = list(starts)
            actions: List[Tuple[int, ...]] = []
            valid = True
            for local_index in order:
                agent_index = agent_indices[local_index]
                blocked = set(food_positions(field)) | {
                    positions[index] for index in range(N_AGENTS) if index != agent_index
                }
                path = shortest_path(
                    positions[agent_index], assigned_slots[local_index], blocked, field.shape
                )
                if path is None:
                    valid = False
                    break
                for action_id in path:
                    delta = next(delta for delta, value in MOVE_ACTIONS.items() if value == action_id)
                    positions[agent_index] = (
                        positions[agent_index][0] + delta[0],
                        positions[agent_index][1] + delta[1],
                    )
                    joint = [0] * N_AGENTS
                    joint[agent_index] = action_id
                    actions.append(tuple(joint))
            if valid:
                return actions
    return None


def execute_targets(
    env: gym.Env,
    foods: Sequence[Tuple[int, int]],
    targets: Sequence[int],
) -> dict:
    collected_value = 0
    planner_attempts = 0
    planner_successes = 0
    terminated = False
    truncated = False
    for target_rank, target_position in enumerate(foods):
        coalition = [index for index, selected in enumerate(targets) if selected == target_rank]
        requirement = int(FOOD_REQUIREMENTS[target_rank])
        if len(coalition) < requirement or int(env.unwrapped.field[target_position]) == 0:
            continue
        planner_attempts += 1
        starts = [tuple(player.position) for player in env.unwrapped.players]
        plan = plan_subset(starts, coalition, target_position, env.unwrapped.field)
        if plan is None:
            continue
        planner_successes += 1
        for actions in plan:
            _, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                break
        if terminated or truncated:
            break
        load_actions = tuple(5 if index in coalition else 0 for index in range(N_AGENTS))
        env.step(load_actions)
        if int(env.unwrapped.field[target_position]) == 0:
            collected_value += requirement
    return {
        "team_reward": collected_value / float(FOOD_REQUIREMENTS.sum()),
        "collected_value": collected_value,
        "planner_attempts": planner_attempts,
        "planner_successes": planner_successes,
    }


def exogenous(seed: int, config: Config) -> Mapping[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "explore": rng.random((config.episodes, N_AGENTS)),
        "action": rng.random((config.episodes, N_AGENTS)),
        "tie": rng.random((config.episodes, N_AGENTS)),
    }


def run_condition(
    seed: int,
    condition: str,
    config: Config,
    random_values: Mapping[str, np.ndarray],
) -> List[dict]:
    q_values = np.zeros((N_AGENTS, N_TARGETS))
    model = ETM(config)
    env = gym.make(config.environment_id, disable_env_checker=True)
    rows = []
    try:
        for episode in range(config.episodes):
            env.reset(seed=seed * 1_000_000 + episode)
            foods = configure_episode(env)
            epsilon = epsilon_at(episode, config)
            proposals = [
                choose_proposal(
                    q_values[index],
                    epsilon,
                    float(random_values["explore"][episode, index]),
                    float(random_values["action"][episode, index]),
                )
                for index in range(N_AGENTS)
            ]
            final_targets, intent_accuracy = select_final_targets(
                condition,
                q_values,
                proposals,
                model,
                random_values["tie"][episode],
                config,
            )
            outcome = execute_targets(env, foods, final_targets)
            reward = float(outcome["team_reward"])
            for index, target in enumerate(final_targets):
                old = q_values[index, target]
                q_values[index, target] = old + config.learning_rate * (reward - old)
            should_update = condition == "dynamic_etm" or (
                condition == "frozen_etm" and episode < config.warmup
            )
            if should_update:
                for observer in AGENTS:
                    for teammate_index, teammate in enumerate(AGENTS):
                        if observer != teammate:
                            model.update(observer, teammate, proposals[teammate_index])
            coalition_sizes = [final_targets.count(target) for target in range(N_TARGETS)]
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "condition": condition,
                    "team_reward": reward,
                    "intent_accuracy": intent_accuracy,
                    "collected_value": outcome["collected_value"],
                    "distinct_targets": len(set(final_targets)),
                    "largest_coalition": max(coalition_sizes),
                    "planner_attempts": outcome["planner_attempts"],
                    "planner_successes": outcome["planner_successes"],
                }
            )
    finally:
        env.close()
    return rows


def run_seed(seed: int, config: Config) -> List[dict]:
    random_values = exogenous(seed, config)
    rows = []
    for condition in CONDITIONS:
        rows.extend(run_condition(seed, condition, config, random_values))
    return rows


def aggregate_windows(rows: Sequence[dict], config: Config) -> List[dict]:
    buckets: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        start = (int(row["episode"]) // config.window) * config.window
        buckets.setdefault((start, str(row["condition"])), []).append(row)
    output = []
    for (start, condition), values in sorted(buckets.items()):
        attempts = sum(row["planner_attempts"] for row in values)
        successes = sum(row["planner_successes"] for row in values)
        output.append(
            {
                "window_start": start,
                "window_end": min(start + config.window, config.episodes) - 1,
                "condition": condition,
                "episodes": len(values),
                "team_reward": float(np.mean([row["team_reward"] for row in values])),
                "intent_accuracy": float(np.mean([row["intent_accuracy"] for row in values])),
                "distinct_targets": float(np.mean([row["distinct_targets"] for row in values])),
                "planner_reliability": successes / attempts if attempts else 1.0,
            }
        )
    return output


def late_seed_values(
    rows: Sequence[dict], config: Config, condition: str, metric: str
) -> np.ndarray:
    start = config.episodes - 3 * config.window
    by_seed: Dict[int, List[float]] = {}
    for row in rows:
        if row["condition"] == condition and row["episode"] >= start:
            by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return np.array([np.mean(by_seed[seed]) for seed in sorted(by_seed)])


def sign_flip_p(differences: np.ndarray, samples: int) -> float:
    observed = abs(float(differences.mean()))
    rng = np.random.default_rng(20260922)
    exceed = 0
    for _ in range(0, samples, 1000):
        size = min(1000, samples)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        exceed += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed))
    return (exceed + 1.0) / (samples + 1.0)


def compare(rows: Sequence[dict], config: Config, metric: str) -> dict:
    dynamic = late_seed_values(rows, config, "dynamic_etm", metric)
    frozen = late_seed_values(rows, config, "frozen_etm", metric)
    differences = dynamic - frozen
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    return {
        "metric": metric,
        "comparison": "dynamic_etm - frozen_etm",
        "late_episodes": 3 * config.window,
        "paired_seeds": len(differences),
        "mean_difference": mean,
        "normal_approx_95ci": [mean - 1.96 * se, mean + 1.96 * se],
        "two_sided_sign_flip_p": sign_flip_p(differences, config.permutation_samples),
    }


def run_experiment(config: Config) -> Tuple[List[dict], dict]:
    rows = []
    for seed in range(config.seeds):
        rows.extend(run_seed(seed, config))
    windows = aggregate_windows(rows, config)
    return windows, {
        "status": "development_route_test",
        "claim_boundary": "Minimal LBF ETM route test; not a formal benchmark result.",
        "config": asdict(config),
        "condition_episodes": len(rows),
        "late_comparisons": [
            compare(rows, config, "intent_accuracy"),
            compare(rows, config, "team_reward"),
        ],
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    config = Config(seeds=args.seeds, episodes=args.episodes)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
    windows, summary = run_experiment(config)
    write_csv(output_dir / "learning_curve.csv", windows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
