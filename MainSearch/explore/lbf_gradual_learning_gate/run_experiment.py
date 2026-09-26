"""Feasibility gate for gradual three-agent learning in real LBF dynamics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import lbforaging  # noqa: F401 - importing registers the environments
import numpy as np


N_AGENTS = 3
N_TARGETS = 3
MOVE_ACTIONS = {
    (-1, 0): 1,  # north
    (1, 0): 2,   # south
    (0, -1): 3,  # west
    (0, 1): 4,   # east
}


@dataclass(frozen=True)
class Config:
    seeds: int = 30
    episodes: int = 800
    window: int = 100
    learning_rate: float = 0.025
    epsilon_start: float = 0.90
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 700
    planner_step_limit: int = 50
    environment_id: str = "Foraging-8x8-3p-3f-coop-v3"


def epsilon_at(episode: int, config: Config) -> float:
    fraction = min(episode / config.epsilon_decay_episodes, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def choose_target(q_values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(N_TARGETS))
    best = np.flatnonzero(np.isclose(q_values, q_values.max()))
    return int(rng.choice(best))


def food_positions(field: np.ndarray) -> List[Tuple[int, int]]:
    return sorted((int(row), int(col)) for row, col in zip(*np.nonzero(field)))


def adjacent_slots(
    target: Tuple[int, int], field: np.ndarray
) -> List[Tuple[int, int]]:
    height, width = field.shape
    candidates = [
        (target[0] - 1, target[1]),
        (target[0] + 1, target[1]),
        (target[0], target[1] - 1),
        (target[0], target[1] + 1),
    ]
    return [
        pos
        for pos in candidates
        if 0 <= pos[0] < height
        and 0 <= pos[1] < width
        and int(field[pos]) == 0
    ]


def shortest_path(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    blocked: set[Tuple[int, int]],
    shape: Tuple[int, int],
) -> Optional[List[int]]:
    if start == goal:
        return []
    queue = [start]
    cursor = 0
    parents: Dict[Tuple[int, int], Tuple[Tuple[int, int], int]] = {}
    seen = {start}
    while cursor < len(queue):
        current = queue[cursor]
        cursor += 1
        for delta, action_id in MOVE_ACTIONS.items():
            nxt = (current[0] + delta[0], current[1] + delta[1])
            if not (0 <= nxt[0] < shape[0] and 0 <= nxt[1] < shape[1]):
                continue
            if nxt in blocked or nxt in seen:
                continue
            parents[nxt] = (current, action_id)
            if nxt == goal:
                actions: List[int] = []
                node = nxt
                while node != start:
                    previous, action = parents[node]
                    actions.append(action)
                    node = previous
                actions.reverse()
                return actions
            seen.add(nxt)
            queue.append(nxt)
    return None


def plan_joint_path(
    starts: Sequence[Tuple[int, int]],
    target: Tuple[int, int],
    field: np.ndarray,
) -> Optional[List[Tuple[int, ...]]]:
    """Fast deterministic planner moving agents one at a time to distinct slots."""
    slots = adjacent_slots(target, field)
    if len(slots) < N_AGENTS:
        return None
    static_obstacles = set(food_positions(field))
    slot_assignments = sorted(
        itertools.permutations(slots, N_AGENTS),
        key=lambda assignment: sum(
            abs(starts[index][0] - assignment[index][0])
            + abs(starts[index][1] - assignment[index][1])
            for index in range(N_AGENTS)
        ),
    )
    for assignment in slot_assignments:
        for order in itertools.permutations(range(N_AGENTS)):
            positions = list(starts)
            joint_actions: List[Tuple[int, ...]] = []
            valid = True
            for agent_index in order:
                blocked = static_obstacles | {
                    positions[index]
                    for index in range(N_AGENTS)
                    if index != agent_index
                }
                path = shortest_path(
                    positions[agent_index], assignment[agent_index], blocked, field.shape
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
                    action = [0] * N_AGENTS
                    action[agent_index] = action_id
                    joint_actions.append(tuple(action))
            if valid:
                return joint_actions
    return None


def execute_aligned_episode(env: gym.Env, target_rank: int, config: Config) -> dict:
    unwrapped = env.unwrapped
    foods = food_positions(unwrapped.field)
    target = foods[target_rank]
    starts = [tuple(player.position) for player in unwrapped.players]
    plan = plan_joint_path(starts, target, unwrapped.field)
    if plan is None or len(plan) + 1 > config.planner_step_limit:
        return {"reward": 0.0, "collected": 0, "planner_ok": 0, "steps": 0}

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    for actions in plan:
        _, rewards, terminated, truncated, _ = env.step(actions)
        total_reward += float(sum(rewards))
        steps += 1
        if terminated or truncated:
            break
    if not (terminated or truncated):
        _, rewards, terminated, truncated, _ = env.step((5, 5, 5))
        total_reward += float(sum(rewards))
        steps += 1
    collected = int(int(unwrapped.field[target]) == 0 and total_reward > 0)
    return {
        "reward": total_reward,
        "collected": collected,
        "planner_ok": 1,
        "steps": steps,
    }


def run_seed(seed: int, config: Config) -> List[dict]:
    rng = np.random.default_rng(seed)
    q_values = np.zeros((N_AGENTS, N_TARGETS), dtype=np.float64)
    rows: List[dict] = []
    env = gym.make(config.environment_id, disable_env_checker=True)
    try:
        for episode in range(config.episodes):
            env.reset(seed=seed * 1_000_000 + episode)
            epsilon = epsilon_at(episode, config)
            targets = [
                choose_target(q_values[index], epsilon, rng)
                for index in range(N_AGENTS)
            ]
            aligned = len(set(targets)) == 1
            outcome = {
                "reward": 0.0,
                "collected": 0,
                "planner_ok": 1,
                "steps": 0,
            }
            if aligned:
                outcome = execute_aligned_episode(env, targets[0], config)

            shared_reward = float(outcome["collected"])
            for agent_index, target in enumerate(targets):
                old = q_values[agent_index, target]
                q_values[agent_index, target] = old + config.learning_rate * (
                    shared_reward - old
                )
            rows.append(
                {
                    "seed": seed,
                    "episode": episode,
                    "epsilon": epsilon,
                    "intent_aligned": int(aligned),
                    "team_success": int(outcome["collected"]),
                    "planner_ok": int(outcome["planner_ok"]),
                    "environment_reward": float(outcome["reward"]),
                    "steps": int(outcome["steps"]),
                    "target_A": targets[0],
                    "target_B": targets[1],
                    "target_C": targets[2],
                }
            )
    finally:
        env.close()
    return rows


def aggregate_windows(rows: Sequence[dict], config: Config) -> List[dict]:
    buckets: Dict[int, List[dict]] = {}
    for row in rows:
        start = (int(row["episode"]) // config.window) * config.window
        buckets.setdefault(start, []).append(row)
    output = []
    for start, values in sorted(buckets.items()):
        aligned = [row for row in values if row["intent_aligned"]]
        output.append(
            {
                "window_start": start,
                "window_end": min(start + config.window, config.episodes) - 1,
                "episodes": len(values),
                "intent_alignment_rate": float(
                    np.mean([row["intent_aligned"] for row in values])
                ),
                "team_success_rate": float(
                    np.mean([row["team_success"] for row in values])
                ),
                "mean_environment_reward": float(
                    np.mean([row["environment_reward"] for row in values])
                ),
                "planner_success_given_alignment": float(
                    np.mean([row["team_success"] for row in aligned])
                )
                if aligned
                else 0.0,
            }
        )
    return output


def seed_improvements(rows: Sequence[dict], config: Config) -> np.ndarray:
    by_seed: Dict[int, Dict[str, List[int]]] = {}
    for row in rows:
        phase = "early" if row["episode"] < config.window else "late"
        if row["episode"] >= config.episodes - config.window:
            phase = "late"
        elif phase != "early":
            continue
        by_seed.setdefault(int(row["seed"]), {"early": [], "late": []})[phase].append(
            int(row["team_success"])
        )
    return np.array(
        [
            np.mean(by_seed[seed]["late"]) - np.mean(by_seed[seed]["early"])
            for seed in sorted(by_seed)
        ]
    )


def summarize(rows: Sequence[dict], windows: Sequence[dict], config: Config) -> dict:
    improvement = seed_improvements(rows, config)
    early = windows[0]
    late = windows[-1]
    aligned_rows = [row for row in rows if row["intent_aligned"]]
    adjacent_changes = np.diff([row["team_success_rate"] for row in windows])
    gate = {
        "late_success_at_least_0_50": late["team_success_rate"] >= 0.50,
        "mean_improvement_at_least_0_25": float(improvement.mean()) >= 0.25,
        "at_least_80pct_seeds_improve": float(np.mean(improvement > 0.10)) >= 0.80,
        "planner_reliability_at_least_0_98": float(
            np.mean([row["team_success"] for row in aligned_rows])
        )
        >= 0.98,
    }
    return {
        "status": "development_gate",
        "claim_boundary": "LBF online-learning feasibility only; ETM is not tested.",
        "config": asdict(config),
        "episodes": len(rows),
        "intent_labels": {
            "target_food": "canonical rank of selected food position",
            "intended_collaborators": "the other two fixed-identity agents",
        },
        "level_leakage": (
            "The raw LBF observation exposes player and food levels; mask them before ETM evaluation."
        ),
        "early_window": early,
        "late_window": late,
        "mean_seed_improvement": float(improvement.mean()),
        "improvement_95ci": [
            float(improvement.mean() - 1.96 * improvement.std(ddof=1) / math.sqrt(len(improvement))),
            float(improvement.mean() + 1.96 * improvement.std(ddof=1) / math.sqrt(len(improvement))),
        ],
        "fraction_seeds_improving_over_0_10": float(np.mean(improvement > 0.10)),
        "planner_success_given_alignment": float(
            np.mean([row["team_success"] for row in aligned_rows])
        ),
        "largest_adjacent_window_change": float(np.max(np.abs(adjacent_changes))),
        "gate_checks": gate,
        "gate_pass": bool(all(gate.values())),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def run_experiment(config: Config) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    for seed in range(config.seeds):
        rows.extend(run_seed(seed, config))
    windows = aggregate_windows(rows, config)
    return windows, summarize(rows, windows, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seeds=args.seeds, episodes=args.episodes)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
    windows, summary = run_experiment(config)
    write_csv(output_dir / "learning_curve.csv", windows)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
