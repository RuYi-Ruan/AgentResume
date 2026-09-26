"""Closed-loop toy pilot for gradually co-learning agents with ETMs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


AGENTS = ("A", "B", "C")
CONDITIONS = ("no_model", "frozen_etm", "dynamic_etm", "oracle")
N_ROLES = 3
N_CONTEXTS = 6


@dataclass(frozen=True)
class Config:
    seeds: int = 100
    rounds: int = 1500
    window: int = 100
    warmup: int = 150
    learning_rate: float = 0.018
    temperature: float = 0.60
    cue_accuracy: float = 0.52
    etm_decay: float = 0.990
    etm_prior: float = 1.0
    individual_reward: float = 0.35
    coordination_weight: float = 1.20
    skill_weight: float = 0.55
    permutation_samples: int = 20000


def role_targets() -> Mapping[str, np.ndarray]:
    """Each context assigns a different role permutation to A, B and C."""
    permutations = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    return {agent: permutations[:, index] for index, agent in enumerate(AGENTS)}


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = values / temperature
    probs = np.exp(logits - np.max(logits))
    return probs / probs.sum()


def categorical_from_uniform(probs: np.ndarray, value: float) -> int:
    return int(np.searchsorted(np.cumsum(probs), value, side="right").clip(0, len(probs) - 1))


def cue_likelihood(cue: int, accuracy: float) -> np.ndarray:
    likelihood = np.full(N_ROLES, (1.0 - accuracy) / (N_ROLES - 1))
    likelihood[cue] = accuracy
    return likelihood


def posterior(prior: np.ndarray, cue: int, accuracy: float) -> np.ndarray:
    result = prior * cue_likelihood(cue, accuracy)
    return result / result.sum()


class PairwiseETM:
    """Every observer has a separate context model for every teammate."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.counts = {
            (observer, teammate): np.full(
                (N_CONTEXTS, N_ROLES), config.etm_prior, dtype=np.float64
            )
            for observer in AGENTS
            for teammate in AGENTS
            if observer != teammate
        }

    def distribution(self, observer: str, teammate: str, context: int) -> np.ndarray:
        row = self.counts[(observer, teammate)][context]
        return row / row.sum()

    def update(self, observer: str, teammate: str, context: int, intent: int) -> None:
        table = self.counts[(observer, teammate)]
        table *= self.config.etm_decay
        table += (1.0 - self.config.etm_decay) * self.config.etm_prior
        table[context, intent] += 1.0


def expected_coverage(role: int, teammate_beliefs: Sequence[np.ndarray]) -> float:
    value = 0.0
    for first in range(N_ROLES):
        for second in range(N_ROLES):
            if len({role, first, second}) == N_ROLES:
                value += teammate_beliefs[0][first] * teammate_beliefs[1][second]
    return value


def make_cue(intent: int, accuracy: float, correct_u: float, alt_u: float) -> int:
    if correct_u < accuracy:
        return intent
    alternatives = [role for role in range(N_ROLES) if role != intent]
    return alternatives[min(int(alt_u * len(alternatives)), len(alternatives) - 1)]


def beliefs_for_agent(
    condition: str,
    observer: str,
    context: int,
    intents: Mapping[str, int],
    cues: Mapping[str, int],
    model: PairwiseETM,
    config: Config,
) -> Tuple[List[np.ndarray], int]:
    beliefs = []
    correct = 0
    for teammate in AGENTS:
        if teammate == observer:
            continue
        if condition == "oracle":
            belief = np.zeros(N_ROLES)
            belief[intents[teammate]] = 1.0
        elif condition == "no_model":
            belief = cue_likelihood(cues[teammate], config.cue_accuracy)
        else:
            belief = posterior(
                model.distribution(observer, teammate, context),
                cues[teammate],
                config.cue_accuracy,
            )
        beliefs.append(belief)
        correct += int(np.argmax(belief) == intents[teammate])
    return beliefs, correct


def run_condition(
    seed: int,
    condition: str,
    config: Config,
    exogenous: Mapping[str, np.ndarray],
) -> List[dict]:
    targets = role_targets()
    q_values = {agent: np.zeros((N_CONTEXTS, N_ROLES)) for agent in AGENTS}
    model = PairwiseETM(config)
    rows: List[dict] = []

    for round_index in range(config.rounds):
        context = int(exogenous["contexts"][round_index])
        intents = {}
        cues = {}
        for agent_index, agent in enumerate(AGENTS):
            proposal_probs = softmax(q_values[agent][context], config.temperature)
            intents[agent] = categorical_from_uniform(
                proposal_probs, float(exogenous["proposal_u"][round_index, agent_index])
            )
            cues[agent] = make_cue(
                intents[agent],
                config.cue_accuracy,
                float(exogenous["cue_correct_u"][round_index, agent_index]),
                float(exogenous["cue_alt_u"][round_index, agent_index]),
            )

        actions = {}
        total_correct = 0
        for agent_index, agent in enumerate(AGENTS):
            teammate_beliefs, correct = beliefs_for_agent(
                condition, agent, context, intents, cues, model, config
            )
            total_correct += correct
            skill_probs = softmax(q_values[agent][context], config.temperature)
            utilities = np.array(
                [
                    config.skill_weight * skill_probs[role]
                    + config.coordination_weight
                    * expected_coverage(role, teammate_beliefs)
                    for role in range(N_ROLES)
                ]
            )
            best = np.flatnonzero(np.isclose(utilities, utilities.max()))
            tie_u = float(exogenous["tie_u"][round_index, agent_index])
            actions[agent] = int(best[min(int(tie_u * len(best)), len(best) - 1)])

        team_success = int(len(set(actions.values())) == N_ROLES)
        target_accuracy = float(
            np.mean(
                [
                    actions[agent] == int(targets[agent][context])
                    for agent in AGENTS
                ]
            )
        )
        rows.append(
            {
                "seed": seed,
                "round": round_index,
                "condition": condition,
                "target_accuracy": target_accuracy,
                "intent_accuracy": total_correct / (len(AGENTS) * (len(AGENTS) - 1)),
                "team_success": team_success,
            }
        )

        # Shared team outcome couples all three learning processes.
        for agent in AGENTS:
            own_correct = actions[agent] == int(targets[agent][context])
            reward = team_success + config.individual_reward * float(own_correct)
            role = actions[agent]
            old = q_values[agent][context, role]
            q_values[agent][context, role] = old + config.learning_rate * (reward - old)

        # The role proposal is treated as the high-level intent revealed by the trajectory.
        if condition in ("dynamic_etm", "frozen_etm"):
            should_update = condition == "dynamic_etm" or round_index < config.warmup
            if should_update:
                for observer in AGENTS:
                    for teammate in AGENTS:
                        if observer != teammate:
                            model.update(observer, teammate, context, intents[teammate])

    return rows


def exogenous_randomness(seed: int, config: Config) -> Mapping[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "contexts": rng.integers(N_CONTEXTS, size=config.rounds),
        "proposal_u": rng.random((config.rounds, len(AGENTS))),
        "cue_correct_u": rng.random((config.rounds, len(AGENTS))),
        "cue_alt_u": rng.random((config.rounds, len(AGENTS))),
        "tie_u": rng.random((config.rounds, len(AGENTS))),
    }


def run_seed(seed: int, config: Config) -> List[dict]:
    exogenous = exogenous_randomness(seed, config)
    rows = []
    for condition in CONDITIONS:
        rows.extend(run_condition(seed, condition, config, exogenous))
    return rows


def phase_bounds(config: Config) -> Mapping[str, Tuple[int, int]]:
    third = config.rounds // 3
    return {
        "early": (0, third),
        "middle": (third, 2 * third),
        "late": (2 * third, config.rounds),
    }


def aggregate_windows(rows: Sequence[dict], config: Config) -> List[dict]:
    buckets: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        start = (int(row["round"]) // config.window) * config.window
        buckets.setdefault((start, str(row["condition"])), []).append(row)
    output = []
    for (start, condition), values in sorted(buckets.items()):
        output.append(
            {
                "window_start": start,
                "window_end": min(start + config.window, config.rounds) - 1,
                "condition": condition,
                "target_accuracy": float(np.mean([v["target_accuracy"] for v in values])),
                "intent_accuracy": float(np.mean([v["intent_accuracy"] for v in values])),
                "team_success": float(np.mean([v["team_success"] for v in values])),
            }
        )
    return output


def phase_summary(rows: Sequence[dict], config: Config) -> dict:
    output: Dict[str, dict] = {}
    for phase, (start, end) in phase_bounds(config).items():
        output[phase] = {}
        for condition in CONDITIONS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition and start <= row["round"] < end
            ]
            output[phase][condition] = {
                "target_accuracy": float(np.mean([r["target_accuracy"] for r in selected])),
                "intent_accuracy": float(np.mean([r["intent_accuracy"] for r in selected])),
                "team_success": float(np.mean([r["team_success"] for r in selected])),
                "rounds": len(selected),
            }
    return output


def late_seed_values(
    rows: Sequence[dict], config: Config, condition: str, metric: str
) -> np.ndarray:
    start = phase_bounds(config)["late"][0]
    by_seed: Dict[int, List[float]] = {}
    for row in rows:
        if row["condition"] == condition and row["round"] >= start:
            by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return np.array([np.mean(by_seed[seed]) for seed in sorted(by_seed)])


def sign_flip_p(differences: np.ndarray, samples: int) -> float:
    observed = abs(float(differences.mean()))
    if np.allclose(differences, 0.0):
        return 1.0
    rng = np.random.default_rng(20260922)
    exceed = 0
    completed = 0
    while completed < samples:
        size = min(1000, samples - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        exceed += int(np.count_nonzero(np.abs((signs * differences).mean(axis=1)) >= observed))
        completed += size
    return (exceed + 1.0) / (samples + 1.0)


def comparison(rows: Sequence[dict], config: Config, metric: str) -> dict:
    dynamic = late_seed_values(rows, config, "dynamic_etm", metric)
    frozen = late_seed_values(rows, config, "frozen_etm", metric)
    diff = dynamic - frozen
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / math.sqrt(len(diff)))
    return {
        "metric": metric,
        "comparison": "dynamic_etm - frozen_etm",
        "phase": "late",
        "paired_seeds": len(diff),
        "mean_difference": mean,
        "normal_approx_95ci": [mean - 1.96 * se, mean + 1.96 * se],
        "two_sided_sign_flip_p": sign_flip_p(diff, config.permutation_samples),
    }


def run_experiment(config: Config) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    for seed in range(config.seeds):
        rows.extend(run_seed(seed, config))
    windows = aggregate_windows(rows, config)
    summary = {
        "status": "development_pilot",
        "claim_boundary": "Closed-loop toy mechanism check; not a benchmark result.",
        "config": asdict(config),
        "condition_rounds": len(rows),
        "phase_summary": phase_summary(rows, config),
        "paired_late_comparisons": [
            comparison(rows, config, "target_accuracy"),
            comparison(rows, config, "intent_accuracy"),
            comparison(rows, config, "team_success"),
        ],
    }
    return windows, summary


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0].keys()))
        writer.writeheader()
        writer.writerows(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=Config.seeds)
    parser.add_argument("--rounds", type=int, default=Config.rounds)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(seeds=args.seeds, rounds=args.rounds)
    output_dir = args.output_dir or Path(__file__).resolve().parent / "results"
    windows, summary = run_experiment(config)
    write_csv(output_dir / "learning_curve.csv", windows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
