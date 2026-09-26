"""Toy pilot for gradually evolving teammates and a dynamic teammate model.

The experiment is deliberately small and dependency-light. It tests a mechanism,
not a publishable benchmark claim.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


N_ROLES = 3
N_CONTEXTS = 6
PARTNERS = ("B", "C")
CONDITIONS = ("no_model", "frozen_etm", "dynamic_etm", "oracle")


@dataclass(frozen=True)
class Config:
    seeds: int = 100
    rounds: int = 1200
    window: int = 100
    warmup: int = 120
    learning_rate: float = 0.025
    temperature: float = 0.55
    cue_accuracy: float = 0.55
    etm_decay: float = 0.985
    etm_prior: float = 1.0
    permutation_samples: int = 20000


class OnlinePartner:
    """A fixed-identity contextual bandit that learns one small step per round."""

    def __init__(self, targets: np.ndarray, config: Config) -> None:
        self.targets = targets
        self.config = config
        self.q_values = np.zeros((N_CONTEXTS, N_ROLES), dtype=np.float64)

    def act(self, context: int, rng: np.random.Generator) -> int:
        logits = self.q_values[context] / self.config.temperature
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return int(rng.choice(N_ROLES, p=probs))

    def update(self, context: int, role: int) -> bool:
        correct = role == int(self.targets[context])
        reward = float(correct)
        old = self.q_values[context, role]
        self.q_values[context, role] = old + self.config.learning_rate * (reward - old)
        return correct


class TeammateModel:
    """Identity- and context-conditioned role beliefs with optional forgetting."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.counts = {
            partner: np.full(
                (N_CONTEXTS, N_ROLES), config.etm_prior, dtype=np.float64
            )
            for partner in PARTNERS
        }

    def distribution(self, partner: str, context: int) -> np.ndarray:
        row = self.counts[partner][context]
        return row / row.sum()

    def update(self, partner: str, context: int, role: int) -> None:
        table = self.counts[partner]
        table *= self.config.etm_decay
        table += (1.0 - self.config.etm_decay) * self.config.etm_prior
        table[context, role] += 1.0


def target_maps() -> Mapping[str, np.ndarray]:
    """B and C always have distinct context-dependent target roles."""
    b = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    c = np.array([1, 2, 0, 2, 0, 1], dtype=np.int64)
    return {"B": b, "C": c}


def noisy_cue(intent: int, accuracy: float, rng: np.random.Generator) -> int:
    if rng.random() < accuracy:
        return intent
    alternatives = [role for role in range(N_ROLES) if role != intent]
    return int(rng.choice(alternatives))


def cue_likelihood(cue: int, accuracy: float) -> np.ndarray:
    other = (1.0 - accuracy) / (N_ROLES - 1)
    likelihood = np.full(N_ROLES, other, dtype=np.float64)
    likelihood[cue] = accuracy
    return likelihood


def posterior_intent(
    prior: np.ndarray, cue: int, cue_accuracy: float
) -> np.ndarray:
    posterior = prior * cue_likelihood(cue, cue_accuracy)
    total = posterior.sum()
    if total <= 0:
        return np.full(N_ROLES, 1.0 / N_ROLES)
    return posterior / total


def select_complementary_role(
    beliefs: Sequence[np.ndarray], rng: np.random.Generator
) -> int:
    """Select A's role to maximize expected probability of covering all roles."""
    values = np.zeros(N_ROLES, dtype=np.float64)
    for a_role in range(N_ROLES):
        for b_role in range(N_ROLES):
            for c_role in range(N_ROLES):
                if len({a_role, b_role, c_role}) == N_ROLES:
                    values[a_role] += beliefs[0][b_role] * beliefs[1][c_role]
    best = np.flatnonzero(np.isclose(values, values.max()))
    return int(rng.choice(best))


def infer_for_condition(
    condition: str,
    context: int,
    cues: Mapping[str, int],
    intents: Mapping[str, int],
    model: TeammateModel,
    config: Config,
    rng: np.random.Generator,
) -> Tuple[Dict[str, int], int]:
    beliefs: List[np.ndarray] = []
    predictions: Dict[str, int] = {}

    for partner in PARTNERS:
        if condition == "oracle":
            belief = np.zeros(N_ROLES, dtype=np.float64)
            belief[intents[partner]] = 1.0
        elif condition == "no_model":
            belief = cue_likelihood(cues[partner], config.cue_accuracy)
        else:
            prior = model.distribution(partner, context)
            belief = posterior_intent(prior, cues[partner], config.cue_accuracy)
        beliefs.append(belief)
        predictions[partner] = int(np.argmax(belief))

    role = select_complementary_role(beliefs, rng)
    return predictions, role


def run_seed(seed: int, config: Config) -> List[dict]:
    rng = np.random.default_rng(seed)
    maps = target_maps()
    learners = {
        partner: OnlinePartner(maps[partner], config) for partner in PARTNERS
    }
    models = {
        "frozen_etm": TeammateModel(config),
        "dynamic_etm": TeammateModel(config),
    }
    rows: List[dict] = []

    for round_index in range(config.rounds):
        context = int(rng.integers(N_CONTEXTS))
        intents = {
            partner: learners[partner].act(context, rng) for partner in PARTNERS
        }
        cues = {
            partner: noisy_cue(intents[partner], config.cue_accuracy, rng)
            for partner in PARTNERS
        }
        competence = {
            partner: int(intents[partner] == int(maps[partner][context]))
            for partner in PARTNERS
        }

        # Separate RNG streams keep tie-breaking reproducible per condition.
        for condition_index, condition in enumerate(CONDITIONS):
            condition_rng = np.random.default_rng(
                seed * 10_000_000 + round_index * 10 + condition_index
            )
            model = models.get(condition, models["dynamic_etm"])
            predictions, a_role = infer_for_condition(
                condition,
                context,
                cues,
                intents,
                model,
                config,
                condition_rng,
            )
            intent_correct = sum(
                predictions[partner] == intents[partner] for partner in PARTNERS
            )
            team_success = int(
                len({a_role, intents["B"], intents["C"]}) == N_ROLES
            )
            rows.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "condition": condition,
                    "partner_capability": np.mean(list(competence.values())),
                    "intent_accuracy": intent_correct / len(PARTNERS),
                    "team_success": team_success,
                }
            )

        # Evidence is revealed after the round. Frozen ETM stops learning after warmup.
        for partner in PARTNERS:
            models["dynamic_etm"].update(partner, context, intents[partner])
            if round_index < config.warmup:
                models["frozen_etm"].update(partner, context, intents[partner])
            learners[partner].update(context, intents[partner])

    return rows


def aggregate_windows(rows: Sequence[dict], config: Config) -> List[dict]:
    buckets: Dict[Tuple[int, str], List[dict]] = {}
    for row in rows:
        window_start = (int(row["round"]) // config.window) * config.window
        buckets.setdefault((window_start, str(row["condition"])), []).append(row)

    output = []
    for (window_start, condition), values in sorted(buckets.items()):
        output.append(
            {
                "window_start": window_start,
                "window_end": window_start + config.window - 1,
                "condition": condition,
                "partner_capability": float(
                    np.mean([v["partner_capability"] for v in values])
                ),
                "intent_accuracy": float(
                    np.mean([v["intent_accuracy"] for v in values])
                ),
                "team_success": float(np.mean([v["team_success"] for v in values])),
            }
        )
    return output


def phase_bounds(config: Config) -> Mapping[str, Tuple[int, int]]:
    third = config.rounds // 3
    return {
        "early": (0, third),
        "middle": (third, 2 * third),
        "late": (2 * third, config.rounds),
    }


def condition_phase_summary(rows: Sequence[dict], config: Config) -> dict:
    summary: Dict[str, dict] = {}
    for phase, (start, end) in phase_bounds(config).items():
        summary[phase] = {}
        for condition in CONDITIONS:
            selected = [
                row
                for row in rows
                if row["condition"] == condition and start <= row["round"] < end
            ]
            summary[phase][condition] = {
                "partner_capability": float(
                    np.mean([row["partner_capability"] for row in selected])
                ),
                "intent_accuracy": float(
                    np.mean([row["intent_accuracy"] for row in selected])
                ),
                "team_success": float(
                    np.mean([row["team_success"] for row in selected])
                ),
                "events": len(selected),
            }
    return summary


def paired_seed_values(
    rows: Sequence[dict], config: Config, condition: str, metric: str
) -> np.ndarray:
    late_start = phase_bounds(config)["late"][0]
    by_seed: Dict[int, List[float]] = {}
    for row in rows:
        if row["condition"] == condition and row["round"] >= late_start:
            by_seed.setdefault(int(row["seed"]), []).append(float(row[metric]))
    return np.array([np.mean(by_seed[seed]) for seed in sorted(by_seed)])


def paired_sign_flip_test(
    differences: np.ndarray, samples: int, seed: int = 20260922
) -> float:
    observed = abs(float(differences.mean()))
    if np.allclose(differences, 0.0):
        return 1.0
    rng = np.random.default_rng(seed)
    exceed = 0
    processed = 0
    chunk_size = 1000
    while processed < samples:
        size = min(chunk_size, samples - processed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(differences)))
        permuted = np.abs((signs * differences).mean(axis=1))
        exceed += int(np.count_nonzero(permuted >= observed))
        processed += size
    return (exceed + 1.0) / (samples + 1.0)


def paired_comparison(rows: Sequence[dict], config: Config, metric: str) -> dict:
    dynamic = paired_seed_values(rows, config, "dynamic_etm", metric)
    frozen = paired_seed_values(rows, config, "frozen_etm", metric)
    differences = dynamic - frozen
    mean = float(differences.mean())
    se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    return {
        "metric": metric,
        "comparison": "dynamic_etm - frozen_etm",
        "phase": "late",
        "paired_seeds": len(differences),
        "mean_difference": mean,
        "normal_approx_95ci": [mean - 1.96 * se, mean + 1.96 * se],
        "two_sided_sign_flip_p": paired_sign_flip_test(
            differences, config.permutation_samples
        ),
    }


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: Config) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    for seed in range(config.seeds):
        rows.extend(run_seed(seed, config))

    windows = aggregate_windows(rows, config)
    phases = condition_phase_summary(rows, config)
    summary = {
        "status": "development_pilot",
        "claim_boundary": (
            "Toy mechanism check only; not evidence that ETM works in a standard "
            "multi-agent benchmark."
        ),
        "config": asdict(config),
        "events": len(rows),
        "phase_summary": phases,
        "paired_late_comparisons": [
            paired_comparison(rows, config, "intent_accuracy"),
            paired_comparison(rows, config, "team_success"),
        ],
    }
    return windows, summary


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
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
