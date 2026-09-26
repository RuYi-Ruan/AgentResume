"""Compare shared-private, independent, and ETM-free high-level world models."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import (
    play_episode,
)
from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import make_env


@dataclass(frozen=True)
class Config:
    scenarios: int = 180
    train_scenarios: int = 90
    validation_scenarios: int = 30
    horizon: int = 120
    model_seeds: int = 5
    epochs: int = 250
    learning_rate: float = 0.01


class RewardWorldModel(nn.Module):
    def __init__(self, input_dim=14):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, value):
        return self.net(value).squeeze(-1)


def directional_belief(actual_role, scenario, observer, target):
    rng = random.Random(7_000_000 + scenario * 100 + observer * 10 + target)
    confidence = rng.uniform(0.45, 0.85)
    remainder = 1.0 - confidence
    split = rng.uniform(0.2, 0.8)
    belief = [0.0, 0.0, 0.0]
    belief[actual_role] = confidence
    alternatives = [role for role in range(3) if role != actual_role]
    belief[alternatives[0]] = remainder * split
    belief[alternatives[1]] = remainder * (1.0 - split)
    return belief


def feature(row, use_etm=True):
    observer = row["observer"]
    values = [1.0 if observer == index else 0.0 for index in range(3)]
    values.extend([row["own_x"] / 12.0, row["own_y"] / 6.0])
    values.extend([1.0 if row["own_role"] == role else 0.0 for role in range(3)])
    for slot in (0, 1):
        if use_etm:
            values.extend(row[f"belief_{slot}"])
        else:
            values.extend([1.0 / 3.0] * 3)
    return values


def generate_dataset(config):
    mdp, env = make_env(config.horizon)
    rows = []
    scenario_meta = {}
    for scenario in range(config.scenarios):
        rng = random.Random(100_000 + scenario)
        responder = scenario % 3
        others = [index for index in range(3) if index != responder]
        fixed_roles = {target: rng.randrange(3) for target in others}
        scenario_meta[scenario] = {"responder": responder, "fixed_roles": fixed_roles}
        episode_seed = 3_000_000 + scenario
        for candidate in range(3):
            roles = [None, None, None]
            roles[responder] = candidate
            for target in others:
                roles[target] = fixed_roles[target]
            soups, starts = play_episode(mdp, env, tuple(roles), episode_seed, config.horizon)
            for observer in range(3):
                targets = [target for target in range(3) if target != observer]
                rows.append(
                    {
                        "scenario": scenario,
                        "candidate": candidate,
                        "responder": responder,
                        "observer": observer,
                        "own_role": roles[observer],
                        "own_x": starts[observer][0],
                        "own_y": starts[observer][1],
                        "belief_0": directional_belief(
                            roles[targets[0]], scenario, observer, targets[0]
                        ),
                        "belief_1": directional_belief(
                            roles[targets[1]], scenario, observer, targets[1]
                        ),
                        "soups": soups,
                    }
                )
    return rows, scenario_meta


def tensors(rows, use_etm=True):
    x = torch.tensor([feature(row, use_etm) for row in rows], dtype=torch.float32)
    y = torch.tensor([row["soups"] / 10.0 for row in rows], dtype=torch.float32)
    return x, y


def train_model(rows, seed, use_etm=True, config=None):
    torch.manual_seed(seed)
    model = RewardWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x, y = tensors(rows, use_etm)
    for _ in range(config.epochs):
        prediction = model(x)
        loss = nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def predict(model, row, use_etm=True):
    with torch.no_grad():
        value = torch.tensor([feature(row, use_etm)], dtype=torch.float32)
        return float(model(value)[0] * 10.0)


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def direct_rule_choice(rows):
    first = rows[0]
    predicted_others = [
        int(np.argmax(first["belief_0"])),
        int(np.argmax(first["belief_1"])),
    ]
    scores = [(len(set(predicted_others + [role])), -role, role) for role in range(3)]
    return max(scores)[-1]


def evaluate_once(train_rows, test_rows, seed, config):
    shared = train_model(train_rows, seed, True, config)
    no_etm = train_model(train_rows, seed + 1000, False, config)
    independent = {
        observer: train_model(
            [row for row in train_rows if row["observer"] == observer],
            seed + 2000 + observer,
            True,
            config,
        )
        for observer in range(3)
    }

    mae = {
        "shared_private": float(
            np.mean([abs(predict(shared, row) - row["soups"]) for row in test_rows])
        ),
        "independent": float(
            np.mean(
                [
                    abs(predict(independent[row["observer"]], row) - row["soups"])
                    for row in test_rows
                ]
            )
        ),
        "shared_no_etm": float(
            np.mean(
                [abs(predict(no_etm, row, False) - row["soups"]) for row in test_rows]
            )
        ),
    }

    scenario_ids = sorted({row["scenario"] for row in test_rows})
    outcomes = {name: [] for name in (*mae.keys(), "direct_rule", "oracle")}
    regrets = {name: [] for name in outcomes if name != "oracle"}
    for scenario in scenario_ids:
        scenario_rows = [row for row in test_rows if row["scenario"] == scenario]
        responder = scenario_rows[0]["responder"]
        candidates = {
            candidate: next(
                row
                for row in scenario_rows
                if row["candidate"] == candidate and row["observer"] == responder
            )
            for candidate in range(3)
        }
        choices = {
            "shared_private": max(candidates, key=lambda c: predict(shared, candidates[c])),
            "independent": max(
                candidates, key=lambda c: predict(independent[responder], candidates[c])
            ),
            "shared_no_etm": max(
                candidates, key=lambda c: predict(no_etm, candidates[c], False)
            ),
            "direct_rule": direct_rule_choice([candidates[c] for c in range(3)]),
            "oracle": max(candidates, key=lambda c: candidates[c]["soups"]),
        }
        oracle_value = candidates[choices["oracle"]]["soups"]
        for name, choice in choices.items():
            value = candidates[choice]["soups"]
            outcomes[name].append(value)
            if name != "oracle":
                regrets[name].append(oracle_value - value)
    return {
        "mae": mae,
        "mean_soups": {name: float(np.mean(values)) for name, values in outcomes.items()},
        "mean_regret": {name: float(np.mean(values)) for name, values in regrets.items()},
        "parameter_count": {
            "shared_private": parameter_count(shared),
            "independent_total": sum(parameter_count(model) for model in independent.values()),
            "shared_no_etm": parameter_count(no_etm),
        },
    }


def aggregate(runs, config):
    methods = ("shared_private", "independent", "shared_no_etm")
    decision_methods = (*methods, "direct_rule", "oracle")
    return {
        "status": "development_world_model_integration_gate",
        "claim_boundary": "One-step high-level reward world model on one three-agent Overcooked map; not a formal main-experiment result.",
        "config": asdict(config),
        "rollouts": config.scenarios * 3,
        "training_rows": config.train_scenarios * 3 * 3,
        "test_scenarios": config.scenarios
        - config.train_scenarios
        - config.validation_scenarios,
        "test_mae_soups": {
            method: float(np.mean([run["mae"][method] for run in runs]))
            for method in methods
        },
        "decision_mean_soups": {
            method: float(np.mean([run["mean_soups"][method] for run in runs]))
            for method in decision_methods
        },
        "decision_mean_regret": {
            method: float(np.mean([run["mean_regret"][method] for run in runs]))
            for method in decision_methods
            if method != "oracle"
        },
        "parameter_count": runs[0]["parameter_count"],
    }


def serializable_row(row):
    output = dict(row)
    output["belief_0"] = json.dumps(output["belief_0"])
    output["belief_1"] = json.dumps(output["belief_1"])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=int, default=Config.scenarios)
    parser.add_argument("--train-scenarios", type=int, default=Config.train_scenarios)
    parser.add_argument("--validation-scenarios", type=int, default=Config.validation_scenarios)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--model-seeds", type=int, default=Config.model_seeds)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()
    config = Config(
        scenarios=args.scenarios,
        train_scenarios=args.train_scenarios,
        validation_scenarios=args.validation_scenarios,
        horizon=args.horizon,
        model_seeds=args.model_seeds,
        epochs=args.epochs,
    )
    if config.train_scenarios + config.validation_scenarios >= config.scenarios:
        raise ValueError("train + validation scenarios must leave a non-empty test split")
    rows, _ = generate_dataset(config)
    train_rows = [row for row in rows if row["scenario"] < config.train_scenarios]
    test_start = config.train_scenarios + config.validation_scenarios
    test_rows = [row for row in rows if row["scenario"] >= test_start]
    runs = []
    for model_seed in range(config.model_seeds):
        runs.append(evaluate_once(train_rows, test_rows, 90_000 + model_seed, config))
        print(f"model seed {model_seed + 1}/{config.model_seeds} complete", flush=True)
    summary = aggregate(runs, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        values = [serializable_row(row) for row in rows]
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
