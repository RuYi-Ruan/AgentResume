"""Collect counterfactual transitions and test a teammate-conditioned world model."""

from __future__ import annotations

import argparse
import csv
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
from MainSearch.explore.gate3_etm_decision_control.run_experiment import reset_to_seed
from MainSearch.explore.overcooked_three_agent_score_gap.run_experiment import (
    RoleAgent,
    make_env,
)
from MainSearch.explore.overcooked_three_mlp_learning_gate.run_experiment import ACTIONS


ROOT = Path(__file__).parent
SOURCE = ROOT.parent / "gate3_etm_decision_control" / "results_v2" / "evaluations.csv"
BELIEFS = ("frozen", "dynamic", "stale_budget")
CHECKPOINTS_TRAIN = (60, 120, 180, 240)
CHECKPOINT_VAL = 300
CHECKPOINT_TEST = 360
HELD = (None, "onion", "dish", "soup")
ORIENTATIONS = ((0, -1), (0, 1), (1, 0), (-1, 0))


@dataclass(frozen=True)
class Config:
    horizon: int = 180
    switch_step: int = 30
    prediction_steps: int = 30
    radius: int = 4
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.001
    intervention_margin: float = 0.5
    model_seed: int = 20260923


def one_hot(index, count):
    vector = [0.0] * count
    vector[index] = 1.0
    return vector


def held_name(player):
    return player.get_object().name if player.has_object() else None


def local_state(mdp, state, focal, radius):
    focal_player = state.players[focal]
    center = focal_player.position
    values = []
    for index, player in enumerate(state.players):
        dx = player.position[0] - center[0]
        dy = player.position[1] - center[1]
        visible = index == focal or abs(dx) + abs(dy) <= radius
        values.extend([
            float(visible),
            (dx / 12.0) if visible else 0.0,
            (dy / 6.0) if visible else 0.0,
        ])
        values.extend(one_hot(ORIENTATIONS.index(tuple(player.orientation)), 4) if visible else [0.0] * 4)
        values.extend(one_hot(HELD.index(held_name(player)), 4) if visible else [0.0] * 4)
    for pot in sorted(mdp.get_pot_locations()):
        visible = abs(pot[0] - center[0]) + abs(pot[1] - center[1]) <= radius
        if visible and state.has_object(pot):
            soup = state.get_object(pot)
            ingredients = len(soup.ingredients)
            cooking = float(soup.is_cooking)
            ready = float(soup.is_ready)
        else:
            ingredients, cooking, ready = 0, 0.0, 0.0
        values.extend([float(visible), ingredients / 3.0, cooking, ready])
    return np.asarray(values, dtype=np.float32)


def read_scenarios(path):
    scenarios = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(int(row[name]) for name in ("seed", "checkpoint", "eval_offset", "focal"))
            scenario = scenarios.setdefault(key, {"focal_role": int(row["base_role"]), "beliefs": {}})
            if row["condition"] == "base":
                scenario["partner_roles"] = tuple(int(x) for x in row["true_partner_roles"].split("-"))
            elif row["condition"] in BELIEFS:
                scenario["beliefs"][row["condition"]] = tuple(
                    int(x) for x in row["inferred_partner_roles"].split("-")
                )
    for key, scenario in scenarios.items():
        if set(scenario["beliefs"]) != set(BELIEFS) or "partner_roles" not in scenario:
            raise ValueError(f"incomplete scenario {key}")
    return scenarios


def roles_for_scenario(focal, scenario):
    roles = [None] * 3
    roles[focal] = scenario["focal_role"]
    targets = [index for index in range(3) if index != focal]
    for target, role in zip(targets, scenario["partner_roles"]):
        roles[target] = role
    return roles


def collect_candidate(mdp, env, key, scenario, candidate, config):
    seed, checkpoint, offset, focal = key
    episode_seed = 83_000_000 + seed * 100_000 + checkpoint * 100 + offset
    reset_to_seed(mdp, env, episode_seed)
    roles = roles_for_scenario(focal, scenario)
    agents = [RoleAgent(mdp, index, *ACTIONS[role]) for index, role in enumerate(roles)]
    replacement = RoleAgent(mdp, focal, *ACTIONS[candidate])
    start_features = None
    next_features = None
    prefix_soups = 0.0
    short_soups = 0.0
    total_soups = 0.0
    for tick in range(config.horizon):
        if tick == config.switch_step:
            start_features = local_state(mdp, env.state, focal, config.radius)
            prefix_soups = total_soups
        if tick == config.switch_step + config.prediction_steps:
            next_features = local_state(mdp, env.state, focal, config.radius)
        actions = tuple(
            (replacement if index == focal and tick >= config.switch_step else agent).action(env.state)
            for index, agent in enumerate(agents)
        )
        _, reward, done, _ = env.step(actions, joint_agent_action_info=[{}, {}, {}])
        soups = reward / 20.0
        total_soups += soups
        if config.switch_step <= tick < config.switch_step + config.prediction_steps:
            short_soups += soups
        if done:
            break
    if start_features is None or next_features is None:
        raise ValueError(f"horizon too short for transition: {key}")
    return {
        "start": start_features,
        "next": next_features,
        "short_soups": short_soups,
        "future_soups": total_soups - prefix_soups,
        "total_soups": total_soups,
        "prefix_soups": prefix_soups,
    }


def collect_dataset(scenarios, config):
    mdp, env = make_env(config.horizon)
    examples = []
    for index, (key, scenario) in enumerate(sorted(scenarios.items())):
        candidates = [collect_candidate(mdp, env, key, scenario, candidate, config) for candidate in range(3)]
        reference = candidates[0]["start"]
        if any(not np.array_equal(reference, item["start"]) for item in candidates[1:]):
            raise AssertionError(f"counterfactual prefixes differ: {key}")
        for candidate, result in enumerate(candidates):
            examples.append({"key": key, "scenario": scenario, "candidate": candidate, **result})
        if (index + 1) % 90 == 0:
            print(f"collected {index + 1}/{len(scenarios)} scenarios", flush=True)
    return examples


def feature(example, belief, use_belief=True):
    seed, checkpoint, offset, focal = example["key"]
    scenario = example["scenario"]
    result = list(example["start"])
    result.extend(one_hot(focal, 3))
    result.extend(one_hot(scenario["focal_role"], 3))
    result.extend(one_hot(example["candidate"], 3))
    if use_belief:
        for estimated_role in scenario["beliefs"][belief]:
            result.extend(one_hot(estimated_role, 3))
    else:
        result.extend([0.0] * 6)
    return np.asarray(result, dtype=np.float32)


def targets(example):
    return np.concatenate((
        example["next"],
        np.asarray([example["short_soups"], example["future_soups"]], dtype=np.float32),
    ))


class WorldModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, inputs):
        return self.net(inputs)


def fit_model(train, validation, config, use_belief=True):
    torch.manual_seed(config.model_seed)
    rng = np.random.default_rng(config.model_seed)
    train_x = np.stack([feature(example, belief, use_belief) for example in train for belief in BELIEFS])
    train_y = np.stack([targets(example) for example in train for _ in BELIEFS])
    val_x = torch.from_numpy(np.stack([feature(example, belief, use_belief) for example in validation for belief in BELIEFS]))
    val_y = torch.from_numpy(np.stack([targets(example) for example in validation for _ in BELIEFS]))
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)
    model = WorldModel(train_x.shape[1], train_y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = float("inf")
    best_weights = None
    best_epoch = None
    state_dim = train_y.shape[1] - 2
    for epoch in range(config.epochs):
        for indices in np.array_split(rng.permutation(len(train_x)), math.ceil(len(train_x) / config.batch_size)):
            predicted = model(train_x[indices])
            observed = train_y[indices]
            state_loss = nn.functional.mse_loss(predicted[:, :state_dim], observed[:, :state_dim])
            reward_loss = nn.functional.mse_loss(predicted[:, state_dim:], observed[:, state_dim:])
            loss = state_loss + 0.5 * reward_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predicted = model(val_x)
            observed = val_y
            validation_loss = float(
                nn.functional.mse_loss(predicted[:, :state_dim], observed[:, :state_dim])
                + 0.5 * nn.functional.mse_loss(predicted[:, state_dim:], observed[:, state_dim:])
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch + 1
            best_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    model.load_state_dict(best_weights)
    return model, {"best_epoch": best_epoch, "validation_loss": best_loss, "training_rows": len(train_x)}


def predict(model, example, belief, use_belief=True):
    with torch.no_grad():
        return model(torch.from_numpy(feature(example, belief, use_belief)).unsqueeze(0)).squeeze(0).numpy()


def evaluate(examples, model, config, use_belief=True):
    groups = {}
    state_errors = []
    short_errors = []
    future_errors = []
    for example in examples:
        scenario_key = example["key"]
        group = groups.setdefault(scenario_key, {})
        group[example["candidate"]] = example
        for belief in BELIEFS:
            prediction = predict(model, example, belief, use_belief)
            example.setdefault("predicted", {})[belief] = prediction
            state_dim = len(example["next"])
            state_errors.append(float(np.abs(prediction[:state_dim] - example["next"]).mean()))
            short_errors.append(abs(float(prediction[state_dim]) - example["short_soups"]))
            future_errors.append(abs(float(prediction[state_dim + 1]) - example["future_soups"]))
    rows = []
    for key, candidates in sorted(groups.items()):
        if set(candidates) != {0, 1, 2}:
            raise ValueError(f"missing candidate: {key}")
        scenario = candidates[0]["scenario"]
        base_role = scenario["focal_role"]
        baseline = candidates[base_role]["total_soups"]
        actual = np.asarray([candidates[c]["total_soups"] for c in range(3)])
        for belief in BELIEFS:
            predicted = np.asarray([
                candidates[c]["predicted"][belief][-1] for c in range(3)
            ])
            selected = base_role
            gain = predicted - predicted[base_role]
            best = int(np.argmax(gain))
            if gain[best] > config.intervention_margin:
                selected = best
            pair_correct = []
            for left in range(3):
                for right in range(left + 1, 3):
                    if actual[left] != actual[right]:
                        pair_correct.append(int((actual[left] > actual[right]) == (predicted[left] > predicted[right])))
            rows.append({
                "seed": key[0], "checkpoint": key[1], "eval_offset": key[2], "focal": key[3],
                "belief": belief, "base_role": base_role, "selected_role": selected,
                "baseline_soups": baseline, "selected_soups": actual[selected],
                "oracle_soups": float(actual.max()),
                "pairwise_correct": sum(pair_correct), "pairwise_count": len(pair_correct),
            })
    metrics = {}
    for belief in BELIEFS:
        part = [row for row in rows if row["belief"] == belief]
        metrics[belief] = {
            "scenarios": len(part),
            "mean_soups": float(np.mean([row["selected_soups"] for row in part])),
            "mean_gain_vs_keep": float(np.mean([row["selected_soups"] - row["baseline_soups"] for row in part])),
            "interventions": sum(row["selected_role"] != row["base_role"] for row in part),
            "pairwise_ranking_accuracy": sum(row["pairwise_correct"] for row in part) / max(sum(row["pairwise_count"] for row in part), 1),
            "per_seed_gain": {
                str(seed): float(np.mean([
                    row["selected_soups"] - row["baseline_soups"]
                    for row in part if row["seed"] == seed
                ]))
                for seed in sorted({row["seed"] for row in part})
            },
        }
    metrics["keep"] = {
        "scenarios": len(groups),
        "mean_soups": float(np.mean([candidates[candidates[0]["scenario"]["focal_role"]]["total_soups"] for candidates in groups.values()])),
    }
    metrics["oracle"] = {
        "scenarios": len(groups),
        "mean_soups": float(np.mean([max(candidate["total_soups"] for candidate in candidates.values()) for candidates in groups.values()])),
    }
    return rows, {
        "state_mae": float(np.mean(state_errors)),
        "short_reward_mae_soups": float(np.mean(short_errors)),
        "future_reward_mae_soups": float(np.mean(future_errors)),
        "conditions": metrics,
    }


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    config = Config(epochs=args.epochs)
    scenarios = read_scenarios(args.source)
    if set(key[1] for key in scenarios) != set((*CHECKPOINTS_TRAIN, CHECKPOINT_VAL, CHECKPOINT_TEST)):
        raise ValueError("unexpected checkpoint split")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "counterfactual_data.pt"
    if args.reuse_data:
        examples = torch.load(dataset_path, weights_only=False)
    else:
        examples = collect_dataset(scenarios, config)
        torch.save(examples, dataset_path)
    train = [example for example in examples if example["key"][1] in CHECKPOINTS_TRAIN]
    validation = [example for example in examples if example["key"][1] == CHECKPOINT_VAL]
    test = [example for example in examples if example["key"][1] == CHECKPOINT_TEST]
    model, training = fit_model(train, validation, config)
    validation_rows, validation_metrics = evaluate(validation, model, config)
    test_rows, test_metrics = evaluate(test, model, config)
    no_belief_model, no_belief_training = fit_model(
        train, validation, config, use_belief=False
    )
    _, no_belief_validation = evaluate(
        validation, no_belief_model, config, use_belief=False
    )
    _, no_belief_test = evaluate(test, no_belief_model, config, use_belief=False)
    summary = {
        "status": "development_gate3b_world_model",
        "claim_boundary": "Short-step local transition and soup prediction with scripted low-level execution; not a formal main result.",
        "config": asdict(config),
        "source_scenarios": len(scenarios),
        "candidate_rollouts": len(examples),
        "train_candidate_rows": len(train),
        "validation_candidate_rows": len(validation),
        "test_candidate_rows": len(test),
        "training": training,
        "validation": validation_metrics,
        "test": test_metrics,
        "no_belief_ablation": {
            "training": no_belief_training,
            "validation": no_belief_validation,
            "test": no_belief_test,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "validation_decisions.csv", validation_rows)
    write_rows(args.output_dir / "test_decisions.csv", test_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    torch.save(model.state_dict(), args.output_dir / "world_model.pt")
    torch.save(no_belief_model.state_dict(), args.output_dir / "world_model_no_belief.pt")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
