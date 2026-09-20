"""M16 gate 2a: behavior-clone the conservative Alice on the new layout."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ocres.generic_agents import LayoutAwareTrainableAgent, ScriptedMacroAgent
from ocres.grid import World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.runner import run_episode
from ocres.trainable import (
    INTENT_ID,
    INTENTS,
    MacroIntentPolicy,
    ObservationSpec,
    encode_log_observation,
    legal_intent_mask,
    save_policy_checkpoint,
    seed_everything,
    select_device,
)


OUTPUT = pathlib.Path("data/m16_v2_alice_pre_results.json")
CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")


def randomized_world(seed, horizon):
    world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    rng = random.Random(5000 + int(seed))
    first, second = rng.sample(sorted(world.grid.passable), 2)
    world.env.state.players[0].update_pos_and_or(first, (1, 0))
    world.env.state.players[1].update_pos_and_or(second, (1, 0))
    world.env.state.timestep = 0
    return world


def collect(seed, horizon, spec):
    world = randomized_world(seed, horizon)
    alice = ScriptedMacroAgent(world.grid, 0, "cook", False, horizon)
    partner = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
    logs, metrics = run_episode(world, [alice, partner], horizon)
    samples = []
    deliveries = 0
    for row in logs:
        if row["info0"] and row["info0"].get("decision_id"):
            samples.append(
                (
                    encode_log_observation(row, deliveries, horizon, spec),
                    INTENT_ID[row["intent0"]],
                    legal_intent_mask(row),
                )
            )
        if row["r"] > 0:
            deliveries += 1
    return samples, metrics


def evaluate_offline(model, samples, device):
    x = torch.from_numpy(np.stack([row[0] for row in samples])).to(device)
    y = torch.tensor([row[1] for row in samples], dtype=torch.long, device=device)
    mask = torch.from_numpy(np.stack([row[2] for row in samples])).to(device)
    with torch.no_grad():
        prediction = model(x, mask).argmax(dim=1)
    return {"events": len(samples), "accuracy": float((prediction == y).float().mean().item())}


def train(model, samples, device, epochs, seed):
    x = np.stack([row[0] for row in samples])
    y = np.asarray([row[1] for row in samples], dtype=np.int64)
    mask = np.stack([row[2] for row in samples])
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask))
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    counts = np.bincount(y, minlength=len(INTENTS))
    weights = np.zeros(len(INTENTS), dtype=np.float32)
    present = counts > 0
    weights[present] = len(y) / (present.sum() * counts[present])
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    last_loss = None
    for _ in range(epochs):
        model.train()
        losses = []
        for batch_x, batch_y, batch_mask in loader:
            batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
            loss = loss_fn(model(batch_x, batch_mask), batch_y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        last_loss = float(np.mean(losses))
    return last_loss


def evaluate_closed_loop(model, spec, device, seeds, horizon):
    rows = []
    for seed in seeds:
        world = randomized_world(seed, horizon)
        alice = LayoutAwareTrainableAgent(
            world.grid, 0, model, spec, device=device, horizon=horizon, role_hint="cook"
        )
        partner = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
        _, metrics = run_episode(world, [alice, partner], horizon)
        rows.append({"seed": int(seed), "deliveries": metrics["deliveries"], "reward": metrics["reward"]})
    deliveries = [row["deliveries"] for row in rows]
    return {
        "mean_deliveries": float(np.mean(deliveries)),
        "min_deliveries": min(deliveries),
        "max_deliveries": max(deliveries),
        "per_seed": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=list(range(9101, 9125)))
    parser.add_argument("--test-seeds", type=int, nargs="+", default=list(range(9201, 9217)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=CHECKPOINT)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    spec = ObservationSpec(width=11, height=7, max_pots=3, version="m16-v2")
    train_samples, test_samples = [], []
    expert_train_scores, expert_test_scores = [], []
    for destination, scores, seeds in (
        (train_samples, expert_train_scores, args.train_seeds),
        (test_samples, expert_test_scores, args.test_seeds),
    ):
        for seed in seeds:
            samples, metrics = collect(seed, args.horizon, spec)
            destination.extend(samples)
            scores.append(metrics["deliveries"])
    model = MacroIntentPolicy(spec.dim, hidden_dim=args.hidden_dim).to(device)
    final_loss = train(model, train_samples, device, args.epochs, args.seed)
    result = {
        "milestone": "M16-V2 active-wait conservative Alice behavior cloning",
        "device": str(device),
        "seed": args.seed,
        "horizon": args.horizon,
        "split": {"train_seeds": args.train_seeds, "test_seeds": args.test_seeds},
        "observation": {"version": spec.version, "dim": spec.dim},
        "expert": {
            "train_mean_deliveries": float(np.mean(expert_train_scores)),
            "test_mean_deliveries": float(np.mean(expert_test_scores)),
        },
        "final_loss": final_loss,
        "offline_train": evaluate_offline(model, train_samples, device),
        "offline_test": evaluate_offline(model, test_samples, device),
        "closed_loop_test": evaluate_closed_loop(model, spec, device, args.test_seeds, args.horizon),
    }
    save_policy_checkpoint(args.checkpoint, model, spec, extra=result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("checkpoint:", args.checkpoint)


if __name__ == "__main__":
    main()
