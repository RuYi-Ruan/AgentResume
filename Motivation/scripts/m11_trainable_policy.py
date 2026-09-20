"""M11: train and validate the first PyTorch macro-intent policy.

This is an interface/behavior-cloning milestone, not the final capability
improvement experiment.  By default it trains only on scripted L0 episodes;
PPO checkpoints will replace these demonstrations in M13.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

sys.path.insert(0, "src")
from ocres.trainable import (
    INTENTS,
    HistoricalIntentDataset,
    MacroIntentPolicy,
    ObservationSpec,
    save_policy_checkpoint,
    seed_everything,
    select_device,
)
import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = pathlib.Path("data/twopot_v1")
RESULT_PATH = pathlib.Path("data/m11_bc_results.json")
CHECKPOINT_PATH = pathlib.Path("artifacts/checkpoints/m11_bc_l0.pt")


def seed_of(path):
    match = re.search(r"seed(\d+)", path.stem)
    if not match:
        raise ValueError(f"cannot infer seed from {path}")
    return int(match.group(1))


def evaluate(model, dataset, device):
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    truth, prediction = [], []
    model.eval()
    with torch.no_grad():
        for x, y, mask in loader:
            pred = model(x.to(device), mask.to(device)).argmax(dim=1).cpu()
            truth.extend(y.tolist())
            prediction.extend(pred.tolist())
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    per_intent = {}
    for index, name in enumerate(INTENTS):
        selected = truth == index
        per_intent[name] = {
            "support": int(selected.sum()),
            "accuracy": float(np.mean(prediction[selected] == truth[selected])) if selected.any() else None,
        }
    return {
        "n": len(truth),
        "accuracy": float(np.mean(prediction == truth)),
        "per_intent": per_intent,
    }


def train(model, dataset, device, epochs, batch_size, learning_rate):
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    counts = np.bincount(dataset.labels, minlength=len(INTENTS))
    weights = np.zeros(len(INTENTS), dtype=np.float32)
    present = counts > 0
    weights[present] = len(dataset) / (present.sum() * counts[present])
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    last_loss = None
    for _ in range(epochs):
        total_loss = 0.0
        total = 0
        for x, y, mask in loader:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, mask), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            total += len(y)
        last_loss = total_loss / max(total, 1)
    return last_loss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", default=["L0"])
    parser.add_argument("--test-seed", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    paths = []
    for level in args.levels:
        paths.extend(sorted((ROOT / f"level_{level}").glob("ep_seed*.npz"), key=seed_of))
    if not paths:
        raise SystemExit(f"no episodes found for levels {args.levels}")
    train_paths = [path for path in paths if seed_of(path) != args.test_seed]
    test_paths = [path for path in paths if seed_of(path) == args.test_seed]
    if not train_paths or not test_paths:
        raise SystemExit("episode-level split produced an empty train or test set")

    spec = ObservationSpec()
    train_data = HistoricalIntentDataset(train_paths, spec=spec, event_only=True)
    test_data = HistoricalIntentDataset(test_paths, spec=spec, event_only=True)
    model = MacroIntentPolicy(spec.dim, hidden_dim=args.hidden_dim).to(device)
    loss = train(model, train_data, device, args.epochs, args.batch_size, args.learning_rate)
    train_metrics = evaluate(model, train_data, device)
    test_metrics = evaluate(model, test_data, device)
    result = {
        "milestone": "M11 PyTorch macro-intent behavior-cloning baseline",
        "claim_scope": "interface and supervised-learning validation only; Alice is still scripted",
        "source_levels": args.levels,
        "split": {"unit": "episode/seed", "test_seed": args.test_seed},
        "observation": {"version": spec.version, "dim": spec.dim},
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "epochs": args.epochs,
        "final_loss": loss,
        "train": train_metrics,
        "test": test_metrics,
    }
    save_policy_checkpoint(CHECKPOINT_PATH, model, spec, extra=result)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"results: {RESULT_PATH}")


if __name__ == "__main__":
    main()
