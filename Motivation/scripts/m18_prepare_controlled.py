"""Freeze approved M18 A mechanism samples for one passed Alice group."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")
import torch

from ocres.m18_controlled import enumerate_snapshots, freeze_splits, public_rows, split_hash
from ocres.m18_training import config_hash, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save


def save_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"controlled A file differs on resume: {path}")
    else:
        save(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run id")
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id / f"group_{args.group}"
    artifacts = Path("artifacts/m18") / args.run_id
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    if not result["passed"] or result["config_hash"] != config_hash(config):
        raise RuntimeError("Alice capability gate did not pass")
    training = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in training["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("training source changed")
    history = json.loads((root / "history" / "impressions.json").read_text(encoding="utf-8"))
    if not history["passed"]:
        raise RuntimeError("visible history gate did not pass")
    destination = root / "controlled_A"
    if (destination / "manifest.json").exists():
        raise RuntimeError("controlled samples already frozen; inspect instead of replacing")
    destination.mkdir(parents=True, exist_ok=True)
    frozen_start = {"group": args.group, "config_hash": config_hash(config),
                    "preparer_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "training_manifest_hash": hashlib.sha256((artifacts / "manifest.json").read_bytes()).hexdigest(),
                    "history_hash": hashlib.sha256((root / "history/impressions.json").read_bytes()).hexdigest(),
                    "enumerator_hash": hashlib.sha256(Path("src/ocres/m18_controlled.py").read_bytes()).hexdigest(),
                    "bob_protocol_hash": hashlib.sha256(Path("src/ocres/m18_bob.py").read_bytes()).hexdigest()}
    save_once(destination / "sampling_started.json", frozen_start)
    models = []
    for stage in ("pre", "post"):
        model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / f"{stage}.pt")
        model.eval()
        models.append(model)
    candidates, funnel = enumerate_snapshots(group["rows"], models, spec, group["map"], config["horizon"])
    split = freeze_splits(candidates, history, args.group,
                          config["evaluation"]["A_events_per_group"], 10)
    manifest = {"config_hash": config_hash(config), "training_manifest_hash":
                hashlib.sha256((artifacts / "manifest.json").read_bytes()).hexdigest(),
                "history_hash": hashlib.sha256((root / "history" / "impressions.json").read_bytes()).hexdigest(),
                "enumerator_hash": hashlib.sha256(Path("src/ocres/m18_controlled.py").read_bytes()).hexdigest(),
                "bob_protocol_hash": hashlib.sha256(Path("src/ocres/m18_bob.py").read_bytes()).hexdigest(),
                "source_kind": "controlled; no natural episode id", "group": args.group,
                "funnel": {**funnel, **dict(split["funnel"])},
                "development_hash": split_hash(split["development"]),
                "formal_hash": split_hash(split["formal"]),
                "passed": split["passed"], "post_intent_counts": split["post_intent_counts"]}
    save_once(destination / "candidates.json", candidates)
    save_once(destination / "development_scoring.json", split["development"])
    save_once(destination / "formal_scoring.json", split["formal"])
    save_once(destination / "development_public.json", public_rows(split["development"]))
    save_once(destination / "formal_public.json", public_rows(split["formal"]))
    save_once(destination / "manifest.json", manifest)
    print(json.dumps({"group": args.group, "passed": split["passed"], "funnel": manifest["funnel"],
                      "post_intent_counts": manifest["post_intent_counts"]}), flush=True)


if __name__ == "__main__":
    main()
