"""Approved B development gate: natural starts, public triggers only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import torch

from ocres.m18_data import collect_episode
from ocres.m18_training import episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save, emit


def save_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"B development gate file differs on resume: {path}")
    else:
        save(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id
    destination = root / f"group_{args.group}" / "B_development_gate"
    if (destination / "gate.json").exists():
        raise RuntimeError("B development gate already complete")
    growth = json.loads((destination.parent / "result.json").read_text(encoding="utf-8"))
    if not growth["passed"]:
        raise RuntimeError("Alice capability failed")
    artifacts = Path("artifacts/m18") / args.run_id
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in manifest["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("training source changed")
    model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / "post.pt")
    model.eval()
    save_once(destination / "sampling_started.json", {
        "group": args.group, "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_hash": hashlib.sha256((artifacts / "checkpoints" / f"group_{args.group}" / "post.pt").read_bytes()).hexdigest(),
        "bob_hash": hashlib.sha256(Path("src/ocres/m18_bob.py").read_bytes()).hexdigest()})
    episodes = []
    for seed in episode_seeds(config, args.group, "development"):
        target = destination / "episodes" / f"{seed}.json"
        if target.exists():
            episode = json.loads(target.read_text(encoding="utf-8"))
            if episode["seed"] != seed or len(episode["replay"]) != config["horizon"]:
                raise RuntimeError(f"saved B development episode incomplete: {target}")
        else:
            episode = collect_episode(group["rows"], seed, model, spec, map_name=group["map"])
            save(target, episode)
            emit(root / "progress.jsonl", {"stage": "B_development_gate", "group": args.group,
                                           "seed": seed, "trigger_count": len(episode["public_triggers"])})
        episodes.append({"seed": seed, "public_triggers": episode["public_triggers"],
                         "soups": episode["deliveries"],
                         "visible_decisions": episode["visible_task_starts"]})
    triggering = sum(bool(r["public_triggers"]) for r in episodes)
    mean = sum(len(r["public_triggers"]) for r in episodes)/len(episodes)
    sectors = {(p["position"][0]//2, p["position"][1]//2)
               for row in episodes for p in row["public_triggers"]}
    passed = triggering >= 12 and mean >= 3 and len(sectors) >= 2
    report = {"group": args.group, "starts": len(episodes),
              "starts_with_query": triggering, "mean_queries": mean,
              "sectors": len(sectors), "passed": passed, "episodes": episodes}
    save_once(destination / "gate.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "episodes"}), flush=True)


if __name__ == "__main__":
    main()
