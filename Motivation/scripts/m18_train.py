"""Approved M18 local preflight and sequential independent Alice training.

Run with the agentresume interpreter from the repository root. No API calls.
Artifacts are append-only per run ID; interrupted groups are never silently retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch

from ocres.m18_training import (
    M18TrainingEnv, classification_report, collect_expert, config_hash, episode_seeds,
    geometry_report, group_spec, load_config, observation_spec, train_bc,
)
from ocres.ppo import ValueNetwork, clone_state_dict, collect_rollouts, evaluate_policy, ppo_update
from ocres.trainable import INTENTS, MacroIntentPolicy, save_policy_checkpoint, seed_everything, select_device


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def emit(path, data):
    item = {"utc": datetime.now(timezone.utc).isoformat(), **data}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps(item, ensure_ascii=False), flush=True)


def preflight(config, output):
    result = {"config_hash": config_hash(config), "stage": "preflight", "maps": {}}
    for group in (0, 3, 6):
        spec = group_spec(config, group)
        counts, episodes = Counter(), []
        for seed in episode_seeds(config, group, "development"):
            _, metrics, trace = collect_expert(spec["rows"], seed, config["horizon"], keep_trace=True)
            counts.update(metrics["decision_counts"])
            episodes.append(metrics)
            save(output / "preflight_traces" / f"{spec['map']}_{seed}.json", trace)
            emit(output / "progress.jsonl", {"stage": "map_preflight", "map": spec["map"],
                                           "seed": seed, "soups": metrics["deliveries"]})
        missing = [name for name in INTENTS if not counts[name]]
        entry = {"geometry": geometry_report(spec["rows"]), "episode_rows": episodes,
                 "intent_counts": dict(counts), "missing_intents": missing,
                 "delivery_mean": float(np.mean([row["deliveries"] for row in episodes]))}
        entry["passed"] = (not missing and entry["geometry"]["connected"] and
                           entry["geometry"]["accessible"] and entry["delivery_mean"] > 0)
        result["maps"][spec["map"]] = entry
        save(output / "preflight.json", result)
    result["passed"] = all(row["passed"] for row in result["maps"].values())
    save(output / "preflight.json", result)
    return result


def train_group(config, group, output, artifact_dir, device):
    metadata = group_spec(config, group)
    directory = output / f"group_{group}"
    result_path = directory / "result.json"
    if directory.exists():
        raise RuntimeError(f"group directory already exists; inspect before resuming: {directory}")
    directory.mkdir(parents=True)
    started = time.monotonic()
    progress = lambda row: emit(output / "progress.jsonl", {"group": group, **row})
    save(directory / "started.json", {**metadata, "config_hash": config_hash(config)})
    seed_everything(metadata["training_seed"])
    samples, expert_metrics = [], []
    for seed in episode_seeds(config, group, "bc"):
        batch, metrics, trace = collect_expert(metadata["rows"], seed, config["horizon"], keep_trace=True)
        samples.extend(batch)
        expert_metrics.append(metrics)
        save(directory / "demonstrations" / f"{seed}.json", trace)
    counts = Counter(INTENTS[row[1]] for row in samples)
    result = {"group": metadata, "config_hash": config_hash(config), "expert": expert_metrics,
              "expert_intent_counts": dict(counts), "passed": False}
    missing = [name for name in INTENTS if not counts[name]]
    if missing:
        result.update(status="blocked_missing_demonstrations", missing_intents=missing)
        save(result_path, result)
        return result
    spec = observation_spec(metadata["rows"])
    # Scripted expert constructors allocate dummy networks; reset before the real initialization.
    seed_everything(metadata["training_seed"])
    actor = MacroIntentPolicy(spec.dim, hidden_dim=config["hidden_dim"]).to(device)
    result["bc"] = train_bc(config, actor, samples, device, metadata["training_seed"], progress)
    result["bc"]["training_classification"] = classification_report(actor, samples, device)
    checkpoint_dir = artifact_dir / "checkpoints" / f"group_{group}"
    save_policy_checkpoint(checkpoint_dir / "pre.pt", actor, spec, extra=metadata)
    pre_state = clone_state_dict(actor)
    factory = lambda model, obs_spec, dev, horizon: M18TrainingEnv(
        model, obs_spec, dev, metadata["rows"], horizon, config["max_goal_ticks"])
    val_seeds = episode_seeds(config, group, "validation")
    # No independent capability test is opened before checkpoint selection finishes.
    best_val = evaluate_policy(actor, spec, device, val_seeds, config["horizon"], factory)
    best_state, best_update = clone_state_dict(actor), 0
    result["validation"] = [{"update": 0, **best_val}]
    critic = ValueNetwork(spec.dim, actor.hidden_dim).to(device)
    options = config["ppo"]
    optimizer = torch.optim.Adam(list(actor.parameters())+list(critic.parameters()), lr=options["lr"])
    train_seeds = episode_seeds(config, group, "ppo")
    result["updates"] = []
    progress({"stage": "pre_validation", "soups": best_val["delivery_mean"]})
    for update in range(1, options["updates"]+1):
        first = (update-1)*options["episodes_per_update"]
        seeds = train_seeds[first:first+options["episodes_per_update"]]
        records, episodes = collect_rollouts(actor, critic, spec, device, seeds,
                                             config["horizon"], options["temperature"],
                                             options["gamma"], options["gae_lambda"], factory)
        losses = ppo_update(actor, critic, optimizer, records, device,
                            **{key: options[key] for key in ("temperature", "update_epochs", "batch_size",
                               "clip_ratio", "value_coef", "entropy_coef")})
        row = {"update": update, "rollouts": episodes, **losses}
        if update % options["eval_every"] == 0:
            val = evaluate_policy(actor, spec, device, val_seeds, config["horizon"], factory)
            result["validation"].append({"update": update, **val})
            row["validation_mean"] = val["delivery_mean"]
            if val["delivery_mean"] > best_val["delivery_mean"]:
                best_state, best_update, best_val = clone_state_dict(actor), update, val
        result["updates"].append(row)
        save(directory / "training_progress.json", result)
        progress({"stage": "ppo", "update": update,
                  "rollout_soups": float(np.mean([e["deliveries"] for e in episodes])),
                  "best_validation": best_val["delivery_mean"], "best_update": best_update})
    actor.load_state_dict(best_state)
    save_policy_checkpoint(checkpoint_dir / "post.pt", actor, spec,
                           extra={**metadata, "selected_update": best_update})
    test_seeds = episode_seeds(config, group, "capability_test")
    post = evaluate_policy(actor, spec, device, test_seeds, config["horizon"], factory)
    actor.load_state_dict(pre_state)
    pre = evaluate_policy(actor, spec, device, test_seeds, config["horizon"], factory)
    differences = [b["deliveries"]-a["deliveries"] for a,b in zip(pre["rows"], post["rows"])]
    mean_gain = float(np.mean(differences))
    improved = sum(value > 0 for value in differences)
    passed = (mean_gain >= config["growth_gate"]["minimum_mean_gain_soups"] and
              improved >= config["growth_gate"]["minimum_improved_starts"])
    result.update(status="complete" if passed else "blocked_growth_gate", passed=passed,
                  pre_test=pre, post_test=post, paired_differences=differences,
                  mean_gain=mean_gain, improved_starts=improved, selected_update=best_update,
                  elapsed_seconds=time.monotonic()-started)
    save(result_path, result)
    progress({"stage": "growth_gate", "passed": passed, "mean_gain": mean_gain, "improved": improved})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "train"))
    parser.add_argument("--config", default="configs/m18_v1.json")
    parser.add_argument("--run-id", default="v1_20260915")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(9)))
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("run-id must be alphanumeric with underscores")
    torch.set_num_threads(1)
    config = load_config(args.config)
    output = Path("data/m18") / args.run_id
    artifacts = Path("artifacts/m18") / args.run_id
    device = select_device(args.device)
    environment = {"python": sys.executable, "version": platform.python_version(),
                   "torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    sources = ["configs/m18_v1.json", "scripts/m18_train.py", "src/ocres/m18_training.py",
               "src/ocres/trainable.py", "src/ocres/generic_agents.py", "src/ocres/ppo.py",
               "src/ocres/grid.py", "src/ocres/executor.py"]
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in sources}
    manifest_path = artifacts / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["config_hash"] != config_hash(config) or old["source_hashes"] != hashes:
            raise RuntimeError("run configuration or source changed; use a new run ID")
    else:
        save(manifest_path, {"config": config, "config_hash": config_hash(config),
                             "environment": environment, "source_hashes": hashes})
        for source in sources:
            destination = artifacts / "source" / source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    emit(output / "progress.jsonl", {"stage": args.mode, "environment": environment})
    if args.mode == "preflight":
        if (output / "preflight.json").exists():
            raise RuntimeError("preflight already exists; inspect it instead of overwriting")
        result = preflight(config, output)
        print(json.dumps({"preflight_passed": result["passed"]}), flush=True)
    else:
        gate = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
        if not gate.get("passed") or gate["config_hash"] != config_hash(config):
            raise RuntimeError("preflight gate has not passed")
        for group in args.groups:
            result = train_group(config, group, output, artifacts, device)
            if not result["passed"]:
                print(json.dumps({"stopped_at_group": group, "status": result["status"]}), flush=True)
                break


if __name__ == "__main__":
    main()
