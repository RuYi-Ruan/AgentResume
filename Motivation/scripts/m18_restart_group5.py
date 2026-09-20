"""Explicit, resumable PPO restart for interrupted M18 group 5.

The interrupted 44/50 run is left untouched. This is a new stochastic PPO run
from its frozen pre checkpoint, not an exact continuation of update 44.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import numpy as np
import torch

from m18_train import emit, save
from ocres.m18_training import M18TrainingEnv, config_hash, episode_seeds, group_spec, load_config
from ocres.ppo import ValueNetwork, clone_state_dict, collect_rollouts, evaluate_policy, ppo_update
from ocres.trainable import load_policy_checkpoint, save_policy_checkpoint, seed_everything, select_device


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_torch_save(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def rng_snapshot():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    torch.set_num_threads(1)
    config = load_config()
    group = 5
    metadata = group_spec(config, group)
    output = Path("data/m18") / args.run_id
    base = output / "group_5"
    artifacts = Path("artifacts/m18") / args.run_id
    checkpoint_dir = artifacts / "checkpoints/group_5"
    pre_path, post_path = checkpoint_dir / "pre.pt", checkpoint_dir / "post.pt"
    previous_path = base / "training_progress.json"
    result_path = base / "result.json"
    manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != config_hash(config):
        raise RuntimeError("frozen M18 config hash differs")
    for path, expected in manifest["source_hashes"].items():
        if file_hash(path) != expected:
            raise RuntimeError(f"frozen M18 training source changed: {path}")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if (previous["group"] != metadata or previous["config_hash"] != config_hash(config)
            or len(previous["updates"]) != 44 or previous["updates"][-1]["update"] != 44):
        raise RuntimeError("interrupted PPO record differs from audited 44/50 state")
    if result_path.exists() or post_path.exists():
        raise RuntimeError("a group 5 final result or post checkpoint already exists")
    recovery = base / "ppo_restart_20260916"
    recovery.mkdir(exist_ok=True)
    start_path = recovery / "started.json"
    source = {"group": group, "restart_from": "frozen_pre_checkpoint",
              "original_ppo_updates_retained": 44, "resume_semantics": "new_stochastic_ppo_run",
              "rng_seed": metadata["training_seed"] + 100000,
              "ppo_source_episodes": episode_seeds(config, group, "ppo"),
              "pre_sha256": file_hash(pre_path), "original_progress_sha256": file_hash(previous_path),
              "config_hash": config_hash(config), "script_sha256": file_hash(__file__)}
    if start_path.exists():
        if json.loads(start_path.read_text(encoding="utf-8")) != source:
            raise RuntimeError("recovery source fingerprint differs")
    else:
        save(start_path, source)
    device = select_device(args.device)
    actor, spec, extra = load_policy_checkpoint(pre_path)
    if extra != metadata:
        raise RuntimeError("pre checkpoint does not belong to frozen group 5")
    actor = actor.to(device)
    pre_state = clone_state_dict(actor)
    factory = lambda model, obs_spec, dev, horizon: M18TrainingEnv(
        model, obs_spec, dev, metadata["rows"], horizon, config["max_goal_ticks"])
    options = config["ppo"]
    val_seeds = episode_seeds(config, group, "validation")
    train_seeds = episode_seeds(config, group, "ppo")
    state_path = recovery / "state.pt"
    progress_path = recovery / "training_progress.json"
    started = time.monotonic()
    if state_path.exists():
        state = torch.load(state_path, map_location="cpu")
        if state["source"] != source or state["device"] != str(device):
            raise RuntimeError("PPO recovery state has different source or device")
        seed_everything(source["rng_seed"])
        critic = ValueNetwork(spec.dim, actor.hidden_dim).to(device)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=options["lr"])
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        result, best_state, best_update, best_val = (state[key] for key in
                                                    ("result", "best_state", "best_update", "best_val"))
        update_from = state["update"] + 1
        if progress_path.exists():
            old_progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if len(old_progress["updates"]) > state["update"]:
                raise RuntimeError("recovery JSON is ahead of the resumable state")
        save(progress_path, result)
        emit(output / "progress.jsonl", {"stage": "ppo_restart_resumed", "group": group,
                                        "from_complete_update": state["update"]})
    else:
        if progress_path.exists():
            raise RuntimeError("recovery progress exists without resumable state")
        seed_everything(source["rng_seed"])
        best_val = evaluate_policy(actor, spec, device, val_seeds, config["horizon"], factory)
        best_state, best_update = clone_state_dict(actor), 0
        critic = ValueNetwork(spec.dim, actor.hidden_dim).to(device)
        optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=options["lr"])
        result = {key: value for key, value in previous.items() if key not in ("validation", "updates")}
        result.update(validation=[{"update": 0, **best_val}], updates=[], recovery=source, passed=False)
        update_from = 1
        atomic_torch_save(state_path, {"source": source, "device": str(device), "update": 0,
                                       "actor": clone_state_dict(actor), "critic": clone_state_dict(critic),
                                       "optimizer": optimizer.state_dict(), "best_state": best_state,
                                       "best_update": best_update, "best_val": best_val,
                                       "rng": rng_snapshot(), "result": result})
        save(progress_path, result)
        emit(output / "progress.jsonl", {"stage": "ppo_restart_started", "group": group,
                                        "pre_validation_soups": best_val["delivery_mean"]})
    for update in range(update_from, options["updates"] + 1):
        first = (update - 1) * options["episodes_per_update"]
        seeds = train_seeds[first:first + options["episodes_per_update"]]
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
        atomic_torch_save(state_path, {"source": source, "device": str(device), "update": update,
                                       "actor": clone_state_dict(actor), "critic": clone_state_dict(critic),
                                       "optimizer": optimizer.state_dict(), "best_state": best_state,
                                       "best_update": best_update, "best_val": best_val,
                                       "rng": rng_snapshot(), "result": result})
        save(progress_path, result)
        emit(output / "progress.jsonl", {"stage": "ppo_restart", "group": group, "update": update,
                                        "rollout_soups": float(np.mean([e["deliveries"] for e in episodes])),
                                        "best_validation": best_val["delivery_mean"],
                                        "best_update": best_update})
    actor.load_state_dict(best_state)
    temp_post = post_path.with_suffix(".pt.tmp")
    save_policy_checkpoint(temp_post, actor, spec, extra={**metadata, "selected_update": best_update,
                                                        "recovery": "ppo_restart_20260916"})
    temp_post.replace(post_path)
    test_seeds = episode_seeds(config, group, "capability_test")
    post = evaluate_policy(actor, spec, device, test_seeds, config["horizon"], factory)
    actor.load_state_dict(pre_state)
    pre = evaluate_policy(actor, spec, device, test_seeds, config["horizon"], factory)
    differences = [b["deliveries"] - a["deliveries"] for a, b in zip(pre["rows"], post["rows"])]
    gain = float(np.mean(differences))
    improved = sum(value > 0 for value in differences)
    passed = (gain >= config["growth_gate"]["minimum_mean_gain_soups"] and
              improved >= config["growth_gate"]["minimum_improved_starts"])
    result.update(status="complete" if passed else "blocked_growth_gate", passed=passed,
                  pre_test=pre, post_test=post, paired_differences=differences,
                  mean_gain=gain, improved_starts=improved, selected_update=best_update,
                  recovery_elapsed_seconds=time.monotonic() - started)
    save(result_path, result)
    emit(output / "progress.jsonl", {"stage": "ppo_restart_growth_gate", "group": group,
                                    "passed": passed, "mean_gain": gain, "improved": improved})
    print(json.dumps({"group": group, "passed": passed, "mean_gain": gain,
                      "improved_starts": improved, "recovery": "new_stochastic_ppo_run"}), flush=True)


if __name__ == "__main__":
    main()
