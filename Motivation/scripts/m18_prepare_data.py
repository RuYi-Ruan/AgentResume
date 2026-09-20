"""Local per-group M18 visibility/history/pair feasibility, never calls Qwen."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

import torch

from ocres.m18_data import collect_episode, disjoint_candidates, paired_histories, select_balanced
from ocres.m18_training import config_hash, episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save, emit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--stage", choices=("history", "pairs"), default="history")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run-id")
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id
    artifacts = Path("artifacts/m18") / args.run_id
    growth = json.loads((root / f"group_{args.group}" / "result.json").read_text(encoding="utf-8"))
    if not growth["passed"] or growth["config_hash"] != config_hash(config):
        raise RuntimeError("capability gate did not pass")
    train_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in train_manifest["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("training source changed; checkpoint cannot be silently reused")
    directory = root / f"group_{args.group}" / args.stage
    if directory.exists():
        raise RuntimeError("stage already exists; inspect before retrying")
    directory.mkdir(parents=True)
    sources = ["src/ocres/m18_bob.py", "src/ocres/m18_data.py", "scripts/m18_prepare_data.py"]
    save(directory / "manifest.json", {"config_hash": config_hash(config),
         "sources": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources}})
    models = []
    for phase in ("pre", "post"):
        model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / f"{phase}.pt")
        model.eval()
        models.append(model)
    partitions = ("history",) if args.stage == "history" else ("development_sources", "A_sources")
    phases = ("pre", "post")
    all_rows = {}
    for partition in partitions:
        phase_rows = {}
        for phase, model in zip(phases, models):
            episodes = []
            for seed in episode_seeds(config, args.group, partition):
                episode = collect_episode(group["rows"], seed, model, spec,
                                          paired_models=models if args.stage == "pairs" else None,
                                          map_name=group["map"])
                save(directory / partition / phase / f"{seed}.json", episode)
                # Keep bounded summary; replay and snapshots already persisted above.
                episodes.append({key: value for key, value in episode.items() if key not in ("replay", "decisions")})
                emit(root / "progress.jsonl", {"stage": args.stage, "group": args.group, "partition": partition,
                                               "phase": phase, "seed": seed, "soups": episode["deliveries"],
                                               "segments": len(episode["segments"]), "pairs": len(episode["pairs"])})
            phase_rows[phase] = episodes
        all_rows[partition] = phase_rows
    if args.stage == "history":
        histories = paired_histories(all_rows["history"]["pre"], all_rows["history"]["post"])
        save(directory / "impressions.json", histories)
        print(json.dumps({"history_gate": histories["passed"], "segments_per_phase": len(histories["stale"]),
                          "paired_sources": len(histories["accepted_sources"])}), flush=True)
    else:
        history = json.loads((directory.parent / "history" / "impressions.json").read_text(encoding="utf-8"))
        if not history["passed"]:
            raise RuntimeError("history gate failed")
        hashes = {s["event_hash"] for key in ("stale", "updated") for s in history[key]}
        families = {s["family"] for key in ("stale", "updated") for s in history[key]}
        report = {"partitions": {}, "passed": True}
        for partition, requested in (("development_sources", 20), ("A_sources", 50)):
            raw = [row for phase in phases for ep in all_rows[partition][phase] for row in ep["pairs"]]
            available, excluded = disjoint_candidates(raw, hashes, families)
            selected = select_balanced(available, requested, 18000+args.group)
            report["partitions"][partition] = {
                "raw_count": len(raw), "available": len(available), "excluded": excluded,
                "selected_count": len(selected), "required": requested,
                "post_intent_counts": dict(Counter(r["scoring_only"]["post"]["intent"] for r in selected)),
                "selected": selected,
            }
            # All development candidates are consumed development data, not just selected ones.
            hashes.update(row["event_hash"] for row in raw)
            families.update(row["family"] for row in raw)
            report["passed"] &= len(selected) == requested
        save(directory / "split_report.json", report)
        print(json.dumps({"pair_gate": report["passed"],
                          "counts": {p: v["selected_count"] for p,v in report["partitions"].items()}}), flush=True)


if __name__ == "__main__":
    main()
