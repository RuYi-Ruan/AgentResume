"""Freeze approved M18 D-v2: 100 fixed natural starts, exact-event uniqueness."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import torch

from ocres.m18_data import collect_episode
from ocres.m18_d_sampling import choose_natural_d
from ocres.m18_training import config_hash, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save, emit


def save_once(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"frozen D file differs on resume: {path}")
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
    revision_file = Path("configs/m18_d_v2.json")
    revision = json.loads(revision_file.read_text(encoding="utf-8"))
    if not (revision["source_offset_start"] == 5000 and revision["source_offset_end_inclusive"] == 5099
            and revision["events_per_group"] == 50 and revision["max_events_per_episode"] == 2
            and revision["exclude_cross_partition_exact_hash"] and revision["exclude_cross_partition_near_family"]
            and revision["exclude_within_D_exact_hash"] and not revision["exclude_within_D_near_family"]):
        raise RuntimeError("D addendum differs from user-approved revision")
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id
    source = root / f"group_{args.group}"
    destination = source / "natural_D"
    growth = json.loads((source / "result.json").read_text(encoding="utf-8"))
    history = json.loads((source / "history" / "impressions.json").read_text(encoding="utf-8"))
    a = json.loads((source / "controlled_A" / "manifest.json").read_text(encoding="utf-8"))
    if not growth["passed"] or not history["passed"] or not a["passed"] or config_hash(config) != a["config_hash"]:
        raise RuntimeError("upstream M18 gate failed")
    artifacts = Path("artifacts/m18") / args.run_id
    train_manifest = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in train_manifest["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("training source differs")
    model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / "post.pt")
    model.eval()
    destination.mkdir(parents=True, exist_ok=True)
    frozen_start = {"group": args.group, "training_config_hash": config_hash(config),
                    "revision_hash": hashlib.sha256(revision_file.read_bytes()).hexdigest(),
                    "script_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "history_hash": hashlib.sha256((source / "history/impressions.json").read_bytes()).hexdigest(),
                    "controlled_A_hash": hashlib.sha256((source / "controlled_A/manifest.json").read_bytes()).hexdigest()}
    save_once(destination / "sampling_started.json", frozen_start)
    occupied_hashes = {s["event_hash"] for phase in ("stale", "updated") for s in history[phase]}
    occupied_families = {s["family"] for phase in ("stale", "updated") for s in history[phase]}
    for phase in ("development", "formal"):
        for row in json.loads((source / "controlled_A" / f"{phase}_public.json").read_text(encoding="utf-8")):
            occupied_hashes.add(row["event_hash"])
            occupied_families.add(row["family"])
    candidates, counts = [], Counter()
    source_seeds = [group["episode_base"]+offset for offset in range(
        revision["source_offset_start"], revision["source_offset_end_inclusive"]+1)]
    for seed in source_seeds:
        target = destination / "episodes" / f"{seed}.json"
        if target.exists():
            episode = json.loads(target.read_text(encoding="utf-8"))
            if episode["seed"] != seed or len(episode["replay"]) != config["horizon"]:
                raise RuntimeError(f"saved D episode incomplete: {target}")
        else:
            episode = collect_episode(group["rows"], seed, model, spec, map_name=group["map"])
            save(target, episode)
            emit(root / "progress.jsonl", {"stage": "natural_D", "group": args.group, "seed": seed,
                                           "visible_decisions": len(episode["decisions"])})
        counts["visible_task_starts"] += len(episode["decisions"])
        for row in episode["decisions"]:
            if row["event_hash"] in occupied_hashes or row["family"] in occupied_families:
                counts["history_or_A_overlap"] += 1
            else:
                candidates.append({**row, "scoring_only": {"post": row["scoring_only"]},
                                   "source_kind": "natural_post_alice_task_start"})
    counts["eligible_public_hashes"] = len({row["event_hash"] for row in candidates})
    counts["eligible_near_families"] = len({row["family"] for row in candidates})
    chosen, exclusions = choose_natural_d(candidates, revision["events_per_group"],
                                          revision["max_events_per_episode"],
                                          revision["selection_seed_base"]+args.group)
    counts.update(exclusions)
    hashes = {row["event_hash"] for row in chosen}
    per_seed = Counter(row["source_seed"] for row in chosen)
    if len(chosen) != len({row["event_hash"] for row in chosen}):
        raise AssertionError("D exact public event repeated")
    save_once(destination / "candidates.json", candidates)
    save_once(destination / "formal_scoring.json", chosen)
    save_once(destination / "formal_public.json", [{"event_hash": r["event_hash"], "family": r["family"],
                                                     "public_event": r["event"]} for r in chosen])
    manifest = {"group": args.group, "source_kind": "natural_post_alice_task_start",
                "config_hash": config_hash(config), "history_hash": a["history_hash"],
                "controlled_A_formal_hash": a["formal_hash"],
                "D_revision_hash": frozen_start["revision_hash"],
                "source_seed_count": len(source_seeds), "source_seed_range": [source_seeds[0], source_seeds[-1]],
                "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                  for p in ("src/ocres/m18_data.py", "src/ocres/m18_d_sampling.py",
                                            "src/ocres/m18_bob.py", __file__,
                                            str(revision_file))},
                "counts": dict(counts), "eligible_after_cross_isolation": len(candidates),
                "selected": len(chosen), "required": revision["events_per_group"],
                "passed": len(chosen) == revision["events_per_group"],
                "selected_source_episodes": len(per_seed),
                "selected_public_hashes": len(hashes),
                "selected_near_families": len({r["family"] for r in chosen}),
                "selected_family_contributions": dict(Counter(r["family"] for r in chosen)),
                "intent_counts": dict(Counter(r["scoring_only"]["post"]["intent"] for r in chosen))}
    save_once(destination / "manifest.json", manifest)
    print(json.dumps({"D_gate": manifest["passed"], "group": args.group,
                      "selected": manifest["selected"], "intent_counts": manifest["intent_counts"]}), flush=True)


if __name__ == "__main__":
    main()
