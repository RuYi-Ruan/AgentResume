"""Resume an interrupted M18 history without deleting or overwriting episodes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import torch

from ocres.m18_data import collect_episode, paired_histories
from ocres.m18_training import config_hash, episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save, emit


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
    stage = root / f"group_{args.group}" / "history"
    manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_hash"] != config_hash(config):
        raise RuntimeError("partial history config differs")
    for path, expected in manifest["sources"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("partial history source differs")
    growth = json.loads((stage.parent / "result.json").read_text(encoding="utf-8"))
    if not growth["passed"]:
        raise RuntimeError("Alice capability failed")
    artifacts = Path("artifacts/m18") / args.run_id
    sources = {}
    for phase in ("pre", "post"):
        model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / f"{phase}.pt")
        model.eval()
        rows = []
        for seed in episode_seeds(config, args.group, "history"):
            target = stage / "history" / phase / f"{seed}.json"
            if target.exists():
                episode = json.loads(target.read_text(encoding="utf-8"))
                if episode["seed"] != seed or len(episode["replay"]) != config["horizon"]:
                    raise RuntimeError(f"partial saved episode invalid: {target}")
            else:
                episode = collect_episode(group["rows"], seed, model, spec, map_name=group["map"])
                save(target, episode)
                emit(root / "progress.jsonl", {"stage": "history_resume", "group": args.group,
                                               "phase": phase, "seed": seed,
                                               "soups": episode["deliveries"]})
            rows.append({key: value for key, value in episode.items() if key not in ("replay", "decisions")})
        sources[phase] = rows
    histories = paired_histories(sources["pre"], sources["post"])
    output = stage / "impressions.json"
    if output.exists():
        raise RuntimeError("impressions already exist; refusing overwrite")
    save(output, histories)
    save(stage / "resume_record.json", {"reason": "interrupted planner cache read",
                                        "resumer_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                        "history_gate": histories["passed"]})
    print(json.dumps({"history_gate": histories["passed"],
                      "segments_per_phase": len(histories["stale"]),
                      "paired_sources": len(histories["accepted_sources"])}), flush=True)


if __name__ == "__main__":
    main()
