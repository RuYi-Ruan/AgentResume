"""M18 B formal paired natural-start collaboration with resumable Qwen calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, "src")
import torch

from ocres.m18_b import run_b_episode
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_live_qwen import LiveQwen
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
    source = root / f"group_{args.group}"
    gate_file = root / "B_v5_acceptance.json"
    gate = json.loads(gate_file.read_text(encoding="utf-8"))
    history_file = source / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    if not gate["passed"] or not history["passed"]:
        raise RuntimeError("B development or visible history gate failed")
    artifacts = Path("artifacts/m18") / args.run_id
    training = json.loads((artifacts / "manifest.json").read_text(encoding="utf-8"))
    for path, expected in training["source_hashes"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise RuntimeError("training source changed")
    a = json.loads((source / "controlled_A/manifest.json").read_text(encoding="utf-8"))
    if not a["passed"] or hashlib.sha256(Path("src/ocres/m18_bob.py").read_bytes()).hexdigest() != a["bob_protocol_hash"]:
        raise RuntimeError("Bob protocol differs from approved A")
    model, spec, _ = load_policy_checkpoint(artifacts / "checkpoints" / f"group_{args.group}" / "post.pt")
    model.eval()
    destination = source / "B_formal"
    destination.mkdir(parents=True, exist_ok=True)
    source_files = ["src/ocres/m18_b.py", "src/ocres/m18_live_qwen.py",
                    "src/ocres/m18_prompt.py", "src/ocres/m18_qwen.py", "src/ocres/m18_bob.py",
                    "src/ocres/m18_bob_v5.py", "configs/m18_b_v5.json"]
    manifest = {"group": args.group, "config_hash": config_hash(config),
                "history_hash": hashlib.sha256(history_file.read_bytes()).hexdigest(),
                "bob_protocol_hash": a["bob_protocol_hash"],
                "bob_controller": PublicBobV5.version,
                "bob_controller_acceptance_hash": hashlib.sha256(gate_file.read_bytes()).hexdigest(),
                "source_hashes": {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_files},
                "model": config["llm"]["model"], "prompt_version": "v2_mapfix",
                "seeds": list(episode_seeds(config, args.group, "B"))}
    record = destination / "manifest.json"
    if record.exists():
        if json.loads(record.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("B source or config differs on resume")
    else:
        save(record, manifest)
    live = LiveQwen(group["rows"], history, config["llm"], destination / "qwen",
                    root / "api_budget.jsonl", args.group)
    order = [(seed, condition) for seed in manifest["seeds"]
             for condition in config["llm"]["conditions"]+["oracle"]]
    random.Random(18988+args.group).shuffle(order)
    for seed, condition in order:
        target = destination / "episodes" / f"{seed}_{condition}.json"
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if old["seed"] != seed or old["condition"] != condition:
                raise RuntimeError("saved B episode differs")
            continue
        result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                               predict=live.predict if condition != "oracle" else None,
                               bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": "B_formal", "group": args.group,
                                       "seed": seed, "condition": condition,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
        print(json.dumps({"group": args.group, "seed": seed, "condition": condition,
                          "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)
    print(json.dumps({"group": args.group, "B_formal_complete": True,
                      "episodes": len(order)}), flush=True)


if __name__ == "__main__":
    main()
