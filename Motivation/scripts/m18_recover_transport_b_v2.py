"""Corrected B-only recovery after recovery1 failed before its healthcheck."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, "src")
import torch

from m18_recover_transport import (save_exact, sha256, transport_failures,
                                   transport_healthcheck)
from m18_train import emit, save
from ocres.m18_b import run_b_episode
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_live_qwen import LiveQwen
from ocres.m18_prompt import build_prompt
from ocres.m18_qwen import trim_prompt
from ocres.m18_training import episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


TAG = "transport_recovery2"
B_GROUP = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    config = load_config()
    source = root / f"group_{B_GROUP}"
    original = source / "B_formal"
    original_responses = original / "qwen/responses.jsonl"
    outage_failures = transport_failures(original_responses)
    if len(outage_failures) < 500:
        raise RuntimeError("B group 0 no longer matches the audited transport outage")

    original_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in original_manifest["source_hashes"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"B frozen source changed: {name}")

    group = group_spec(config, B_GROUP)
    history_file = source / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    artifacts = Path("artifacts/m18") / root.name
    model, spec, _ = load_policy_checkpoint(
        artifacts / "checkpoints" / f"group_{B_GROUP}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)

    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    seeds = list(episode_seeds(config, B_GROUP, "B"))
    failed_recovery1 = source / "B_formal_transport_recovery1"
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "original B group 0 contained transport-failed Qwen predictions",
        "seeds": seeds,
        "original_manifest_hash": sha256(original / "manifest.json"),
        "original_responses_hash": sha256(original_responses),
        "original_transport_failures": len(outage_failures),
        "failed_recovery1_preserved": failed_recovery1.exists(),
        "failed_recovery1_manifest_hash": (sha256(failed_recovery1 / "manifest.json")
                                             if (failed_recovery1 / "manifest.json").exists() else None),
        "recovery_script_hash": sha256(Path(__file__)),
    }
    save_exact(destination / "manifest.json", manifest)

    probe_event = json.loads((source / "controlled_A/formal_public.json")
                             .read_text(encoding="utf-8"))[0]["public_event"]
    probe_messages, _, _ = build_prompt(probe_event, history["stale"],
                                        group["rows"], "v2_mapfix")
    probe_messages, _ = trim_prompt(probe_messages, config["llm"]["max_input_tokens"])
    transport_healthcheck(probe_messages, config["llm"],
                          destination / "transport_healthcheck.jsonl",
                          root / "api_budget.jsonl", "B_group0_recovery2")

    live = LiveQwen(group["rows"], history, config["llm"], destination / "qwen",
                    root / "api_budget.jsonl", B_GROUP)
    order = [(seed, condition) for seed in seeds
             for condition in config["llm"]["conditions"] + ["oracle"]]
    random.Random(18988 + B_GROUP).shuffle(order)
    for seed, condition in order:
        target = destination / "episodes" / f"{seed}_{condition}.json"
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if old["seed"] != seed or old["condition"] != condition:
                raise RuntimeError("saved recovery B episode differs")
            continue
        result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                               predict=live.predict if condition != "oracle" else None,
                               bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}",
                                       "group": B_GROUP, "seed": seed,
                                       "condition": condition,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
        print(json.dumps({"recovery": "B", "group": B_GROUP, "seed": seed,
                          "condition": condition, "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)
    print(json.dumps({"recovery": "B", "group": B_GROUP,
                      "episodes": len(order), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
