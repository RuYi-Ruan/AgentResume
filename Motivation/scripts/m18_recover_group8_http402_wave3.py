"""Recover four final group-8 episodes affected by the late HTTP 402 wave."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import sys

sys.path.insert(0, "src")
import torch

from m18_recover_transport import save_exact, sha256, transport_healthcheck
from m18_train import emit, save
from ocres.m18_b import run_b_episode
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_live_qwen import LiveQwen
from ocres.m18_prompt import build_prompt
from ocres.m18_qwen import read_jsonl, trim_prompt
from ocres.m18_training import config_hash, episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


GROUP = 8
TAG = "anomaly_recovery3"
OLD_LIMIT = 28000
NEW_LIMIT = 29000
SELECTED_NAMES = {
    "906001_none", "906012_updated", "906007_updated", "906013_stale",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path("data/m18") / args.run_id
    source = root / f"group_{GROUP}"
    original = source / "B_formal"
    original_responses = original / "qwen/responses.jsonl"
    config = load_config()
    if config["llm"]["request_limit"] != OLD_LIMIT:
        raise RuntimeError("frozen request limit unexpectedly changed")
    options = {**config["llm"], "request_limit": NEW_LIMIT}

    failures = {}
    pattern = re.compile(r"^B:g8:s(?P<seed>\d+):c(?P<condition>[^:]+):q\d+:r\d+$")
    for row in read_jsonl(original_responses):
        if row.get("kind") != "completion" or "http_402" not in row.get("errors", []):
            continue
        match = pattern.match(row["id"])
        if match is None:
            raise RuntimeError(f"unexpected failed request ID: {row['id']}")
        name = f"{match.group('seed')}_{match.group('condition')}"
        failures[name] = failures.get(name, 0) + 1
    if not SELECTED_NAMES <= failures.keys():
        raise RuntimeError(f"recovery set is not backed by HTTP 402 failures: {failures}")
    for name in SELECTED_NAMES:
        if not (original / "episodes" / f"{name}.json").exists():
            raise RuntimeError(f"failed original episode is missing: {name}")

    original_manifest_path = original / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    if original_manifest["config_hash"] != config_hash(config):
        raise RuntimeError("frozen B config differs")
    for name, expected in original_manifest["source_hashes"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"B frozen source changed: {name}")

    group = group_spec(config, GROUP)
    history_file = source / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    model, spec, _ = load_policy_checkpoint(
        Path("artifacts/m18") / args.run_id / "checkpoints" / f"group_{GROUP}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)
    seeds = list(episode_seeds(config, GROUP, "B"))
    order = [(seed, condition) for seed in seeds
             for condition in config["llm"]["conditions"] + ["oracle"]]
    random.Random(18988 + GROUP).shuffle(order)
    selected = [(seed, condition) for seed, condition in order
                if f"{seed}_{condition}" in SELECTED_NAMES]
    if len(selected) != len(SELECTED_NAMES):
        raise RuntimeError("recovery set is outside the frozen B design")

    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "four final episodes affected by the late HTTP 402 wave",
        "selected_episodes": [f"{seed}_{condition}" for seed, condition in selected],
        "failure_counts": {name: failures[name] for name in sorted(SELECTED_NAMES)},
        "original_manifest_hash": sha256(original_manifest_path),
        "original_responses_hash": sha256(original_responses),
        "approved_request_limit_before": OLD_LIMIT,
        "approved_request_limit_after": NEW_LIMIT,
        "recovery_script_hash": sha256(Path(__file__)),
    }
    save_exact(destination / "manifest.json", manifest)

    probe_event = json.loads((source / "controlled_A/formal_public.json")
                             .read_text(encoding="utf-8"))[0]["public_event"]
    probe_messages, _, _ = build_prompt(probe_event, history["stale"],
                                        group["rows"], "v2_mapfix")
    probe_messages, _ = trim_prompt(probe_messages, options["max_input_tokens"])
    transport_healthcheck(probe_messages, options,
                          destination / "transport_healthcheck.jsonl",
                          root / "api_budget.jsonl", "B_group8_anomaly_recovery3")
    live = LiveQwen(group["rows"], history, options, destination / "qwen",
                    root / "api_budget.jsonl", GROUP)
    selection = {"group": GROUP, "recovery": TAG, "episodes": {}}
    for seed, condition in selected:
        name = f"{seed}_{condition}"
        target = destination / "episodes" / f"{name}.json"
        if target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
        else:
            result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                                   predict=live.predict, bob_class=PublicBobV5)
            save(target, result)
            emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}", "group": GROUP,
                                           "seed": seed, "condition": condition,
                                           "score_300": result["score_first_300"],
                                           "queries": result["query_count"]})
        selection["episodes"][name] = {
            "selected_path": str(target), "selected_hash": sha256(target),
            "original_path": str(original / "episodes" / f"{name}.json"),
            "original_hash": sha256(original / "episodes" / f"{name}.json"),
        }
        print(json.dumps({"recovery": TAG, "seed": seed, "condition": condition,
                          "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)
    save_exact(source / f"B_formal_{TAG}_selection.json", selection)
    print(json.dumps({"recovery": TAG, "episodes": len(selected),
                      "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
