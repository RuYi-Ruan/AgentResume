"""Recover the one B group-6 episode interrupted by the second HTTP 402 outage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil
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
from ocres.m18_training import episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


GROUP = 6
TAG = "http402_recovery1"
ID_RE = re.compile(r"^B:g6:s(?P<seed>\d+):c(?P<condition>[^:]+):q\d+:r\d+$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")

    root = Path("data/m18") / args.run_id
    source = root / f"group_{GROUP}"
    original = source / "B_formal"
    original_responses = original / "qwen/responses.jsonl"
    pairs: set[tuple[int, str]] = set()
    failure_count = 0
    for record in read_jsonl(original_responses):
        if record.get("kind") != "completion" or "http_402" not in record.get("errors", []):
            continue
        failure_count += 1
        match = ID_RE.match(record["id"])
        if match is None:
            raise RuntimeError(f"unexpected HTTP 402 request ID: {record['id']}")
        pairs.add((int(match.group("seed")), match.group("condition")))
    if len(pairs) != 1 or failure_count != 9:
        raise RuntimeError(f"expected one episode and 9 HTTP 402 completions, got {pairs}, {failure_count}")

    config = load_config()
    group = group_spec(config, GROUP)
    seeds = list(episode_seeds(config, GROUP, "B"))
    allowed = {(seed, condition) for seed in seeds
               for condition in config["llm"]["conditions"] + ["oracle"]}
    if not pairs <= allowed or any(condition == "oracle" for _, condition in pairs):
        raise RuntimeError("HTTP 402 recovery set is outside the frozen B design")

    original_manifest_path = original / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    for name, expected in original_manifest["source_hashes"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"B frozen source changed: {name}")

    history_file = source / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    artifacts = Path("artifacts/m18") / args.run_id
    model, spec, _ = load_policy_checkpoint(
        artifacts / "checkpoints" / f"group_{GROUP}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)

    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    order = [(seed, condition) for seed in seeds
             for condition in config["llm"]["conditions"] + ["oracle"]]
    random.Random(18988 + GROUP).shuffle(order)
    selected = [(seed, condition) for seed, condition in order if (seed, condition) in pairs]
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "one B group-6 episode interrupted by the second HTTP 402 outage",
        "selected_episodes": [f"{seed}_{condition}" for seed, condition in selected],
        "original_manifest_hash": sha256(original_manifest_path),
        "original_responses_hash": sha256(original_responses),
        "original_http402_completions": failure_count,
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
                          root / "api_budget.jsonl", "B_group6_http402_recovery1")

    live = LiveQwen(group["rows"], history, config["llm"], destination / "qwen",
                    root / "api_budget.jsonl", GROUP)
    seed, condition = selected[0]
    target = destination / "episodes" / f"{seed}_{condition}.json"
    if not target.exists():
        result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                               predict=live.predict, bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}", "group": GROUP,
                                       "seed": seed, "condition": condition,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
    else:
        result = json.loads(target.read_text(encoding="utf-8"))
        if result["seed"] != seed or result["condition"] != condition:
            raise RuntimeError("saved HTTP 402 recovery episode differs")

    bridge = original / "episodes" / target.name
    if bridge.exists():
        if sha256(bridge) != sha256(target):
            raise RuntimeError("resume bridge differs from selected recovery episode")
    else:
        bridge.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, bridge)
    selection = {
        "group": GROUP,
        "recovery": TAG,
        "reason": manifest["reason"],
        "selected_path": str(target),
        "selected_hash": sha256(target),
        "resume_bridge": str(bridge),
        "resume_bridge_hash": sha256(bridge),
    }
    save_exact(source / f"B_formal_{TAG}_selection.json", selection)
    print(json.dumps({"recovery": "B_http402", "group": GROUP, "seed": seed,
                      "condition": condition, "score_300": result["score_first_300"],
                      "queries": result["query_count"], "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
