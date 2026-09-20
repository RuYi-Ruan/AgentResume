"""Recover only B group-4 episodes touched by the 2026-09-18 HTTP 402 outage."""
from __future__ import annotations

import argparse
import hashlib
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


GROUP = 4
TAG = "http402_recovery1"
ID_RE = re.compile(r"^B:g4:s(?P<seed>\d+):c(?P<condition>[^:]+):q\d+:r\d+$")


def affected_pairs(path: Path) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    for record in read_jsonl(path):
        if record.get("kind") != "completion" or "http_402" not in record.get("errors", []):
            continue
        match = ID_RE.match(record["id"])
        if match is None:
            raise RuntimeError(f"unexpected HTTP 402 request ID: {record['id']}")
        pairs.add((int(match.group("seed")), match.group("condition")))
    return pairs


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
    pairs = affected_pairs(original_responses)
    if len(pairs) != 7:
        raise RuntimeError(f"expected 7 HTTP-402-affected episodes, found {len(pairs)}")

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
        "reason": "seven B group-4 episodes touched by the HTTP 402 outage",
        "selected_episodes": [f"{seed}_{condition}" for seed, condition in selected],
        "original_manifest_hash": sha256(original_manifest_path),
        "original_responses_hash": sha256(original_responses),
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
                          root / "api_budget.jsonl", "B_group4_http402_recovery1")

    live = LiveQwen(group["rows"], history, config["llm"], destination / "qwen",
                    root / "api_budget.jsonl", GROUP)
    for seed, condition in selected:
        target = destination / "episodes" / f"{seed}_{condition}.json"
        if target.exists():
            old = json.loads(target.read_text(encoding="utf-8"))
            if old["seed"] != seed or old["condition"] != condition:
                raise RuntimeError("saved HTTP 402 recovery episode differs")
            continue
        result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                               predict=live.predict, bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}", "group": GROUP,
                                       "seed": seed, "condition": condition,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
        print(json.dumps({"recovery": "B_http402", "group": GROUP, "seed": seed,
                          "condition": condition, "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)

    selection = {
        "group": GROUP,
        "recovery": TAG,
        "reason": manifest["reason"],
        "episodes": {},
    }
    for seed, condition in selected:
        name = f"{seed}_{condition}.json"
        recovered = destination / "episodes" / name
        selection["episodes"][name] = {
            "selected_path": str(recovered),
            "selected_hash": sha256(recovered),
            "original_path": str(original / "episodes" / name),
            "original_preserved": (original / "episodes" / name).exists(),
        }

    # One interrupted episode had no original result. Put an identical bridge copy in
    # the normal episode folder so the frozen resumable runner skips it. The recovery
    # directory remains the explicitly selected source for final analysis.
    missing = [name for name, item in selection["episodes"].items()
               if not item["original_preserved"]]
    if len(missing) != 1:
        raise RuntimeError(f"expected one interrupted episode without a result, found {missing}")
    bridge = original / "episodes" / missing[0]
    recovered = destination / "episodes" / missing[0]
    if bridge.exists():
        if sha256(bridge) != sha256(recovered):
            raise RuntimeError("resume bridge differs from selected recovery episode")
    else:
        bridge.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(recovered, bridge)
    selection["resume_bridge"] = {"path": str(bridge), "hash": sha256(bridge)}
    save_exact(source / f"B_formal_{TAG}_selection.json", selection)
    print(json.dumps({"recovery": "B_http402", "group": GROUP,
                      "episodes": len(selected), "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
