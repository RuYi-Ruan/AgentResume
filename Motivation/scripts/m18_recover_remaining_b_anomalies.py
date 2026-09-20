"""Recover the 12 remaining B episodes touched by transport/HTTP failures."""
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
from ocres.m18_training import episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


TAG = "anomaly_recovery1"
EXPECTED = {
    2: {"306005_none"},
    6: {"706000_none"},
    7: {
        "806001_stale", "806007_updated", "806010_none",
        "806004_updated", "806010_updated", "806012_updated",
    },
    8: {"906010_none", "906014_updated", "906008_updated", "906001_stale"},
}
TRANSPORT_ERRORS = {"URLError", "TimeoutError", "ConnectionError", "transport_exhausted"}


def failed_episode_names(path: Path, group: int) -> tuple[set[str], int]:
    names: set[str] = set()
    failures = 0
    pattern = re.compile(rf"^B:g{group}:s(?P<seed>\d+):c(?P<condition>[^:]+):q\d+:r\d+$")
    for row in read_jsonl(path):
        if row.get("kind") != "completion":
            continue
        errors = set(row.get("errors", []))
        if not (errors & TRANSPORT_ERRORS or any(str(e).startswith("http_") for e in errors)):
            continue
        failures += 1
        match = pattern.match(row["id"])
        if match is None:
            raise RuntimeError(f"unexpected failed request ID: {row['id']}")
        names.add(f"{match.group('seed')}_{match.group('condition')}")
    return names, failures


def recover_group(root: Path, config: dict, group_index: int,
                  selected_names: set[str]) -> dict:
    source = root / f"group_{group_index}"
    original = source / "B_formal"
    original_responses = original / "qwen/responses.jsonl"
    failed_names, failure_count = failed_episode_names(original_responses, group_index)
    if not selected_names <= failed_names:
        raise RuntimeError(
            f"group {group_index} recovery set is not backed by recorded failures: "
            f"selected={sorted(selected_names)}, failed={sorted(failed_names)}")
    for name in selected_names:
        if not (original / "episodes" / f"{name}.json").exists():
            raise RuntimeError(f"group {group_index} original failed episode is missing: {name}")

    original_manifest_path = original / "manifest.json"
    original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
    for name, expected in original_manifest["source_hashes"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"B frozen source changed: {name}")

    group = group_spec(config, group_index)
    history_file = source / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    artifacts = Path("artifacts/m18") / root.name
    model, spec, _ = load_policy_checkpoint(
        artifacts / "checkpoints" / f"group_{group_index}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)

    seeds = list(episode_seeds(config, group_index, "B"))
    full_order = [(seed, condition) for seed in seeds
                  for condition in config["llm"]["conditions"] + ["oracle"]]
    random.Random(18988 + group_index).shuffle(full_order)
    selected = [(seed, condition) for seed, condition in full_order
                if f"{seed}_{condition}" in selected_names]
    if len(selected) != len(selected_names):
        raise RuntimeError(f"group {group_index} recovery set is outside the frozen B design")

    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "episodes touched by recorded transport or HTTP failures",
        "selected_episodes": [f"{seed}_{condition}" for seed, condition in selected],
        "all_failed_episodes_in_original": sorted(failed_names),
        "original_failure_completions": failure_count,
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
                          root / "api_budget.jsonl", f"B_group{group_index}_{TAG}")

    live = LiveQwen(group["rows"], history, config["llm"], destination / "qwen",
                    root / "api_budget.jsonl", group_index)
    result_map = {}
    for seed, condition in selected:
        name = f"{seed}_{condition}"
        target = destination / "episodes" / f"{name}.json"
        if target.exists():
            result = json.loads(target.read_text(encoding="utf-8"))
            if result["seed"] != seed or result["condition"] != condition:
                raise RuntimeError("saved anomaly recovery episode differs")
        else:
            result = run_b_episode(group["rows"], seed, model, spec, config, condition,
                                   predict=live.predict, bob_class=PublicBobV5)
            save(target, result)
            emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}",
                                           "group": group_index, "seed": seed,
                                           "condition": condition,
                                           "score_300": result["score_first_300"],
                                           "queries": result["query_count"]})
        result_map[name] = {
            "selected_path": str(target),
            "selected_hash": sha256(target),
            "original_path": str(original / "episodes" / f"{name}.json"),
            "original_hash": sha256(original / "episodes" / f"{name}.json"),
        }
        print(json.dumps({"recovery": TAG, "group": group_index, "seed": seed,
                          "condition": condition, "score_300": result["score_first_300"],
                          "queries": result["query_count"]}), flush=True)

    selection = {
        "group": group_index,
        "recovery": TAG,
        "episodes": result_map,
    }
    save_exact(source / f"B_formal_{TAG}_selection.json", selection)
    return selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    config = load_config()
    selections = {}
    for group_index, names in EXPECTED.items():
        selections[str(group_index)] = recover_group(root, config, group_index, names)
    final = {
        "recovery": TAG,
        "episode_count": sum(len(names) for names in EXPECTED.values()),
        "groups": selections,
        "recovery_script_hash": sha256(Path(__file__)),
    }
    save_exact(root / f"B_formal_{TAG}_final_selection.json", final)
    print(json.dumps({"recovery": TAG, "episodes": final["episode_count"],
                      "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
