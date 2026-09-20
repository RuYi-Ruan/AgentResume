"""Recover the final uncovered B episode containing one HTTP 500 response."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
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
from ocres.m18_training import config_hash, group_spec, load_config
from ocres.trainable import load_policy_checkpoint


GROUP, SEED, CONDITION = 4, 506009, "stale"
TAG = "anomaly_recovery2"
NEW_LIMIT = 29000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path("data/m18") / args.run_id
    source = root / f"group_{GROUP}"
    original = source / "B_formal"
    responses = original / "qwen/responses.jsonl"
    target_prefix = f"B:g{GROUP}:s{SEED}:c{CONDITION}:"
    failures = [row for row in read_jsonl(responses)
                if row.get("kind") == "completion"
                and row.get("id", "").startswith(target_prefix)
                and any(str(e).startswith("http_") for e in row.get("errors", []))]
    if len(failures) != 1 or failures[0].get("errors") != ["http_500"]:
        raise RuntimeError("episode no longer matches the audited single HTTP 500 failure")

    config = load_config()
    options = {**config["llm"], "request_limit": NEW_LIMIT}
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
    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "final uncovered episode containing one HTTP 500 response",
        "selected_episodes": [f"{SEED}_{CONDITION}"],
        "original_manifest_hash": sha256(original_manifest_path),
        "original_responses_hash": sha256(responses),
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
                          root / "api_budget.jsonl", "B_group4_http500_recovery")
    live = LiveQwen(group["rows"], history, options, destination / "qwen",
                    root / "api_budget.jsonl", GROUP)
    target = destination / "episodes" / f"{SEED}_{CONDITION}.json"
    if target.exists():
        result = json.loads(target.read_text(encoding="utf-8"))
    else:
        result = run_b_episode(group["rows"], SEED, model, spec, config, CONDITION,
                               predict=live.predict, bob_class=PublicBobV5)
        save(target, result)
        emit(root / "progress.jsonl", {"stage": f"B_formal_{TAG}", "group": GROUP,
                                       "seed": SEED, "condition": CONDITION,
                                       "score_300": result["score_first_300"],
                                       "queries": result["query_count"]})
    selection = {
        "group": GROUP, "recovery": TAG,
        "episodes": {f"{SEED}_{CONDITION}": {
            "selected_path": str(target), "selected_hash": sha256(target),
            "original_path": str(original / "episodes" / f"{SEED}_{CONDITION}.json"),
            "original_hash": sha256(original / "episodes" / f"{SEED}_{CONDITION}.json"),
        }},
    }
    save_exact(source / f"B_formal_{TAG}_selection.json", selection)
    print(json.dumps({"recovery": TAG, "group": GROUP, "seed": SEED,
                      "condition": CONDITION, "score_300": result["score_first_300"],
                      "queries": result["query_count"], "status": "complete"}), flush=True)


if __name__ == "__main__":
    main()
