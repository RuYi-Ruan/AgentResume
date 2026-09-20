"""Auditable M18 recovery for the 2026-09-17 transport outage.

The original failed artifacts are never edited.  This script replays only the
accepted D group and B group into new recovery directories, while retaining
hash links to the failed originals.  It is resumable inside those new dirs.
"""
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
from ocres.m18_qwen import (append_jsonl, plan_requests, read_jsonl, request_once,
                            run_requests, summarize, trim_prompt, utc)
from ocres.m18_training import config_hash, episode_seeds, group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import emit, save


TAG = "transport_recovery1"
D_GROUP = 8
B_GROUP = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transport_failures(path: Path) -> list[dict]:
    return [
        row for row in read_jsonl(path)
        if row.get("kind") == "completion"
        and any(error in {"URLError", "TimeoutError", "ConnectionError", "transport_exhausted"}
                for error in row.get("errors", []))
    ]


def save_exact(path: Path, data) -> None:
    """Create once, or require an exact match when resuming."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != data:
            raise RuntimeError(f"recovery artifact differs on resume: {path}")
        return
    save(path, data)


def transport_healthcheck(messages: list[dict], options: dict, output: Path,
                          budget_file: Path, name: str) -> None:
    """Require one fresh HTTP response before a recovery stage starts.

    This call is logged but excluded from every score.  A failed check aborts
    immediately, preventing another whole batch of transport-only failures.
    """
    records = read_jsonl(output)
    if any(row.get("success") for row in records):
        return
    budget = len(read_jsonl(budget_file))
    if budget >= options["request_limit"]:
        raise RuntimeError("approved M18 request cap reached before recovery healthcheck")
    attempt = len(records)
    check_id = f"recovery_healthcheck:{name}:a{attempt}"
    append_jsonl(budget_file, {"id": check_id, "attempt": 0, "utc": utc(),
                               "scoring": False})
    try:
        response = request_once(messages, options)
    except Exception as exc:
        append_jsonl(output, {"id": check_id, "utc": utc(), "success": False,
                              "error": type(exc).__name__})
        raise RuntimeError(f"recovery transport healthcheck failed: {type(exc).__name__}") from exc
    append_jsonl(output, {"id": check_id, "utc": utc(), "success": True,
                          "response_id": response.get("id"), "model": response.get("model")})


def recover_d(root: Path, config: dict) -> None:
    source = root / f"group_{D_GROUP}" / "natural_D"
    original = source / "qwen" / "formal_v2_mapfix"
    original_responses = original / "responses.jsonl"
    failures = transport_failures(original_responses)
    completions = [row for row in read_jsonl(original_responses)
                   if row.get("kind") == "completion"]
    if len(completions) != 450 or len(failures) != 450:
        raise RuntimeError("D group 8 no longer matches the approved all-transport-failure audit")

    public = json.loads((source / "formal_public.json").read_text(encoding="utf-8"))
    truth = json.loads((source / "formal_scoring.json").read_text(encoding="utf-8"))
    history_file = source.parent / "history" / "impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    plan = plan_requests(public, history, group_spec(config, D_GROUP)["rows"],
                         config["llm"], "D_formal_v2_mapfix", D_GROUP, "v2_mapfix")
    frozen = [{key: value for key, value in item.items() if key != "vocabulary"}
              for item in plan]
    old_prompts = json.loads((original / "prompts.json").read_text(encoding="utf-8"))
    if frozen != old_prompts:
        raise RuntimeError("D recovery prompts differ from the failed formal prompts")

    destination = source / "qwen" / f"formal_v2_mapfix_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    save_exact(destination / "prompts.json", frozen)
    manifest = {
        "recovery": TAG,
        "reason": "all 450 original completions failed at transport",
        "group": D_GROUP,
        "request_count": len(plan),
        "model": config["llm"]["model"],
        "config_hash": config_hash(config),
        "original_manifest_hash": sha256(original / "prediction_manifest.json"),
        "original_prompts_hash": sha256(original / "prompts.json"),
        "original_responses_hash": sha256(original_responses),
        "original_transport_failures": len(failures),
        "recovery_script_hash": sha256(Path(__file__)),
    }
    save_exact(destination / "recovery_manifest.json", manifest)
    responses = destination / "responses.jsonl"
    transport_healthcheck(plan[0]["messages"], config["llm"],
                          destination / "transport_healthcheck.jsonl",
                          root / "api_budget.jsonl", "D_group8")
    run_requests(plan, config["llm"], responses, root / "api_budget.jsonl")
    scored = summarize(plan, responses, truth)
    save_exact(destination / "scored.json", scored)
    print(json.dumps({"recovery": "D", "group": D_GROUP,
                      "requests": len(plan), "status": "complete"}), flush=True)


def recover_b(root: Path, config: dict) -> None:
    source = root / f"group_{B_GROUP}"
    original = source / "B_formal"
    original_responses = original / "qwen" / "responses.jsonl"
    failures = transport_failures(original_responses)
    if len(failures) < 500:
        raise RuntimeError("B group 0 no longer matches the audited transport outage")

    original_manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in original_manifest["source_hashes"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError(f"B frozen source changed: {name}")

    group = group_spec(config, B_GROUP)
    history_file = source / "history" / "impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    artifacts = Path("artifacts/m18") / root.name
    model, spec, _ = load_policy_checkpoint(
        artifacts / "checkpoints" / f"group_{B_GROUP}" / "post.pt")
    model.eval()
    torch.set_num_threads(1)

    destination = source / f"B_formal_{TAG}"
    destination.mkdir(parents=True, exist_ok=True)
    seeds = list(episode_seeds(config, B_GROUP, "B"))
    manifest = {
        **original_manifest,
        "recovery": TAG,
        "reason": "original B group 0 contained transport-failed Qwen predictions",
        "seeds": seeds,
        "original_manifest_hash": sha256(original / "manifest.json"),
        "original_responses_hash": sha256(original_responses),
        "original_transport_failures": len(failures),
        "recovery_script_hash": sha256(Path(__file__)),
    }
    save_exact(destination / "manifest.json", manifest)

    probe_event = json.loads((source / "controlled_A" / "formal_public.json")
                             .read_text(encoding="utf-8"))[0]["public_event"]
    from ocres.m18_prompt import build_prompt
    probe_messages, _, _ = build_prompt(probe_event, history, group["rows"], "v2_mapfix")
    probe_messages, _ = trim_prompt(probe_messages, config["llm"]["max_input_tokens"])
    transport_healthcheck(probe_messages, config["llm"],
                          destination / "transport_healthcheck.jsonl",
                          root / "api_budget.jsonl", "B_group0")
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("D", "B", "all"), default="all")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    config = load_config()
    if args.stage in ("D", "all"):
        recover_d(root, config)
    if args.stage in ("B", "all"):
        recover_b(root, config)


if __name__ == "__main__":
    main()
