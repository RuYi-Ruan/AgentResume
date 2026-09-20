"""Approved M18 controlled A development/formal Qwen runner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
from ocres.m18_qwen import plan_requests, run_requests, summarize
from ocres.m18_training import group_spec, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--stage", choices=("development", "formal"), required=True)
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v2_mapfix"), default="v1")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    source = root / f"group_{args.group}" / "controlled_A"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["passed"] or manifest["group"] != args.group:
        raise RuntimeError("approved controlled A gate failed")
    config = load_config()
    if hashlib.sha256(Path("configs/m18_v1.json").read_bytes()).hexdigest() != manifest["config_hash"]:
        # config_hash uses canonical JSON, checked below using project helper
        from ocres.m18_training import config_hash
        if config_hash(config) != manifest["config_hash"]:
            raise RuntimeError("configuration differs from frozen controlled A")
    for file, key in (("src/ocres/m18_controlled.py", "enumerator_hash"),
                      ("src/ocres/m18_bob.py", "bob_protocol_hash")):
        if hashlib.sha256(Path(file).read_bytes()).hexdigest() != manifest[key]:
            raise RuntimeError("controlled A source code differs from frozen protocol")
    if hashlib.sha256((source.parent / "history" / "impressions.json").read_bytes()).hexdigest() != manifest["history_hash"]:
        raise RuntimeError("visible history changed")
    public_file = source / f"{args.stage}_public.json"
    scoring_file = source / f"{args.stage}_scoring.json"
    public = json.loads(public_file.read_text(encoding="utf-8"))
    truth = json.loads(scoring_file.read_text(encoding="utf-8"))
    if [r["event_hash"] for r in public] != [r["event_hash"] for r in truth]:
        raise RuntimeError("public/scoring split mismatch")
    history = json.loads((source.parent / "history" / "impressions.json").read_text(encoding="utf-8"))
    options = config["llm"]
    suffix = "" if args.prompt_version == "v1" else "_"+args.prompt_version
    planned = plan_requests(public, history, group_spec(config, args.group)["rows"],
                            options, "A_"+args.stage+suffix, args.group,
                            args.prompt_version)
    destination = source / "qwen" / (args.stage+suffix)
    destination.mkdir(parents=True, exist_ok=True)
    prompt_path = destination / "prompts.json"
    frozen = [{key: value for key, value in item.items() if key != "vocabulary"} for item in planned]
    if prompt_path.exists():
        if json.loads(prompt_path.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("Qwen prompts differ on resume")
    else:
        prompt_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_manifest = {"prompt_version": args.prompt_version,
                           "prompt_source_hash": hashlib.sha256(Path("src/ocres/m18_prompt.py").read_bytes()).hexdigest(),
                           "request_source_hash": hashlib.sha256(Path("src/ocres/m18_qwen.py").read_bytes()).hexdigest(),
                           "public_split_hash": hashlib.sha256(public_file.read_bytes()).hexdigest(),
                           "history_hash": manifest["history_hash"], "model": options["model"],
                           "request_count": len(planned)}
    record = destination / "prediction_manifest.json"
    if record.exists():
        if json.loads(record.read_text(encoding="utf-8")) != prediction_manifest:
            raise RuntimeError("Qwen protocol differs on resume")
    else:
        record.write_text(json.dumps(prediction_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = destination / "responses.jsonl"
    run_requests(planned, options, predictions, root / "api_budget.jsonl")
    summary = summarize(planned, predictions, truth)
    (destination / "scored.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    by_condition = {condition: {"events": len([x for x in summary if x["condition"] == condition]),
                                "intent_correct": sum(x["intent_correct"] for x in summary if x["condition"] == condition)}
                    for condition in options["conditions"]}
    print(json.dumps({"stage": args.stage, "group": args.group, "scores": by_condition}), flush=True)


if __name__ == "__main__":
    main()
