"""M18 D visible natural event Qwen runner, development or formal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
from ocres.m18_qwen import plan_requests, run_requests, summarize
from ocres.m18_training import config_hash, group_spec, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--stage", choices=("development", "formal"), required=True)
    parser.add_argument("--prompt-version", choices=("v2", "v2_mapfix"), default="v2_mapfix")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    config = load_config()
    root = Path("data/m18") / args.run_id
    group = root / f"group_{args.group}"
    source = group / ("development_D" if args.stage == "development" else "natural_D")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if args.stage == "formal" and manifest["config_hash"] != config_hash(config):
        raise RuntimeError("formal natural D training config differs")
    if args.stage == "formal":
        revision = Path("configs/m18_d_v2.json")
        acceptance_config = Path("configs/m18_d_v3.json")
        acceptance_path = source / "d_v3_acceptance.json"
        run_acceptance_path = root / "D_v3_acceptance.json"
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        run_acceptance = json.loads(run_acceptance_path.read_text(encoding="utf-8"))
        if not acceptance["accepted"] or acceptance["actual_events"] != manifest["selected"]:
            raise RuntimeError("approved D-v3 group acceptance failed")
        if (not run_acceptance["passed"]
                or run_acceptance["config_hash"] != hashlib.sha256(acceptance_config.read_bytes()).hexdigest()
                or run_acceptance["total_events"] != 392):
            raise RuntimeError("approved D-v3 run acceptance failed")
        if hashlib.sha256(revision.read_bytes()).hexdigest() != manifest["D_revision_hash"]:
            raise RuntimeError("approved D sampling addendum differs")
        if hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() != acceptance["v2_manifest_hash"]:
            raise RuntimeError("D-v2 manifest changed after D-v3 acceptance")
        for path, expected in manifest["source_hashes"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
                raise RuntimeError("frozen D source differs")
    public_file = source / f"{args.stage}_public.json"
    scoring_file = source / f"{args.stage}_scoring.json"
    public = json.loads(public_file.read_text(encoding="utf-8"))
    truth = json.loads(scoring_file.read_text(encoding="utf-8"))
    if [r["event_hash"] for r in public] != [r["event_hash"] for r in truth]:
        raise RuntimeError("public/scoring mismatch")
    history_file = group / "history/impressions.json"
    history = json.loads(history_file.read_text(encoding="utf-8"))
    if not history["passed"]:
        raise RuntimeError("history gate failed")
    options = config["llm"]
    plan = plan_requests(public, history, group_spec(config, args.group)["rows"],
                         options, "D_"+args.stage+"_"+args.prompt_version,
                         args.group, args.prompt_version)
    output = source / "qwen" / (args.stage+"_"+args.prompt_version)
    output.mkdir(parents=True, exist_ok=True)
    frozen = [{k: v for k, v in item.items() if k != "vocabulary"} for item in plan]
    prompt_file = output / "prompts.json"
    if prompt_file.exists():
        if json.loads(prompt_file.read_text(encoding="utf-8")) != frozen:
            raise RuntimeError("D prompts differ on resume")
    else:
        prompt_file.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction_manifest = {"stage": args.stage, "prompt_version": args.prompt_version,
                           "prompt_source_hash": hashlib.sha256(Path("src/ocres/m18_prompt.py").read_bytes()).hexdigest(),
                           "request_source_hash": hashlib.sha256(Path("src/ocres/m18_qwen.py").read_bytes()).hexdigest(),
                           "public_split_hash": hashlib.sha256(public_file.read_bytes()).hexdigest(),
                           "D_v3_acceptance_hash": (hashlib.sha256((source / "d_v3_acceptance.json").read_bytes()).hexdigest()
                                                    if args.stage == "formal" else None),
                           "history_hash": hashlib.sha256(history_file.read_bytes()).hexdigest(),
                           "model": options["model"], "request_count": len(plan)}
    record = output / "prediction_manifest.json"
    if record.exists():
        if json.loads(record.read_text(encoding="utf-8")) != prediction_manifest:
            raise RuntimeError("D Qwen protocol changed")
    else:
        record.write_text(json.dumps(prediction_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    results = output / "responses.jsonl"
    run_requests(plan, options, results, root / "api_budget.jsonl")
    scored = summarize(plan, results, truth)
    (output / "scored.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"stage": args.stage, "group": args.group,
                      "scores": {c: {"N": sum(r["condition"] == c for r in scored),
                                     "correct": sum(r["intent_correct"] for r in scored if r["condition"] == c),
                                     "unknown": sum(r["prediction"]["intent"] == "unknown" for r in scored if r["condition"] == c)}
                                 for c in options["conditions"]}}), flush=True)


if __name__ == "__main__":
    main()
