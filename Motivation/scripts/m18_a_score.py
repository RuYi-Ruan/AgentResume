"""Score frozen controlled A continuations with one offline Qwen prediction."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, "src")
import torch

from ocres.m18_score import continue_controlled
from ocres.m18_bob import PublicBob
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_training import group_spec, load_config
from ocres.trainable import load_policy_checkpoint
from m18_train import save, emit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--group", type=int, required=True)
    parser.add_argument("--stage", choices=("development", "formal"), required=True)
    parser.add_argument("--prompt-version", choices=("v2", "v2_mapfix"), default="v2_mapfix")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, args.group)
    root = Path("data/m18") / args.run_id
    source = root / f"group_{args.group}" / "controlled_A"
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not manifest["passed"]:
        raise RuntimeError("controlled A gate failed")
    if hashlib.sha256(Path("src/ocres/m18_bob.py").read_bytes()).hexdigest() != manifest["bob_protocol_hash"]:
        raise RuntimeError("Bob policy differs from controlled A frozen protocol")
    rows = json.loads((source / f"{args.stage}_scoring.json").read_text(encoding="utf-8"))
    qwen_scores = source / "qwen" / f"{args.stage}_{args.prompt_version}" / "scored.json"
    predictions = json.loads(qwen_scores.read_text(encoding="utf-8"))
    lookup = {(p["event_index"], p["condition"]): p["prediction"] for p in predictions}
    model, spec, _ = load_policy_checkpoint(Path("artifacts/m18") / args.run_id /
                                            "checkpoints" / f"group_{args.group}" / "post.pt")
    model.eval()
    destination = source / "A_score" / f"{args.stage}_{args.prompt_version}"
    destination.mkdir(parents=True, exist_ok=True)
    bob_class = PublicBobV5 if args.stage == "formal" else PublicBob
    controller_path = Path("src/ocres/m18_bob_v5.py") if args.stage == "formal" else Path("src/ocres/m18_bob.py")
    if args.stage == "formal":
        gate = json.loads((root / "B_v5_acceptance.json").read_text(encoding="utf-8"))
        if not gate["passed"] or gate["controller"] != PublicBobV5.version:
            raise RuntimeError("formal A-score requires frozen B-v5 controller")
    score_manifest = {"group": args.group, "stage": args.stage,
                      "controller": getattr(bob_class, "version", "m18-public-bob-v1"),
                      "controller_hash": hashlib.sha256(controller_path.read_bytes()).hexdigest(),
                      "score_source_hash": hashlib.sha256(Path("src/ocres/m18_score.py").read_bytes()).hexdigest(),
                      "scoring_split_hash": hashlib.sha256((source / f"{args.stage}_scoring.json").read_bytes()).hexdigest(),
                      "qwen_scores_hash": hashlib.sha256(qwen_scores.read_bytes()).hexdigest(),
                      "events": len(rows), "conditions": config["llm"]["conditions"]+["oracle"]}
    record = destination / "manifest.json"
    if record.exists():
        if json.loads(record.read_text(encoding="utf-8")) != score_manifest:
            raise RuntimeError("A-score protocol differs on resume")
    else:
        save(record, score_manifest)
    summary = []
    for index, row in enumerate(rows):
        truth = row["scoring_only"]["post"]
        for condition in config["llm"]["conditions"] + ["oracle"]:
            prediction = truth if condition == "oracle" else lookup[(index, condition)]
            target = destination / "events" / f"{index:03}_{condition}.json"
            if target.exists():
                item = json.loads(target.read_text(encoding="utf-8"))
                if item["event_hash"] != row["event_hash"] or item["condition"] != condition or item["prediction"] != prediction:
                    raise RuntimeError("saved A-score event differs")
            else:
                outcome = continue_controlled(row, group["rows"], model, spec, prediction,
                                              config["evaluation"]["continuation_ticks"], bob_class=bob_class)
                item = {"event_index": index, "event_hash": row["event_hash"],
                        "condition": condition, "prediction": prediction, **outcome}
                save(target, item)
                emit(root / "progress.jsonl", {"stage": "A_score_"+args.stage, "group": args.group,
                                               "event_index": index, "condition": condition,
                                               "soups": outcome["soups"]})
            summary.append(item)
    scores_file = destination / "scores.json"
    if scores_file.exists():
        if json.loads(scores_file.read_text(encoding="utf-8")) != summary:
            raise RuntimeError("aggregated A-score differs on resume")
    else:
        save(scores_file, summary)
    print(json.dumps({"group": args.group, "stage": args.stage,
                      "mean_reward": {condition: sum(r["reward"] for r in summary if r["condition"] == condition)/len(rows)
                                      for condition in config["llm"]["conditions"]+["oracle"]}}), flush=True)


if __name__ == "__main__":
    main()
