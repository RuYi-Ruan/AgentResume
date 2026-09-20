"""Serial frozen M18 formal Qwen A/D, controlled score, and natural B stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from m18_posttrain import stage_call


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--through", choices=("A", "D", "A_score", "B"), default="B")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    for group in range(9):
        base = root / f"group_{group}"
        checks = ((base / "result.json", "passed"),
                  (base / "history/impressions.json", "passed"),
                  (base / "controlled_A/manifest.json", "passed"),
                  (base / "natural_D/d_v3_acceptance.json", "accepted"))
        for path, field in checks:
            if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))[field]:
                raise RuntimeError(f"M18 formal gates not complete: group {group} {path}")
    b_gate = root / "B_v5_acceptance.json"
    if not b_gate.exists() or not json.loads(b_gate.read_text(encoding="utf-8"))["passed"]:
        raise RuntimeError("M18 formal B-v5 development gate not complete")
    for group in range(9):
        stage_call(root, group, "Qwen_A_formal_v2_mapfix", ["scripts/m18_qwen_a.py", "--run-id", args.run_id,
                                                      "--group", str(group), "--stage", "formal",
                                                      "--prompt-version", "v2_mapfix"])
    if args.through == "A":
        print(json.dumps({"M18_formal_A_complete": True, "run_id": args.run_id}), flush=True)
        return
    for group in range(9):
        stage_call(root, group, "Qwen_D_formal_v2_mapfix", ["scripts/m18_qwen_d.py", "--run-id", args.run_id,
                                                      "--group", str(group), "--stage", "formal",
                                                      "--prompt-version", "v2_mapfix"])
    if args.through == "D":
        print(json.dumps({"M18_formal_A_D_complete": True, "run_id": args.run_id}), flush=True)
        return
    for group in range(9):
        stage_call(root, group, "A_score_formal", ["scripts/m18_a_score.py", "--run-id", args.run_id,
                                                  "--group", str(group), "--stage", "formal",
                                                  "--prompt-version", "v2_mapfix"])
    if args.through == "A_score":
        print(json.dumps({"M18_formal_A_D_A_score_complete": True, "run_id": args.run_id}), flush=True)
        return
    for group in range(9):
        stage_call(root, group, "B_formal", ["scripts/m18_b_run.py", "--run-id", args.run_id,
                                            "--group", str(group)])
    print(json.dumps({"M18_formal_complete": True, "run_id": args.run_id}), flush=True)


if __name__ == "__main__":
    main()
