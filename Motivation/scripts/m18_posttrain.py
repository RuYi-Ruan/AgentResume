"""Serial M18 local source gates for completed approved Alice groups."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def stage_call(root, group, name, args):
    directory = root / f"group_{group}" / "local_stage_logs"
    directory.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while (directory / f"{name}.attempt{attempt}.log").exists():
        attempt += 1
    log = directory / f"{name}.attempt{attempt}.log"
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = "src"
    env["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run([sys.executable, *args], cwd=Path(__file__).resolve().parents[1],
                                env=env, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        print(json.dumps({"group": group, "stage": name, "exit": result.returncode,
                          "log": str(log)}), flush=True)
        raise RuntimeError(f"M18 local source stage failed: group {group} {name}")
    print(json.dumps({"group": group, "stage": name, "status": "complete",
                      "log": str(log)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--through", choices=("controlled", "D", "B"), default="B")
    args = parser.parse_args()
    if not args.run_id.replace("_", "").isalnum():
        raise ValueError("invalid run ID")
    root = Path("data/m18") / args.run_id
    groups = sorted(set(args.groups))
    if not groups or any(group not in range(9) for group in groups):
        raise ValueError("groups must be a nonempty subset of 0..8")
    for group in groups:
        result_path = root / f"group_{group}" / "result.json"
        if not result_path.exists() or not json.loads(result_path.read_text(encoding="utf-8"))["passed"]:
            raise RuntimeError(f"Alice growth gate not ready at requested group {group}")
    failures = []
    for group in groups:
        base = root / f"group_{group}"
        history = base / "history"
        if not (history / "impressions.json").exists():
            script = "scripts/m18_resume_history.py" if history.exists() else "scripts/m18_prepare_data.py"
            stage_call(root, group, "history", [script, "--run-id", args.run_id,
                                                "--group", str(group), *([] if "resume" in script else ["--stage", "history"])])
        if not json.loads((history / "impressions.json").read_text(encoding="utf-8"))["passed"]:
            failures.append((group, "history"))
            continue
        controlled = base / "controlled_A"
        if not (controlled / "manifest.json").exists():
            stage_call(root, group, "controlled_A", ["scripts/m18_prepare_controlled.py",
                                                      "--run-id", args.run_id, "--group", str(group)])
        if not json.loads((controlled / "manifest.json").read_text(encoding="utf-8"))["passed"]:
            failures.append((group, "controlled_A"))
    if failures:
        raise RuntimeError(f"M18 controlled source gates failed: {failures}; no formal Qwen")
    if args.through == "controlled":
        print(json.dumps({"local_M18_controlled_gates": True, "run_id": args.run_id,
                          "groups": groups}), flush=True)
        return
    for group in groups:
        base = root / f"group_{group}"
        d = base / "natural_D"
        if not (d / "manifest.json").exists():
            stage_call(root, group, "natural_D", ["scripts/m18_prepare_d.py", "--run-id", args.run_id,
                                                    "--group", str(group)])
        if not json.loads((d / "manifest.json").read_text(encoding="utf-8"))["passed"]:
            failures.append((group, "natural_D"))
    if failures:
        raise RuntimeError(f"M18 natural D quota failed: {failures}; no formal Qwen")
    if args.through == "D":
        print(json.dumps({"local_M18_D_gates": True, "run_id": args.run_id,
                          "groups": groups}), flush=True)
        return
    for group in groups:
        base = root / f"group_{group}"
        b = base / "B_development_gate"
        if not (b / "gate.json").exists():
            stage_call(root, group, "B_development_gate", ["scripts/m18_b_gate.py",
                                                            "--run-id", args.run_id, "--group", str(group)])
        if not json.loads((b / "gate.json").read_text(encoding="utf-8"))["passed"]:
            failures.append((group, "B_development_gate"))
    if failures:
        raise RuntimeError(f"M18 B public trigger gates failed: {failures}; no B formal Qwen")
    print(json.dumps({"all_local_M18_gates": True, "run_id": args.run_id,
                      "groups": groups}), flush=True)


if __name__ == "__main__":
    main()
