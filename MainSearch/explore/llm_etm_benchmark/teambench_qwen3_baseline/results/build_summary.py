#!/usr/bin/env python3
"""Build ``results/summary.csv`` for the Phase C calibration (9 runs).

Why this script exists
----------------------
The official ablation JSON (``harness/ablation.py``) records, per run, only
``condition / task_id / seed / run_id / run_dir / pass / partial_score /
elapsed_sec / failure_modes / error`` -- there is **no token field** in this
commit. The only per-call token record produced while running is the adapter
trace (``TEAMBENCH_TRACE_LOG``): the adapter appends one ``response`` event per
``generate_with_tools`` call carrying ``call_input_tokens`` / ``call_output_tokens``
(per-call deltas of the adapter's own cumulative usage counters).

``AgentLoop`` issues exactly one ``generate_with_tools`` call per turn, so the
trace's ``response`` records can be sliced by each run's turn count and summed
per run. That is what this script does; the count/multiset of tool calls in each
slice is cross-checked against the run's own per-turn logs, so a mapping error
cannot pass silently.

Usage (WSL python3):
    python3 build_summary.py \
      --results results/ablation_results.json \
      --trace   results/phase_c_trace.jsonl \
      --log     results/phase_c_run.log \
      --out     results/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter


def turn_logs(run_dir: str) -> list[pathlib.Path]:
    """All per-turn logs of one run (logs/<role>[/attempt_N]/turn_*.json)."""
    return sorted(pathlib.Path(run_dir).rglob("turn_*.json"))


def call_signature(name: str, args) -> str:
    return name + "|" + json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def calls_of_turn_log(path: pathlib.Path) -> list[str]:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return [call_signature(str(c.get("name")), c.get("args") or {})
            for c in (d.get("tool_calls") or [])]


def parse_log_runs(log_text: str) -> list[dict]:
    """Per-run lines printed by the harness: '[i/9] cond x task (seed=0)' + 'PASS (..)'.

    Returned in printed order; used only as a cross-check of turns / elapsed / partial.
    """
    out: list[dict] = []
    cur = None
    head = re.compile(r"^\[(\d+)/(\d+)\]\s+(\S+)\s+x\s+(\S+)\s+\(seed=(\d+)\)")
    res = re.compile(r"^\s+(PASS|FAIL)\s+\(partial=([\d.]+),\s*([\d.]+)s,\s*(\d+) turns\)")
    for line in log_text.splitlines():
        m = head.match(line)
        if m:
            cur = {"index": int(m.group(1)), "condition": m.group(3), "task": m.group(4),
                   "seed": int(m.group(5))}
            out.append(cur)
            continue
        m = res.match(line)
        if m and cur is not None:
            cur.update({"status": m.group(1), "partial": float(m.group(2)),
                        "elapsed_sec": float(m.group(3)), "turns": int(m.group(4))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    report = json.loads(pathlib.Path(args.results).read_text(encoding="utf-8"))
    runs = report["runs"]

    responses: list[dict] = []
    requests = 0
    for line in pathlib.Path(args.trace).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ev = rec.get("event")
        if ev == "response":
            responses.append(rec)
        elif ev == "request":
            requests += 1
    if requests != len(responses):
        print(f"NOTE: {requests} request record(s) vs {len(responses)} response record(s) "
              f"(in-flight call when the process stopped, or a retry)", file=sys.stderr)

    log_runs = parse_log_runs(pathlib.Path(args.log).read_text(encoding="utf-8")) \
        if args.log and pathlib.Path(args.log).is_file() else []

    problems: list[str] = []
    cursor = 0
    rows = []
    for i, run in enumerate(runs):
        logs = turn_logs(run["run_dir"])
        n = len(logs)
        if n == 0:
            problems.append(f"run {i} {run['condition']}/{run['task_id']}: no turn log found "
                            f"under {run['run_dir']} (moved/deleted run tree?)")
        slice_ = responses[cursor:cursor + n]
        cursor += n
        in_tok = sum(int(r.get("call_input_tokens") or 0) for r in slice_)
        out_tok = sum(int(r.get("call_output_tokens") or 0) for r in slice_)

        # --- cross-check: the slice must contain exactly this run's recorded calls ---
        want = Counter(sig for p in logs for sig in calls_of_turn_log(p))
        got = Counter(call_signature(str(c.get("name")), c.get("args") or {})
                      for r in slice_ for c in (r.get("tool_calls") or []))
        if want != got:
            problems.append(
                f"run {i} {run['condition']}/{run['task_id']}: trace slice tool-call multiset "
                f"!= per-turn logs (missing={sum((want - got).values())}, "
                f"extra={sum((got - want).values())})")
        roles = Counter(r.get("role_hint") or "unknown" for r in slice_)
        row = {
            "task": run["task_id"],
            "condition": run["condition"],
            "seed": run["seed"],
            "partial_score": run["partial_score"],
            "passed": int(bool(run["pass"])),
            "turns": n,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "elapsed_sec": run["elapsed_sec"],
            "run_id": run["run_id"],
            "failure_modes": ";".join(run.get("failure_modes") or []),
            "error": run.get("error") or "",
        }
        rows.append(row)

        # harness log line, if present for this run
        if i < len(log_runs):
            lr = log_runs[i]
            if (lr["condition"], lr["task"]) != (run["condition"], run["task_id"]):
                problems.append(f"run {i}: log line {lr['condition']}/{lr['task']} does not "
                                f"match JSON {run['condition']}/{run['task_id']}")
            elif lr.get("turns") != n:
                problems.append(f"run {i}: log says {lr.get('turns')} turns, {n} turn logs found")
        print(f"[{i+1}/9] {run['condition']:<10} {run['task_id']:<22} partial="
              f"{run['partial_score']:.2f} pass={int(bool(run['pass']))} turns={n} "
              f"tok={in_tok}/{out_tok} {run['elapsed_sec']}s roles={dict(roles)}")

    if cursor != len(responses):
        problems.append(f"trace has {len(responses) - cursor} trailing response record(s) "
                        f"not attributed to any run")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} rows)")

    if problems:
        print("\nCONSISTENCY PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        print("consistency: FAILED", file=sys.stderr)
    else:
        print("\nconsistency: OK (trace slices match every run's per-turn logs)")

    # --- dev-gate numbers (plan 4.3) ---
    by_key = {(r["condition"], r["task"]): r for r in rows}
    tasks = sorted({r["task"] for r in rows})
    need = {"oracle", "restricted", "full"}
    if not all((c, t) in by_key for c in need for t in tasks):
        print("\ngate summary skipped: the result set does not cover every "
              "(condition, task) cell")
        return 1 if problems else 0
    print("\nper-task full - restricted:")
    for t in tasks:
        f = by_key[("full", t)]["partial_score"]
        r = by_key[("restricted", t)]["partial_score"]
        print(f"  {t:<24} full={f:.2f} restricted={r:.2f} diff={f - r:+.2f}")
    diffs = [by_key[("full", t)]["partial_score"] - by_key[("restricted", t)]["partial_score"]
             for t in tasks]
    print(f"  mean diff = {sum(diffs) / len(diffs):+.3f}   positive = "
          f"{sum(1 for d in diffs if d > 0)}/{len(diffs)}")
    print("\nper-condition means:")
    for c in ("oracle", "restricted", "full"):
        rs = [r for r in rows if r["condition"] == c]
        print(f"  {c:<10} partial={sum(r['partial_score'] for r in rs)/len(rs):.3f} "
              f"passes={sum(r['passed'] for r in rs)}/{len(rs)} "
              f"tokens={sum(r['input_tokens'] for r in rs)}/"
              f"{sum(r['output_tokens'] for r in rs)} "
              f"turns={sum(r['turns'] for r in rs)} "
              f"elapsed={sum(r['elapsed_sec'] for r in rs):.1f}s")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
