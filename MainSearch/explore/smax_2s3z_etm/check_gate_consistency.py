"""Consistency checks requested for the one-hot gate archival.

Check 1: the one-hot arm's initial observer parameters are bitwise identical to the placeholder arm's
         `checkpoint_00000000.bin` (same --seed => same init; only the z channel differs).
Check 2: in the new tri-arm evaluation, the profile and placeholder arms reproduce the previous
         two-arm evaluation item by item (per-arm primary metrics and the per-pair return diffs).

Usage:
    python check_gate_consistency.py \
        --onehot-init results/oracle_gate/og_onehot_1250x64x64/checkpoint_00000000.bin \
        --placeholder-init results/oracle_gate/og_placeholder_1250x64x64/checkpoint_00000000.bin \
        --old-eval results/oracle_gate/observer_gate_eval_full.json \
        --new-eval results/oracle_gate/observer_gate_eval_onehot_full.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np
from flax import serialization


def load_payload(path: Path) -> dict:
    return serialization.msgpack_restore(path.read_bytes())


def flatten(tree) -> dict[str, np.ndarray]:
    leaves = jax.tree_util.tree_leaves_with_path(tree)
    return {"/".join(str(k) for k in path): np.asarray(v) for path, v in leaves}


def compare_trees(name: str, a, b, out: list[str]) -> None:
    fa, fb = flatten(a), flatten(b)
    if sorted(fa) != sorted(fb):
        out.append(f"  {name}: FAIL - different structure "
                   f"({len(fa)} vs {len(fb)} leaves)")
        return
    worst, worst_key = 0.0, None
    mismatched = 0
    for key in fa:
        x, y = fa[key], fb[key]
        if x.shape != y.shape:
            mismatched += 1
            continue
        if x.dtype == bool or not np.issubdtype(x.dtype, np.number):
            diff = 0.0 if np.array_equal(x, y) else float("inf")
        else:
            diff = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
        if diff > 0.0:
            mismatched += 1
            if diff > worst:
                worst, worst_key = diff, key
    status = "PASS" if mismatched == 0 else "FAIL"
    out.append(f"  {name}: {status} - {len(fa)} leaves, mismatched={mismatched}, "
               f"max|diff|={worst:.3e}" + (f" (worst leaf: {worst_key})" if worst_key else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onehot-init", required=True)
    ap.add_argument("--placeholder-init", required=True)
    ap.add_argument("--old-eval", required=True)
    ap.add_argument("--new-eval", required=True)
    args = ap.parse_args()

    lines: list[str] = []
    ok = True

    # ---- check 1: identical initial observer parameters --------------------------------
    lines.append("check 1: one-hot initial observer state == placeholder initial observer state")
    a = load_payload(Path(args.onehot_init))
    b = load_payload(Path(args.placeholder_init))
    for field in ("params", "opt_states", "hidden"):
        if field in a and field in b:
            compare_trees(field, a[field], b[field], lines)
    for field in ("update_count", "ep_index"):
        if field in a and field in b:
            same = np.array_equal(np.asarray(a[field]), np.asarray(b[field]))
            lines.append(f"  {field}: {'PASS' if same else 'FAIL'} - "
                         f"{np.asarray(a[field]).tolist()} vs {np.asarray(b[field]).tolist()}")
            ok &= bool(same)
    # the metadata deliberately differs (different --arm/--run-name), so it is not compared
    lines.append("  (_meta is not compared: it embeds each run's own argv, which differs by --arm)")

    # ---- check 2: the two old arms reproduce item by item ------------------------------
    lines.append("")
    lines.append("check 2: profile / placeholder reproduce the previous two-arm evaluation")
    old = json.loads(Path(args.old_eval).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new_eval).read_text(encoding="utf-8"))

    for arm in ("profile", "placeholder"):
        o = old.get("results", {}).get(arm, {})
        n = new.get("results", {}).get(arm, {})
        for key in sorted(set(o) | set(n)):
            if key not in o or key not in n:
                continue
            same = o[key] == n[key]
            if not same:
                ok = False
            lines.append(f"  results.{arm}.{key}: {'PASS' if same else 'FAIL'} "
                         f"({o[key]!r} vs {n[key]!r})")

    old_pairs = old.get("paired", {}).get("per_pair_return_diff")
    new_comps = new.get("comparisons") or {}
    match = next((k for k in sorted(new_comps)
                  if "profile" in k and "placeholder" in k and "onehot" not in k), None)
    new_pp = (new_comps.get(match) or {}).get("per_pair_return_diff") if match else None
    if old_pairs is None or new_pp is None:
        keys = sorted(new_comps)
        lines.append(f"  per-pair diff comparison: SKIPPED - old={old_pairs is not None}, "
                     f"matched new key={match}, new comparison keys={keys}")
    else:
        same = np.array_equal(np.asarray(old_pairs), np.asarray(new_pp))
        worst = float(np.max(np.abs(np.asarray(old_pairs) - np.asarray(new_pp))))
        ok &= bool(same)
        lines.append(f"  per-pair return diffs (profile - placeholder), {len(old_pairs)} pairs: "
                     f"{'PASS' if same else 'FAIL'} - max|diff|={worst:.3e}")

    print("\n".join(lines))
    print(f"\nsummary: {'all checks passed' if ok else 'CHECKS FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
