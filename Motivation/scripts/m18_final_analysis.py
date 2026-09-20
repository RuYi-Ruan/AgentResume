"""M18 final aggregate statistics across 9 groups (3 Alice training seeds x 3 maps).

Reads the frozen run directory and produces:
  A) controlled-A intent/facility accuracy (450 events/condition) + paired tests
  B) natural-D scoring from saved Qwen completions (majority vote) + paired tests
  C) B-formal cooperation scores (soups / points) + paired sign tests
Writes JSON to <run>/final_analysis.json and prints markdown tables.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

try:
    from scipy.stats import binomtest
except Exception:  # pragma: no cover
    binomtest = None


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - r) / d, (c + r) / d


def mcnemar_exact(b, c):
    """b = only first correct, c = only second correct."""
    n = b + c
    if n == 0:
        return 1.0
    if binomtest is not None:
        return binomtest(min(b, c), n, 0.5).pvalue
    return 1.0


def sign_test(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    if binomtest is not None:
        return binomtest(min(better, worse), n, 0.5).pvalue
    return 1.0


def boot_diff(pairs, iters=4000, seed=7):
    """pairs: list of (a,b) per unit; bootstrap mean(b)-mean(a) with unit resample."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return None
    diffs = []
    for _ in range(iters):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(y - x for x, y in s) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def load_group_meta(run, g):
    res = json.loads((run / f"group_{g}" / "result.json").read_text(encoding="utf-8"))
    return {"group": g, "map": res["group"]["map"], "training_seed": res["group"]["training_seed"]}


# ---------------------------------------------------------------- A
def analyze_A(run, groups, out):
    events = []  # (group, event_index, condition, correct, facility_correct, unknown)
    per_group = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for g in groups:
        p = run / f"group_{g}" / "controlled_A" / "qwen" / "formal_v2_mapfix" / "scored.json"
        rows = json.loads(p.read_text(encoding="utf-8"))
        for r in rows:
            cond = r["condition"]
            ok = bool(r["intent_correct"])
            fac = bool(r["facility_correct"])
            unk = r["prediction"]["intent"] == "unknown"
            events.append({"group": g, "event": r["event_index"], "cond": cond,
                           "ok": ok, "fac": fac, "unknown": unk})
            per_group[cond]["intent"][0] += ok
            per_group[cond]["intent"][1] += 1
            per_group[cond]["facility"][0] += fac
    summ = {}
    meta = {g: load_group_meta(run, g) for g in groups}
    for cond in ("stale", "updated", "none"):
        k = sum(e["ok"] for e in events if e["cond"] == cond)
        n = sum(1 for e in events if e["cond"] == cond)
        kf = sum(e["fac"] for e in events if e["cond"] == cond)
        ku = sum(1 for e in events if e["cond"] == cond and e["unknown"])
        p, lo, hi = wilson(k, n)
        pf, lof, hif = wilson(kf, n)
        summ[cond] = {"n": n, "intent_acc": p, "intent_ci": [lo, hi],
                      "facility_acc": kf / n if n else None, "facility_ci": [lof, hif],
                      "unknown_rate": ku / n if n else None}
    # paired tests by (group,event)
    idx = collections.defaultdict(dict)
    for e in events:
        idx[(e["group"], e["event"])][e["cond"]] = e
    pairs = {("updated", "stale"): [], ("updated", "none"): [], ("none", "stale"): []}
    for key, d in idx.items():
        for (c1, c2) in pairs:
            if c1 in d and c2 in d:
                pairs[(c1, c2)].append((int(d[c2]["ok"]), int(d[c1]["ok"])))
    tests = {}
    for (c1, c2), ps in pairs.items():
        b = sum(1 for x, y in ps if x == 1 and y == 0)  # only c2 correct
        c = sum(1 for x, y in ps if x == 0 and y == 1)  # only c1 correct
        ci = boot_diff(ps)
        tests[f"{c1}_minus_{c2}"] = {
            "n_pairs": len(ps), "b_only_second": b, "c_only_first": c,
            "diff": sum(y - x for x, y in ps) / len(ps) if ps else None,
            "boot95": ci, "mcnemar_p": mcnemar_exact(b, c),
        }
    # per-map
    per_map = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for e in events:
        m = meta[e["group"]]["map"]
        per_map[m][(e["cond"], "ok")][0] += e["ok"]
        per_map[m][(e["cond"], "ok")][1] += 1
    out["A"] = {"summary": summ, "paired": tests,
                "per_map": {m: {f"{c}_acc": (v[0] / v[1]) for (c, _), v in d.items()} for m, d in per_map.items()},
                "per_group": {c: {"intent": v["intent"], "facility": v["facility"]} for c, v in per_group.items()}}
    return out


# ---------------------------------------------------------------- D
def majority_vote(vals):
    cnt = collections.Counter(v for v in vals if v not in (None, "unknown"))
    if not cnt:
        return "unknown"
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unknown"
    return top[0][0]


def analyze_D(run, groups, out):
    events = []
    per_group_events = {}
    for g in groups:
        base = run / f"group_{g}" / "natural_D"
        qdir = base / "qwen" / "formal_v2_mapfix" / "responses.jsonl"
        truth = json.loads((base / "formal_scoring.json").read_text(encoding="utf-8"))
        n_events = len(truth)
        per_group_events[g] = n_events
        by = collections.defaultdict(lambda: {"intent": [], "facility": []})
        for line in qdir.read_text(encoding="utf-8").splitlines():
            j = json.loads(line)
            if j.get("kind") != "completion":
                continue
            key = (j["event_index"], j["condition"])
            if j.get("valid"):
                by[key]["intent"].append(j["parsed"].get("intent"))
                by[key]["facility"].append(j["parsed"].get("target_facility"))
            else:
                by[key]["intent"].append("unknown")
                by[key]["facility"].append("unknown")
        for (ei, cond), votes in by.items():
            t = truth[ei]["scoring_only"]["post"]
            pi = majority_vote(votes["intent"])
            pf = majority_vote(votes["facility"])
            events.append({"group": g, "event": ei, "cond": cond,
                           "ok": pi == t["intent"], "fac": pf == t["target_facility"],
                           "unknown": pi == "unknown", "truth": t["intent"], "pred": pi})
    summ = {}
    for cond in ("stale", "updated", "none"):
        sub = [e for e in events if e["cond"] == cond]
        k = sum(e["ok"] for e in sub)
        n = len(sub)
        p, lo, hi = wilson(k, n)
        summ[cond] = {"n": n, "intent_acc": p, "intent_ci": [lo, hi],
                      "facility_acc": (sum(e["fac"] for e in sub) / n if n else None),
                      "unknown_rate": (sum(e["unknown"] for e in sub) / n if n else None)}
    idx = collections.defaultdict(dict)
    for e in events:
        idx[(e["group"], e["event"])][e["cond"]] = e
    tests = {}
    for (c1, c2) in [("updated", "stale"), ("updated", "none"), ("none", "stale")]:
        ps = [(int(d[c2]["ok"]), int(d[c1]["ok"])) for d in idx.values() if c1 in d and c2 in d]
        b = sum(1 for x, y in ps if x == 1 and y == 0)
        c = sum(1 for x, y in ps if x == 0 and y == 1)
        tests[f"{c1}_minus_{c2}"] = {"n_pairs": len(ps), "b": b, "c": c,
                                     "diff": (sum(y - x for x, y in ps) / len(ps)) if ps else None,
                                     "boot95": boot_diff(ps), "mcnemar_p": mcnemar_exact(b, c)}
    out["D"] = {"events_per_group": per_group_events, "summary": summ, "paired": tests,
                "total_events": sum(per_group_events.values())}
    return out


# ---------------------------------------------------------------- B
def pick_episode_paths(run, g):
    base = run / f"group_{g}" / "B_formal" / "episodes"
    chosen = {}
    for f in base.glob("*.json"):
        chosen[f.name] = f
    for sel in (run / f"group_{g}").glob("B_formal*_selection.json"):
        try:
            data = json.loads(sel.read_text(encoding="utf-8"))
        except Exception:
            continue
        eps = data.get("episodes") or {}
        for name, info in eps.items():
            p = Path(info["selected_path"])
            if not p.is_absolute():
                p = Path.cwd() / p
            chosen[name + ("" if name.endswith(".json") else ".json")] = p
    return chosen


def analyze_B(run, groups, out):
    rows = []
    qstats = collections.defaultdict(lambda: [0, 0])
    per_group = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in groups:
        for name, path in pick_episode_paths(run, g).items():
            j = json.loads(path.read_text(encoding="utf-8"))
            cond = j["condition"]
            rows.append({"group": g, "seed": j["seed"], "cond": cond,
                         "soups": j["soups"], "full": j["score_full_700"], "short": j["score_first_300"]})
            per_group[g][cond].append(j["soups"])
            for q in j.get("public_queries", []):
                if q.get("intent_correct") is not None:
                    qstats[cond][0] += int(bool(q["intent_correct"]))
                    qstats[cond][1] += 1
    meta = {g: load_group_meta(run, g) for g in groups}
    per_cond = {}
    for cond in ("stale", "updated", "none", "oracle"):
        sub = [r for r in rows if r["cond"] == cond]
        if not sub:
            continue
        per_cond[cond] = {"n": len(sub),
                          "soups_mean": sum(r["soups"] for r in sub) / len(sub),
                          "full_mean": sum(r["full"] for r in sub) / len(sub),
                          "short_mean": sum(r["short"] for r in sub) / len(sub)}
    keys = collections.defaultdict(dict)
    for r in rows:
        keys[(r["group"], r["seed"])][r["cond"]] = r
    paired = {}
    for c1, c2 in [("updated", "stale"), ("updated", "none"), ("none", "stale"), ("oracle", "none")]:
        ps = [(d[c2]["soups"], d[c1]["soups"]) for d in keys.values() if c1 in d and c2 in d]
        if not ps:
            continue
        diffs = [y - x for x, y in ps]
        better = sum(1 for d in diffs if d > 0)
        worse = sum(1 for d in diffs if d < 0)
        paired[f"{c1}_minus_{c2}"] = {
            "n_pairs": len(ps), "mean_diff_soups": sum(diffs) / len(diffs),
            "better": better, "same": sum(1 for d in diffs if d == 0), "worse": worse,
            "sign_p": sign_test(better, worse), "boot95": boot_diff(ps),
        }
    per_map = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        per_map[meta[r["group"]]["map"]][r["cond"]].append(r["soups"])
    out["B"] = {"per_condition": per_cond, "paired": paired,
                "query_intent": {c: (v[0] / v[1] if v[1] else None, v[1]) for c, v in qstats.items()},
                "per_map_soups": {m: {c: (sum(v) / len(v) if v else None) for c, v in d.items()} for m, d in per_map.items()},
                "n_episodes": len(rows)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v1_waitfix_20260915")
    ap.add_argument("--groups", default="0,1,2,3,4,5,6,7,8")
    args = ap.parse_args()
    groups = [int(x) for x in args.groups.split(",")]
    run = Path("data/m18") / args.run_id
    out = {}
    analyze_A(run, groups, out)
    analyze_D(run, groups, out)
    analyze_B(run, groups, out)
    (run / "final_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print("== A: controlled paired events ==")
    for c, v in out["A"]["summary"].items():
        print(f"  {c:8s} n={v['n']} intent={v['intent_acc']:.4f} CI[{v['intent_ci'][0]:.3f},{v['intent_ci'][1]:.3f}] "
              f"facility={v['facility_acc']:.4f} unknown={v['unknown_rate']:.3f}")
    print("  paired:", json.dumps(out["A"]["paired"], ensure_ascii=False))
    print("== D: natural events ==")
    for c, v in out["D"]["summary"].items():
        print(f"  {c:8s} n={v['n']} intent={v['intent_acc']:.4f} CI[{v['intent_ci'][0]:.3f},{v['intent_ci'][1]:.3f}] "
              f"facility={v['facility_acc']:.4f} unknown={v['unknown_rate']:.3f}")
    print("  paired:", json.dumps(out["D"]["paired"], ensure_ascii=False))
    print("== B: cooperation ==")
    for c, v in out["B"]["per_condition"].items():
        print(f"  {c:8s} n={v['n']} soups={v['soups_mean']:.3f} full={v['full_mean']:.2f} short={v['short_mean']:.2f}")
    print("  paired:", json.dumps(out["B"]["paired"], ensure_ascii=False))
    print("  query intent:", json.dumps(out["B"]["query_intent"], ensure_ascii=False))


if __name__ == "__main__":
    main()
