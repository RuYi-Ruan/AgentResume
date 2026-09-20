"""Aggregate M19 formal results across groups (M17 protocol on M18 checkpoints).

Reads data/m19/<run>/group_*/formal_test_<variant>_responses.json, majority-votes
the 3 repeats per (event, impression), then reports:
  1) post-truth scoring (truth = upgraded Alice's intent, always FETCH here);
  2) counterbalanced scoring: half the events scored against the pre-Alice
     truth (PRE) and half against the post-Alice truth (FETCH), which removes
     the "always answer FETCH" advantage of the no-impression baseline;
  3) 2x2 impression x partner-version table + cluster bootstrap (groups then events).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from scipy.stats import binomtest  # noqa: E402

RUN = "v1_waitfix_20260915"
DATA = pathlib.Path("data/m19") / RUN


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - r) / d, (c + r) / d


def majority(vals):
    cnt = collections.Counter(v for v in vals if v not in (None, "unknown"))
    if not cnt:
        return "unknown"
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unknown"
    return top[0][0]


def load_group(g, variant):
    p = DATA / f"group_{g}" / f"formal_test_{variant}_responses.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    truth = {}
    for row in data["rows"]:
        eid, imp = row["event_id"], row["impression"]
        truth[eid] = {
            "pre": row["scoring_only"]["pre"]["intent"],
            "post": row["scoring_only"]["post"]["intent"],
            "post_facility": row["scoring_only"]["post"]["target_facility"],
        }
        if "response" in row and row["response"]["parsed"] is not None:
            by[eid][imp].append(row["response"]["parsed"].get("intent"))
        else:
            by[eid][imp].append("unknown")
    events = {}
    for eid, imps in by.items():
        events[eid] = {imp: majority(v) for imp, v in imps.items()}
        events[eid]["truth"] = truth[eid]
    return events


def partner_version(eid):
    """Deterministic 50/50 assignment of the scored partner version."""
    return "post" if int(hashlib.sha1(eid.encode()).hexdigest(), 16) % 2 == 0 else "pre"


def rate(events, imp, truth_key):
    k = n = 0
    for eid, d in events.items():
        if imp not in d or "truth" not in d:
            continue
        n += 1
        k += int(d[imp] == d["truth"][truth_key])
    return k, n


def report(events, label, truth_key=None):
    """truth_key None -> counterbalanced assignment per event."""
    out = {}
    for imp in ("pre", "post", "none"):
        k = n = refusal = 0
        for eid, d in events.items():
            if imp not in d or "truth" not in d:
                continue
            n += 1
            t = d["truth"][truth_key] if truth_key else d["truth"][partner_version(eid)]
            if d[imp] == "unknown":
                refusal += 1
            k += int(d[imp] == t)
        if n:
            p, lo, hi = wilson(k, n)
            out[imp] = {"n": n, "acc": p, "ci": [lo, hi], "refusal": refusal / n}
    print(f"[{label}] " + " | ".join(
        f"{i}={out[i]['acc']:.3f}" for i in out))
    return out


def table_2x2(events):
    """accuracy by impression x partner version, plus match/mismatch."""
    cells = collections.defaultdict(lambda: [0, 0])
    for eid, d in events.items():
        v = partner_version(eid)
        for imp in ("pre", "post", "none"):
            if imp not in d:
                continue
            cells[(imp, v)][0] += int(d[imp] == d["truth"][v])
            cells[(imp, v)][1] += 1
    match = collections.defaultdict(lambda: [0, 0])
    for eid, d in events.items():
        v = partner_version(eid)
        for imp in ("pre", "post"):
            if imp not in d:
                continue
            key = "match" if imp == v else "mismatch"
            match[key][0] += int(d[imp] == d["truth"][v])
            match[key][1] += 1
    return (
        {f"{i}|partner={v}": (c[0] / c[1]) for (i, v), c in cells.items() if c[1]},
        {k: (c[0] / c[1], c[1]) for k, c in match.items()},
    )


def mcnemar(events, a, b, truth_key=None):
    bb = cc = 0
    for eid, d in events.items():
        if a not in d or b not in d:
            continue
        t = d["truth"][truth_key] if truth_key else d["truth"][partner_version(eid)]
        ga, gb = d[a] == t, d[b] == t
        if ga and not gb:
            bb += 1
        elif gb and not ga:
            cc += 1
    n = bb + cc
    return {"b_only_a": bb, "b_only_b": cc, "pair": n,
            "p": binomtest(min(bb, cc), n, 0.5).pvalue if n else 1.0}


def cluster_boot(group_events, a, b, truth_key=None, iters=4000, seed=11):
    rng = random.Random(seed)
    keys = list(group_events)
    diffs = []
    for _ in range(iters):
        pa = pb = na = nb = 0
        for _ in range(len(keys)):
            g = keys[rng.randrange(len(keys))]
            evs = list(group_events[g].values())
            for _ in range(len(evs)):
                d = evs[rng.randrange(len(evs))]
                if "truth" not in d:
                    continue
                t = d["truth"][truth_key] if truth_key else d["truth"][partner_version(d.get("_eid", ""))]
                if a in d:
                    na += 1
                    pa += int(d[a] == t)
                if b in d:
                    nb += 1
                    pb += int(d[b] == t)
        diffs.append((pb / nb if nb else 0) - (pa / na if na else 0))
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="1,2,3,4,6,7")
    ap.add_argument("--variant", default="v1")
    ap.add_argument("--out", default=str(DATA / "final_analysis.json"))
    args = ap.parse_args()
    groups = [int(x) for x in args.groups.split(",")]
    per_group, all_events = {}, {}
    for g in groups:
        ev = load_group(g, args.variant)
        if ev is None:
            print(f"group {g}: no results yet, skipped")
            continue
        per_group[g] = ev
        for eid, d in ev.items():
            key = f"g{g}:{eid}"
            d["_eid"] = key
            all_events[key] = d
        print(f"g{g}: post-truth " + " ".join(
            f"{i}={rate(ev, i, 'post')[0]}/{rate(ev, i, 'post')[1]}" for i in ("pre", "post", "none")))
    if not per_group:
        raise SystemExit("no groups finished")
    print("\n== post-truth scoring (partner = upgraded Alice) ==")
    post = report(all_events, "post-truth", truth_key="post")
    print("== counterbalanced scoring (half events partner=pre) ==")
    bal = report(all_events, "counterbalanced", truth_key=None)
    cells, match = table_2x2(all_events)
    print("2x2 cells:", json.dumps({k: round(v, 3) for k, v in cells.items()}, ensure_ascii=False))
    print("impression-match:", json.dumps({k: [round(v[0], 3), v[1]] for k, v in match.items()}, ensure_ascii=False))
    tests = {
        "post_truth:post_vs_pre": mcnemar(all_events, "pre", "post", truth_key="post"),
        "counterbalanced:post_vs_pre": mcnemar(all_events, "pre", "post", truth_key=None),
        "counterbalanced:post_vs_none": mcnemar(all_events, "none", "post", truth_key=None),
    }
    ci = {"counterbalanced_post_minus_pre": cluster_boot(per_group, "pre", "post", truth_key=None)}
    result = {"groups": groups, "variant": args.variant, "per_group": per_group,
              "post_truth": post, "counterbalanced": bal, "cells_2x2": cells,
              "impression_match": match, "tests": tests, "cluster_boot": ci}
    pathlib.Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("tests:", json.dumps(tests, ensure_ascii=False))
    print("written:", args.out)


if __name__ == "__main__":
    main()
