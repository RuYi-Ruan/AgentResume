"""Paired same-battle analysis over the reference-team diagnostic JSON.

The diagnostic fixes four teammates to a reference checkpoint and swaps one identity across stages,
replaying the SAME 20 battles (5 fixed seeds x 4 episode indices) at every stage.  This script turns
that JSON into paired stage-to-stage differences: each pair is the same battle (same seed AND same
episode index), so the difference is a within-battle comparison, not a sample-size claim.

Usage:
    python analyze_reference_team.py results/<run>/reference_team_eval.json \
        --pairs u50:u1250 u0:u50 u600:u1250
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

METRICS = ("damage_dealt", "damage_taken", "kills", "alive_steps", "nearest_enemy_dist")


def load_per_battle(stage_entry: dict) -> dict[tuple[int, int], dict]:
    """(seed, episode_index) -> metrics, taken from the per-episode records."""
    out = {}
    for rec in stage_entry["per_episode"]:
        # per-episode records are flat (metrics at top level); tolerate a nested "metrics" dict
        metrics = rec.get("metrics", rec)
        out[(int(rec["seed"]), int(rec["episode"]))] = metrics
    return out


def stage_lookup(identity: dict) -> dict[str, dict]:
    return identity["stages"]


def paired_stats(a: dict, b: dict, metrics=METRICS):
    """b - a on the same battles, with per-seed-group means and sign counts."""
    shared = sorted(set(a) & set(b))
    per_pair = {m: [] for m in metrics}
    by_seed: dict[int, dict[str, list[float]]] = defaultdict(lambda: {m: [] for m in metrics})
    for key in shared:
        seed = key[0]
        for m in metrics:
            diff = float(b[key][m]) - float(a[key][m])
            per_pair[m].append(diff)
            by_seed[seed][m].append(diff)
    summary = {}
    for m in metrics:
        vals = per_pair[m]
        seed_means = [sum(v) / len(v) for v in (by_seed[s][m] for s in sorted(by_seed)) if v]
        summary[m] = {
            "mean": sum(vals) / len(vals) if vals else float("nan"),
            "pairs": len(vals),
            "pairs_positive": sum(1 for v in vals if v > 0),
            "pairs_negative": sum(1 for v in vals if v < 0),
            "seed_means": seed_means,
            "seeds_positive": sum(1 for v in seed_means if v > 0),
            "seeds_negative": sum(1 for v in seed_means if v < 0),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_path")
    ap.add_argument("--pairs", nargs="+", default=["u50:u1250"],
                    help="stage-label pairs as FROM:TO, e.g. u0:u50 u50:u1250")
    args = ap.parse_args()

    data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    label_of = {lab: st["update_count"] for lab, st in data["identities"][0]["stages"].items()}
    print(f"run: {data['run_dir']}  reference = u{data['reference_update_count']}")
    print(f"stages: " + ", ".join(f"{lab}=u{u}" for lab, u in label_of.items()))
    print(f"battles per stage: {data['statistics']['battles_per_configuration']} "
          f"(= {data['statistics']['eval_seeds']} seeds x {data['statistics']['eval_episodes_per_seed']})  "
          f"-> pairs are the same battle, not independent samples\n")

    # accept either the raw stage label ("stage1") or "u<update_count>" ("u50")
    by_update = {f"u{u}": lab for lab, u in label_of.items()}

    def resolve(tag: str) -> str:
        return by_update.get(tag, tag)

    for pair in args.pairs:
        raw_from, raw_to = pair.split(":")
        from_lab, to_lab = resolve(raw_from), resolve(raw_to)
        print(f"=== {from_lab} (u{label_of[from_lab]}) -> {to_lab} (u{label_of[to_lab]}) ===")
        print(f"  {'identity':<16}" + "".join(f"{m:>22}" for m in METRICS))
        for ident in data["identities"]:
            stages = stage_lookup(ident)
            a = load_per_battle(stages[from_lab])
            b = load_per_battle(stages[to_lab])
            stats = paired_stats(a, b)
            cells = []
            for m in METRICS:
                s = stats[m]
                cells.append(f"{s['mean']:>+9.2f} ({s['seeds_positive']}+/{s['seeds_negative']}-)")
            print(f"  {ident['agent'] + '/' + ident['unit_type']:<16}" + "".join(f"{c:>22}" for c in cells))
        print()


if __name__ == "__main__":
    main()
