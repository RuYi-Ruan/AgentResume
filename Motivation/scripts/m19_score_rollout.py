"""M19 score rollout: replay frozen events with saved Qwen predictions and
measure downstream team score (deliveries / points) per impression condition.

Local only (no API). Uses the M17 controller (PredictionRoleBob) and the
group's post-Alice checkpoint. Conditions:
  stale  = predict from the old impression (pre)
  updated= predict from the new impression (post)
  none   = predict without impression
  oracle = feed the true post intent (controller sanity / upper bound)
  inverted = feed the wrong intent (controller sanity / lower bound)
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import m17_fixed_online_rollout as mod  # noqa: E402
import m17_prepare_fixed_events as prep  # noqa: E402
from ocres.trainable import load_policy_checkpoint  # noqa: E402

RUN = "v1_waitfix_20260915"
DATA = pathlib.Path("data/m19") / RUN
CKPT = pathlib.Path("artifacts/m18") / RUN / "checkpoints"
COND_TO_IMPRESSION = {"stale": "pre", "updated": "post", "none": "none"}


def majority(vals):
    cnt = collections.Counter(v for v in vals if v not in (None, "unknown"))
    if not cnt:
        return "unknown"
    top = cnt.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "unknown"
    return top[0][0]


def load_predictions(group):
    d = json.loads((DATA / f"group_{group}" / "formal_test_v1_responses.json").read_text(encoding="utf-8"))
    by = collections.defaultdict(lambda: {"intent": [], "facility": []})
    for row in d["rows"]:
        key = (row["event_id"], row["impression"])
        parsed = row.get("response", {}).get("parsed") if "response" in row else None
        if parsed:
            by[key]["intent"].append(parsed.get("intent"))
            by[key]["facility"].append(parsed.get("target_facility"))
        else:
            by[key]["intent"].append("unknown")
            by[key]["facility"].append("unknown")
    return {k: {"intent": majority(v["intent"]), "target_facility": majority(v["facility"])} for k, v in by.items()}


def patched_prediction_for(condition, event, responses):
    if condition == "oracle":
        return dict(event["scoring_only"]["post"]), "oracle"
    if condition == "inverted":
        return dict(event["scoring_only"]["pre"]), "inverted"
    imp = COND_TO_IMPRESSION[condition]
    return responses.get((event["event_id"], imp)), f"saved_qwen_{imp}"


def run_group(group, conditions, window, events_limit):
    cfg = json.loads(pathlib.Path("configs/m18_v1.json").read_text(encoding="utf-8"))
    result = json.loads((pathlib.Path("data/m18") / RUN / f"group_{group}" / "result.json").read_text(encoding="utf-8"))
    rows = tuple(cfg["maps"][result["group"]["map"]])
    mod.AMBIGUOUS_KITCHEN_V2 = rows
    prep.AMBIGUOUS_KITCHEN_V2 = rows
    splits = json.loads((DATA / f"group_{group}" / "splits.json").read_text(encoding="utf-8"))
    events = splits["formal_test"]["events"][:events_limit]
    responses = load_predictions(group)
    mod.response_lookup = lambda _path: responses
    mod.prediction_for = patched_prediction_for
    post_model, spec, _ = load_policy_checkpoint(CKPT / f"group_{group}" / "post.pt", map_location="cpu")
    out = collections.defaultdict(list)
    for event in events:
        for cond in conditions:
            row = mod.run_trial(event, cond, post_model, spec, responses, window)
            out[cond].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="1,2,3,4,6,7")
    ap.add_argument("--conditions", default="stale,updated,none,oracle,inverted")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--events", type=int, default=50)
    ap.add_argument("--out", default=str(DATA / "score_rollout.json"))
    ap.add_argument("--bob", choices=("default", "gate", "llm"), default="default")
    ap.add_argument("--llm-cache", default=str(DATA / "llm_bob_calls.json"))
    args = ap.parse_args()
    if args.bob == "gate":
        from ocres.bob_gated import PredictionRoleBobGated

        mod.PredictionRoleBob = PredictionRoleBobGated
    elif args.bob == "llm":
        import pathlib as _p

        from ocres.bob_llm import LLMRoleBob
        from ocres.llm import chat_json

        cache_path = _p.Path(args.llm_cache)
        cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}

        def _chat(system, user):
            return chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             temperature=0.0, max_tokens=200)

        def _factory(grid, me=1, horizon=700):
            bob = LLMRoleBob(grid, me, horizon, chat_fn=_chat, cache=cache)
            bob.cache_path = cache_path
            return bob

        mod.PredictionRoleBob = _factory
    groups = [int(x) for x in args.groups.split(",")]
    conds = args.conditions.split(",")
    all_rows = {}
    for g in groups:
        rows = run_group(g, conds, args.window, args.events)
        all_rows[g] = rows
        line = " ".join(
            f"{c}: {sum(r['deliveries'] for r in rows[c]) / len(rows[c]):.2f}碗/"
            f"{sum(r['reward'] for r in rows[c]) / len(rows[c]):.1f}分" for c in conds)
        print(f"g{g}: {line}", flush=True)
    summary = {}
    for c in conds:
        pooled = [r for g in groups for r in all_rows[g][c]]
        summary[c] = {"n": len(pooled),
                      "mean_deliveries": sum(r["deliveries"] for r in pooled) / len(pooled),
                      "mean_reward": sum(r["reward"] for r in pooled) / len(pooled)}
    print("pooled:", json.dumps(summary, ensure_ascii=False))
    pathlib.Path(args.out).write_text(json.dumps({"per_group": all_rows, "summary": summary}, ensure_ascii=False), encoding="utf-8")
    print("written:", args.out)


if __name__ == "__main__":
    main()
