"""Prepare or run the M17 Qwen intent experiment on a frozen split.

Without ``--live`` this script only writes audited prompts.  Labels under
``scoring_only`` are attached after prompt construction and never passed to
the model.  Formal-test use is intentionally disabled until development is
accepted with ``--allow-formal``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

from ocres.grid import World
from ocres.impression_events import FacilityVocabulary
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.llm import BASE_URL, MODEL, chat
from ocres.m17_protocol import validate_intent_response


SPLITS = pathlib.Path("data/m17_fixed_event_splits.json")
IMPRESSIONS = pathlib.Path("data/m17_paired_impressions.json")
OUTPUT = pathlib.Path("data/m17_qwen_development.json")

SYSTEM_PROMPT = """你是Overcooked中的Bob。你刚亲眼看见Alice完成了一步动作，需要推测她当前动作背后的意图。

意图只有两种：
- PRE：Alice准备在正在烹饪的锅附近等待或照看，不准备自己去取新洋葱。
- FETCH：Alice准备自己去某个洋葱台取洋葱。

输入由程序签发的事实组成。当前这一小步经过刻意配对，本身同时可能属于PRE或FETCH。你可以结合过去亲眼看到的伙伴行为，但不得编造视野外经历、隐藏目标、路线或锅状态。

只输出以下四个字段的JSON，不要输出解释文字：
{"intent":"PRE或FETCH","target_facility":"设施名称或unknown","confidence":0到1之间的数字,"used_fact_ids":["引用的事实编号"]}

target_facility是推测，可以暂时位于视野外；它不是“已经看见”的事实。used_fact_ids只能引用输入实际提供的编号，至少引用一个。"""


def compact_frame(frame):
    return {
        "t": int(frame["t"]),
        "alice_position": frame.get("position"),
        "alice_held": frame.get("held"),
        "observed_action_result": frame.get("action_result"),
        "bob_position": frame.get("bob_position"),
    }


def representative_frames(frames):
    if len(frames) <= 4:
        return [compact_frame(frame) for frame in frames]
    return [
        compact_frame(frames[0]),
        compact_frame(frames[1]),
        compact_frame(frames[-2]),
        compact_frame(frames[-1]),
    ]


def retrieval_score(public_event, segment):
    current = public_event["realized_alice_action"]
    first = segment["frames"][0]["action_result"]
    score = 0.0
    if current.get("kind") == first.get("kind"):
        score += 2.0
    if current.get("delta") == first.get("delta"):
        score += 4.0
    current_bob = public_event["before"]["bob_position"]
    history_bob = segment["frames"][0].get("bob_position", current_bob)
    score -= 0.05 * (
        abs(current_bob[0] - history_bob[0]) + abs(current_bob[1] - history_bob[1])
    )
    return score


def impression_facts(public_event, impression_name, impression_data, k=5):
    if impression_name == "none":
        return []
    segments = impression_data[impression_name]["segments"]
    outcomes = Counter(row["outcome"] for row in segments)
    ranked = sorted(
        segments,
        key=lambda row: (-retrieval_score(public_event, row), row["event_id"]),
    )[:k]
    facts = [
        {
            "id": "H1",
            "value": {
                "past_visible_segments": len(segments),
                "outcomes": dict(sorted(outcomes.items())),
                "note": "只统计连续可见片段；lost_visibility之后没有补写",
            },
        }
    ]
    for index, segment in enumerate(ranked, start=2):
        facts.append({
            "id": f"H{index}",
            "value": {
                "visible_steps": int(segment["visible_steps"]),
                "representative_observed_frames": representative_frames(segment["frames"]),
                "observed_action_counts": dict(Counter(
                    frame["action_result"]["kind"] for frame in segment["frames"]
                )),
                "observed_outcome": segment["outcome"],
            },
        })
    return facts


def current_facts(public_event, vocabulary):
    facilities = sorted(
        (
            {"name": vocabulary.names[index], "position": list(coordinate)}
            for coordinate, index in vocabulary.coordinate_to_id.items()
        ),
        key=lambda item: item["name"],
    )
    return [
        {"id": "M1", "value": {"known_static_map_facilities": facilities}},
        {"id": "F1", "value": {"before_local_observation": public_event["before"]}},
        {"id": "F2", "value": {"realized_alice_action": public_event["realized_alice_action"]}},
        {"id": "F3", "value": {"after_local_observation": public_event["after"]}},
    ]


def build_messages(event, impression_name, impression_data, vocabulary):
    # Only the explicitly public subtree enters this function.
    public_event = event["public_event"]
    facts = current_facts(public_event, vocabulary)
    facts.extend(impression_facts(public_event, impression_name, impression_data))
    payload = {
        "task": "根据已签发事实推测Alice当前意图和目标设施",
        "facts": facts,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ], {fact["id"] for fact in facts}


def prompt_hash(messages):
    return hashlib.sha256(
        json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def call_validated(messages, fact_ids, facilities):
    attempts = []
    for _ in range(2):
        raw = chat(messages, temperature=0.0, max_tokens=180, json_mode=True)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
            errors = ["invalid_json"]
        else:
            valid, errors = validate_intent_response(parsed, fact_ids, facilities)
            if valid:
                attempts.append({"raw": raw, "parsed": parsed, "validation_errors": []})
                return parsed, attempts
        attempts.append({"raw": raw, "parsed": parsed, "validation_errors": errors})
    return None, attempts


def metrics(rows):
    valid = [row for row in rows if row.get("response", {}).get("parsed") is not None]
    by_impression = {}
    for name in ("pre", "post", "none"):
        subset = [row for row in rows if row["impression"] == name]
        parsed = [row for row in subset if row.get("response", {}).get("parsed") is not None]
        denominator = len(subset)
        by_impression[name] = {
            "requests": denominator,
            "valid": len(parsed),
            "pre_intent_accuracy": (
                sum(row["response"]["parsed"]["intent"] == "PRE" for row in parsed)
                / denominator if denominator else 0.0
            ),
            "post_intent_accuracy": (
                sum(row["response"]["parsed"]["intent"] == "FETCH" for row in parsed)
                / denominator if denominator else 0.0
            ),
            "pre_target_accuracy": (
                sum(
                    row["response"]["parsed"]["target_facility"]
                    == row["scoring_only"]["pre"]["target_facility"]
                    for row in parsed
                ) / denominator if denominator else 0.0
            ),
            "post_target_accuracy": (
                sum(
                    row["response"]["parsed"]["target_facility"]
                    == row["scoring_only"]["post"]["target_facility"]
                    for row in parsed
                ) / denominator if denominator else 0.0
            ),
            "unknown_target_rate": (
                sum(
                    row["response"]["parsed"]["target_facility"] == "unknown"
                    for row in parsed
                ) / denominator if denominator else 0.0
            ),
            "history_fact_use_rate": (
                sum(
                    any(
                        fact_id.startswith("H")
                        for fact_id in row["response"]["parsed"]["used_fact_ids"]
                    )
                    for row in parsed
                ) / denominator if denominator else 0.0
            ),
        }
    repeat_groups = {}
    for row in valid:
        key = row["event_id"], row["impression"]
        repeat_groups.setdefault(key, []).append(row["response"]["parsed"]["intent"])
    majority_fractions = [
        max(Counter(values).values()) / len(values)
        for values in repeat_groups.values()
    ]
    return {
        "requests": len(rows),
        "valid": len(valid),
        "repeat_groups": len(repeat_groups),
        "unanimous_repeat_rate": (
            sum(len(set(values)) == 1 for values in repeat_groups.values())
            / len(repeat_groups) if repeat_groups else None
        ),
        "mean_repeat_majority_fraction": (
            sum(majority_fractions) / len(majority_fractions)
            if majority_fractions else None
        ),
        "by_impression": by_impression,
        "core_post_updated_minus_stale": (
            by_impression["post"]["post_intent_accuracy"]
            - by_impression["pre"]["post_intent_accuracy"]
        ),
        "full_factorial_correct_impression_accuracy": (
            by_impression["pre"]["pre_intent_accuracy"]
            + by_impression["post"]["post_intent_accuracy"]
        ) / 2.0,
        "full_factorial_mismatched_impression_accuracy": (
            by_impression["post"]["pre_intent_accuracy"]
            + by_impression["pre"]["post_intent_accuracy"]
        ) / 2.0,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=pathlib.Path, default=SPLITS)
    parser.add_argument("--impressions", type=pathlib.Path, default=IMPRESSIONS)
    parser.add_argument("--split", choices=("development", "formal_test"), default="development")
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--allow-formal", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.split == "formal_test" and not args.allow_formal:
        raise SystemExit("formal test is locked; pass --allow-formal only after development approval")
    split_data = json.loads(args.splits.read_text(encoding="utf-8"))
    impression_data = json.loads(args.impressions.read_text(encoding="utf-8"))
    events = split_data[args.split]["events"][: args.events]
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
    vocabulary = FacilityVocabulary.from_grid(probe.grid)
    allowed_facilities = {name for name in vocabulary.names if name != "none"}
    old = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    calls = old.get("calls", {})
    rows = []
    calls_sidecar = args.output.with_suffix(args.output.suffix + ".calls.json")
    if calls_sidecar.exists():
        calls.update(json.loads(calls_sidecar.read_text(encoding="utf-8")))

    def save_calls():
        calls_sidecar.write_text(json.dumps(calls, ensure_ascii=False), encoding="utf-8")

    import time as _time
    from ocres.llm import LLMError as _LLMError

    for event in events:
        for impression_name in ("pre", "post", "none"):
            messages, fact_ids = build_messages(
                event, impression_name, impression_data, vocabulary
            )
            digest = prompt_hash(messages)
            for repeat in range(args.repeats):
                call_id = f"{digest}:{repeat}"
                if args.live and call_id not in calls:
                    for attempt in range(60):  # survive network outages: wait, retry
                        try:
                            parsed, attempts = call_validated(
                                messages, fact_ids, allowed_facilities
                            )
                            break
                        except _LLMError as exc:
                            msg = str(exc)
                            if "402" in msg or "insufficient" in msg:
                                save_calls()
                                raise SystemExit(
                                    f"API balance insufficient (HTTP 402); progress saved ({len(calls)} calls). "
                                    "Top up the SiliconFlow account, then re-run this exact command to resume."
                                )
                            print(f"[retry {attempt+1}/60] {call_id}: {msg[:120]}", flush=True)
                            _time.sleep(30)
                    else:
                        save_calls()
                        raise SystemExit(f"network unavailable after retries; progress saved ({len(calls)} calls)")
                    calls[call_id] = {"parsed": parsed, "attempts": attempts}
                    save_calls()
                row = {
                    "event_id": event["event_id"],
                    "split": args.split,
                    "impression": impression_name,
                    "repeat": repeat,
                    "prompt_hash": digest,
                    "messages": messages,
                    "allowed_fact_ids": sorted(fact_ids),
                    "scoring_only": event["scoring_only"],
                }
                if call_id in calls:
                    row["response"] = calls[call_id]
                rows.append(row)
    result = {
        "milestone": "M17 Qwen fixed-event development" if args.split == "development" else "M17 Qwen fixed-event formal test",
        "live": args.live,
        "qwen_calls_requested": len(rows) if args.live else 0,
        "model": MODEL,
        "base_url": BASE_URL,
        "split": args.split,
        "events": len(events),
        "repeats": args.repeats,
        "prompt_contract": "messages built from public_event and observable impression segments only",
        "metrics": metrics(rows) if args.live else {},
        "rows": rows,
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "calls")}, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
