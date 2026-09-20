"""Frozen public-only M18 prediction prompts and deterministic history retrieval."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json

from ocres.m18_bob import normalized_event


SYSTEM = """你是厨房里的 Bob。任务是根据刚观察到的 Alice 动作、当前可见事实以及过去真实相处的可见片段，猜她做出这个动作时想完成的任务目标。过去的记录不一定适用于现在。不要把推测说成观察；看不到的动作、物品、锅况和心理状态一律不能当作事实。如果证据不足，可以回答 unknown。地图和设施位置是双方已知的静态信息；目标设施是你的预测，不是输入已经告知的答案。
意图只可从以下八类选择：FETCH=去取洋葱；PLACE=将手中洋葱放入锅；COOK_START=启动满锅烹饪；GET_DISH=去取盘子；PICKUP=用盘子盛熟汤；DELIVER=将汤送到出餐台；PRE=去锅附近照看、准备或等待；HOLD=暂停任务、等待或让行。unknown=无法可靠判断，不是 Alice 的任务类。
只能引用输入中标有 C 或 H 的事实 ID，不得引用地图坐标作为额外观察到的 Alice 心理状态。只输出一个 JSON 对象，字段严格为 intent、target_facility、confidence、used_fact_ids。intent 为上述八类或 unknown；target_facility 为输入提供的同类型设施 ID、none（仅 HOLD）或 unknown；confidence 为 0 到 1 的数；used_fact_ids 为至少一个有效事实 ID 的数组。不要输出解释文字。"""

SYSTEM_V1 = SYSTEM
SYSTEM_V2 = SYSTEM.replace(
    "如果证据不足，可以回答 unknown。",
    "这里要求推测最可能的目标，不要求看到确凿证据。若可见的方向、持物、设施距离或过去可见片段提供相对线索，应从八类中选一个最可能的目标，并用较低 confidence 表示不确定。只有动作不可见或现有线索完全无法作相对判断时才回答 unknown。不要为作答编造输入没有的细节。")


def _frame(frame):
    frame = deepcopy(frame)
    now = frame.pop("t")
    for value in frame["remembered_pots"].values():
        value["age"] = now - value.pop("seen_t")
    return frame


def _segment_facts(segment):
    frames = segment["frames"]
    indices = sorted({0, 1, max(0, len(frames)-2), len(frames)-1})
    return {"frames": [{"relative_t": i, **_frame(frames[i])} for i in indices],
            "actions": [{"relative_t": i, "action": segment["actions"][i]}
                        for i in sorted({0, max(0, len(segment["actions"])-1)})],
            "end_reason": segment["end_reason"]}


def retrieval_score(event, segment):
    action = event["action"]
    first = segment["actions"][0]
    before = event["before"]
    historical = segment["frames"][0]
    score = 2 if action["kind"] == first["kind"] else 0
    if action.get("delta") is not None and action.get("delta") == first.get("delta"):
        score += 4
    if before["alice"] and historical["alice"] and before["alice"]["held"] == historical["alice"]["held"]:
        score += 2
    here, there = before["bob"]["position"], historical["bob"]["position"]
    score -= .05 * sum(abs(a-b) for a, b in zip(here, there))
    return score


def top_history(event, segments, limit=5):
    ordered = sorted(segments, key=lambda s: (-retrieval_score(event, s), s["source_seed"], s["start_t"]))
    return ordered[:limit]


@lru_cache(maxsize=3)
def _vocabulary(rows):
    result = {}
    for kind, marker in (("pot", "P"), ("onion", "O"), ("dish", "D"), ("serve", "S")):
        positions = sorted((x, y) for y, row in enumerate(rows) for x, cell in enumerate(row) if cell == marker)
        for index, position in enumerate(positions):
            result[f"{kind}_{chr(65+index)}"] = {"kind": kind, "position": list(position)}
    return result


def build_prompt(public_event, segments, rows, prompt_version="v1"):
    """No scoring-only row, checkpoint phase, or original source ID is accepted."""
    if set(public_event) != {"before", "after", "action"}:
        raise ValueError("unexpected public event fields")
    event = normalized_event(public_event)
    for name in ("before", "after"):
        if set(event[name]) != {"bob", "alice", "visible_pots", "remembered_pots"}:
            raise ValueError("unexpected public frame fields")
    vocabulary = _vocabulary(tuple(rows))
    facts = {"C1": {"when": "before_observed_action", **event["before"]},
             "C2": {"observed_action": event["action"]},
             "C3": {"when": "after_observed_action", **event["after"]}}
    for i, segment in enumerate(top_history(public_event, segments), 1):
        facts[f"H{i}"] = _segment_facts(segment)
    known_map = ([row.replace("1", " ").replace("2", " ") for row in rows]
                 if prompt_version == "v2_mapfix" else list(rows))
    user = {"known_map": known_map, "facility_ids": vocabulary, "facts": facts,
            "answer_schema": {"intent": "one of eight intent names or unknown",
                              "target_facility": "facility ID, none or unknown",
                              "confidence": "number 0..1", "used_fact_ids": "nonempty list of C/H IDs"}}
    if prompt_version not in ("v1", "v2", "v2_mapfix"):
        raise ValueError("unapproved prompt version")
    return ([{"role": "system", "content": SYSTEM_V1 if prompt_version == "v1" else SYSTEM_V2},
             {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))}],
            set(facts), vocabulary)
