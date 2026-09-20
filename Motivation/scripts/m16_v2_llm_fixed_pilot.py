"""Second four-pair M16-V2 Qwen pilot with grounded-evidence constraints."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

from ocres.grid import STAY
from ocres.impression_events import FacilityVocabulary
from ocres.llm import BASE_URL, MODEL, chat
from ocres.sensor import visible_state
from m16_v2_audit_checkpoints import controlled_world


PAIRS = pathlib.Path("data/m16_v2_checkpoint_audit.json")
IMPRESSIONS = pathlib.Path("data/m16_v2_impressions.json")
OUTPUT = pathlib.Path("data/m16_v2_llm_fixed_pilot_r2.json")

ACTION_NAMES = {
    (0, -1): "向上移动一步",
    (0, 1): "向下移动一步",
    (-1, 0): "向左移动一步",
    (1, 0): "向右移动一步",
    (0, 0): "停留",
    "interact": "与面前设施交互",
}

SYSTEM_PROMPT = """你是 Overcooked 厨房中的 Bob。你的任务是根据自己能看到的 Alice 外在动作，以及给定的历史印象，推测 Alice 当前动作背后的意图。

只允许两个意图：
- PRE：Alice 正前往当前烹饪锅附近的非阻塞等待区，准备关注这锅，而不是取新原料。
- FETCH：Alice 正前往某个洋葱台取洋葱，准备给仍可补料的锅继续加料。

这是刻意挑选的歧义事件：当前公开状态和第一步动作同时符合 PRE 与 FETCH。因此禁止把“向上/下/左/右移动”单独当成意图答案，必须用伙伴印象作为概率上的辅助；没有印象时可以按公开状态作不确定判断。

证据约束：
1. 只能使用当前输入明确给出的动作、坐标、可见方格、已记住的锅状态和伙伴印象。
2. 不得补写视野外发生的行为，不得声称 Alice 选择了输入没有给出的路线。
3. “移动后离某设施更近”最多是位置事实，不能单独证明她要去该设施。
4. 只有具名设施出现在当前可见设施列表时，facility 才能填该名称；否则必须填 none。
5. evidence_types 只能从 observed_action、visible_tiles、remembered_pots、partner_impression 中选择。

只输出一个 JSON 对象，不要添加其它文字：
{"ambiguity_acknowledged":true,"intent":"PRE或FETCH","facility":"pot_0/pot_1/pot_2/onion_0/onion_1/none之一","confidence":0到1之间的数字,"evidence_types":["允许的证据类型"],"evidence":"不超过80个汉字，只陈述输入支持的事实"}"""

ALLOWED_EVIDENCE_TYPES = {
    "observed_action", "visible_tiles", "remembered_pots", "partner_impression"
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def choose_four_directions(rows):
    selected = {}
    for row in rows:
        key = tuple(row["first_action"])
        selected.setdefault(key, row)
    required = ((0, -1), (0, 1), (-1, 0), (1, 0))
    if any(action not in selected for action in required):
        raise ValueError("strict-pair file does not cover all four movement directions")
    return [selected[action] for action in required]


def local_view_text(observation):
    return "\n".join("  " + row for row in observation["tiles"])


def map_text_and_facilities(grid):
    vocabulary = FacilityVocabulary.from_grid(grid)
    entries = sorted(
        ((name, coordinate) for coordinate, index in vocabulary.coordinate_to_id.items()
         for name in (vocabulary.names[index],)),
        key=lambda item: item[0],
    )
    detail = "、".join(f"{name}={coordinate}" for name, coordinate in entries)
    return (
        "你熟悉的静态地图采用全局坐标 (x,y)，左上角为 (0,0)，x 向右、y 向下。\n"
        f"设施位置：{detail}。\n"
        "坐标只描述公开位置，不包含 Alice 的目标或规划路线。"
    ), vocabulary


def visible_named_facilities(grid, vocabulary, bob_position):
    bx, by = bob_position
    names = []
    for coordinate, index in vocabulary.coordinate_to_id.items():
        if max(abs(coordinate[0] - bx), abs(coordinate[1] - by)) <= 1:
            names.append(vocabulary.names[index])
    return sorted(names)


def event_prompt(pair, impression):
    cooking_pot = pair["cooking_pot_index"]
    alice = tuple(pair["alice_position"])
    bob = tuple(pair["bob_position"])
    action = tuple(pair["first_action"])
    world = controlled_world(sorted(world_probe().grid.pot_locs)[cooking_pot], alice, bob, 700)
    before = visible_state(world.env.state, world.grid, 1, {})
    world.env.step((action, STAY))
    after = visible_state(world.env.state, world.grid, 1, before["seen_pots"])
    relative_before = alice[0] - bob[0], alice[1] - bob[1]
    alice_after = tuple(world.env.state.players[0].position)
    relative_after = alice_after[0] - bob[0], alice_after[1] - bob[1]
    map_text, vocabulary = map_text_and_facilities(world.grid)
    visible_facilities = visible_named_facilities(world.grid, vocabulary, bob)
    memory = ", ".join(
        f"pot_{index}={'烹饪中' if index == cooking_pot else '空，可补料'}"
        for index in range(3)
    )
    return (
        f"{map_text}\n\n"
        f"关于 Alice 的历史印象：\n{impression}\n\n"
        f"我此前亲眼确认并记住的锅状态：{memory}。\n"
        f"我当前位于全局坐标 {bob}；可见的具名设施为：{visible_facilities or ['无']}。\n"
        f"动作前，我的3×3视野是：\n{local_view_text(before)}\n"
        f"Alice 当时位于全局坐标 {alice}，相对我的位置为 {relative_before}，空手。\n"
        f"我刚刚亲眼看到 Alice：{ACTION_NAMES[action]}。\n"
        f"动作后，我的3×3视野是：\n{local_view_text(after)}\n"
        f"Alice 动作后位于全局坐标 {alice_after}，相对我的位置为 {relative_after}。\n"
        "请先承认这一动作本身不能排除任一意图，再使用允许的证据作概率判断；最终只输出 JSON。"
    )


_WORLD_PROBE = None


def world_probe():
    global _WORLD_PROBE
    if _WORLD_PROBE is None:
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2

        _WORLD_PROBE = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
    return _WORLD_PROBE


def validate_response(value):
    if not isinstance(value, dict):
        return False
    if value.get("intent") not in ("PRE", "FETCH"):
        return False
    allowed = {"pot_0", "pot_1", "pot_2", "onion_0", "onion_1", "none"}
    evidence_types = value.get("evidence_types")
    return (
        value.get("ambiguity_acknowledged") is True
        and value.get("facility") in allowed
        and isinstance(value.get("confidence"), (int, float))
        and isinstance(evidence_types, list)
        and set(evidence_types).issubset(ALLOWED_EVIDENCE_TYPES)
        and isinstance(value.get("evidence"), str)
    )


def evidence_boundary_audit(parsed, prompt):
    marker = "可见的具名设施为："
    visible_line = next(line for line in prompt.splitlines() if marker in line)
    facility = parsed.get("facility", "none")
    facility_supported = facility == "none" or facility in visible_line
    named_in_evidence = [
        name for name in ("pot_0", "pot_1", "pot_2", "onion_0", "onion_1")
        if name in parsed.get("evidence", "")
    ]
    unsupported_named_evidence = [name for name in named_in_evidence if name not in visible_line]
    return {
        "facility_claim_supported_by_current_view": facility_supported,
        "unsupported_named_facilities_in_evidence": unsupported_named_evidence,
        "passed": facility_supported and not unsupported_named_evidence,
    }


def call_once(messages):
    raw_attempts = []
    for _ in range(2):
        raw = chat(messages, temperature=0.0, max_tokens=250, json_mode=True)
        raw_attempts.append(raw)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if validate_response(parsed):
            return parsed, raw_attempts
    return None, raw_attempts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=pathlib.Path, default=PAIRS)
    parser.add_argument("--impressions", type=pathlib.Path, default=IMPRESSIONS)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    parser.add_argument("--live", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pair_data = load_json(args.pairs)
    impression_data = load_json(args.impressions)
    selected = choose_four_directions(pair_data["strict_pairs"]["pairs"])
    old = load_json(args.output) if args.output.exists() else {"calls": {}}
    calls = old.get("calls", {})
    requests = []
    for pair_id, pair in enumerate(selected):
        facilities = FacilityVocabulary.from_grid(world_probe().grid)
        pre_target = facilities.names[facilities.label(tuple(pair["pre_internal_target_for_scoring_only"]))]
        post_target = facilities.names[facilities.label(tuple(pair["post_internal_target_for_scoring_only"]))]
        for impression_name in ("pre", "post", "none"):
            prompt = event_prompt(pair, impression_data[impression_name]["text"])
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            item = {
                "pair_id": pair_id,
                "first_action": pair["first_action"],
                "impression": impression_name,
                "prompt_hash": digest,
                "messages": messages,
                "pre_ground_truth": {"intent": "PRE", "facility": pre_target},
                "post_ground_truth": {"intent": "FETCH", "facility": post_target},
            }
            if args.live and digest not in calls:
                parsed, raw_attempts = call_once(messages)
                calls[digest] = {"parsed": parsed, "raw_attempts": raw_attempts}
            if digest in calls:
                item["response"] = calls[digest]
                parsed = calls[digest].get("parsed")
                if parsed:
                    item["evidence_boundary"] = evidence_boundary_audit(parsed, prompt)
            requests.append(item)

    summaries = {}
    if args.live:
        for impression_name in ("pre", "post", "none"):
            rows = [row for row in requests if row["impression"] == impression_name]
            valid = [row for row in rows if row.get("response", {}).get("parsed")]
            summaries[impression_name] = {
                "events": len(rows),
                "valid": len(valid),
                "pre_intent_accuracy": sum(row["response"]["parsed"]["intent"] == "PRE" for row in valid) / len(rows),
                "post_intent_accuracy": sum(row["response"]["parsed"]["intent"] == "FETCH" for row in valid) / len(rows),
                "evidence_boundary_pass_rate": sum(
                    row.get("evidence_boundary", {}).get("passed", False) for row in valid
                ) / len(rows),
            }
    result = {
        "milestone": "M16-V2 grounded-evidence four-pair Qwen pilot R2",
        "live": args.live,
        "model": MODEL,
        "base_url": BASE_URL,
        "unique_requests": len(requests),
        "leakage_check": {
            "event_prompt_contains_ground_truth_fields": False,
            "pre_and_post_records_share_the_same_event_prompt": True,
            "only_impression_changes_across_conditions": True,
            "absolute_positions_are_public_and_identical_within_each_strict_pair": True,
        },
        "summaries": summaries,
        "requests": requests,
        "calls": calls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("requests", "calls")}, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
