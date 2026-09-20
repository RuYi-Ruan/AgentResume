"""Auditable M18 Qwen requests; invalid answers are never repaired."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import urllib.error
import urllib.request

from ocres.llm import API_KEY, BASE_URL
from ocres.m18_bob import majority, validate_response
from ocres.m18_prompt import build_prompt


def trim_prompt(messages, maximum_chars=4096):
    """Conservative input cap: remove only lowest-ranked H facts."""
    messages = [dict(item) for item in messages]
    user = json.loads(messages[1]["content"])
    while len(messages[0]["content"])+len(messages[1]["content"]) > maximum_chars:
        history = [name for name in user["facts"] if name.startswith("H")]
        if not history:
            raise ValueError("current facts and schema exceed input cap")
        user["facts"].pop(history[-1])
        messages[1]["content"] = json.dumps(user, ensure_ascii=False, separators=(",", ":"))
    return messages, set(user["facts"])


def request_once(messages, options):
    if not API_KEY:
        raise RuntimeError("M18 Qwen API key is absent")
    body = {"model": options["model"], "messages": messages,
            "temperature": options["temperature"], "max_tokens": options["max_tokens"],
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(BASE_URL.rstrip("/")+"/chat/completions",
                                 data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer "+API_KEY}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result


def decode_response(result, fact_ids, vocabulary):
    try:
        raw = result["choices"][0]["message"]["content"]
        parsed = json.loads(raw)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None, ["non_json_or_missing_content"]
    good, errors = validate_response(parsed, fact_ids, vocabulary)
    return (parsed if good else None), errors


def utc():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"))+"\n")
        handle.flush()


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def plan_requests(public_rows, history, rows, options, stage, group, prompt_version="v1"):
    planned = []
    for event_index, row in enumerate(public_rows):
        for condition in options["conditions"]:
            segments = history[condition] if condition != "none" else []
            messages, _, vocabulary = build_prompt(row["public_event"], segments, rows, prompt_version)
            messages, fact_ids = trim_prompt(messages, options["max_input_tokens"])
            for repeat in range(options["repeats"]):
                planned.append({"id": f"{stage}:g{group}:e{event_index}:c{condition}:r{repeat}",
                                "event_index": event_index, "event_hash": row["event_hash"],
                                "condition": condition, "repeat": repeat,
                                "messages": messages, "fact_ids": sorted(fact_ids),
                                "vocabulary": vocabulary})
    random.Random(18977+group+(0 if stage.endswith("development") else 100)).shuffle(planned)
    return planned


def run_requests(plan, options, output, budget_file, request_fn=request_once):
    """Append an attempt *before* sending it; resume skips completed IDs only."""
    records = read_jsonl(output)
    completed = {item["id"] for item in records if item["kind"] == "completion"}
    attempted = Counter(item["id"] for item in records if item["kind"] == "attempt")
    budget = len(read_jsonl(budget_file))
    for item in plan:
        if item["id"] in completed:
            continue
        if attempted[item["id"]] >= 1+options["transport_retries"]:
            append_jsonl(output, {"kind": "completion", "id": item["id"], "utc": utc(),
                                  "valid": False, "errors": ["transport_exhausted"]})
            continue
        answer, errors, response = None, ["transport_exhausted"], None
        for attempt in range(attempted[item["id"]], 1+options["transport_retries"]):
            if budget >= options["request_limit"]:
                raise RuntimeError("approved M18 request cap reached")
            append_jsonl(budget_file, {"id": item["id"], "utc": utc(), "attempt": attempt})
            append_jsonl(output, {"kind": "attempt", "id": item["id"], "utc": utc(),
                                  "attempt": attempt})
            budget += 1
            try:
                response = request_fn(item["messages"], options)
                answer, errors = decode_response(response, set(item["fact_ids"]), item["vocabulary"])
                break  # invalid model output is not a transport error
            except urllib.error.HTTPError as exc:
                errors = [f"http_{exc.code}"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                errors = [type(exc).__name__]
            if attempt < options["transport_retries"]:
                time.sleep(4*(attempt+1))
        append_jsonl(output, {"kind": "completion", "id": item["id"], "utc": utc(),
                              "event_index": item["event_index"], "event_hash": item["event_hash"],
                              "condition": item["condition"], "repeat": item["repeat"],
                              "valid": answer is not None, "parsed": answer, "errors": errors,
                              "raw_response": response})
        print(json.dumps({"id": item["id"], "valid": answer is not None, "budget_used": budget}), flush=True)


def summarize(plan, output, truths):
    completed = {row["id"]: row for row in read_jsonl(output) if row["kind"] == "completion"}
    expected = {row["id"] for row in plan}
    if set(completed) != expected:
        raise RuntimeError(f"incomplete Qwen results: {len(completed)}/{len(expected)}")
    grouped = {}
    for item in plan:
        key = item["event_index"], item["condition"]
        grouped.setdefault(key, [None]*3)[item["repeat"]] = completed[item["id"]]["parsed"]
    events = []
    for (index, condition), predictions in sorted(grouped.items()):
        voted = majority(predictions)
        truth = truths[index]["scoring_only"]["post"]
        events.append({"event_index": index, "condition": condition,
                       "truth_intent": truth["intent"], "truth_facility": truth["target_facility"],
                       "prediction": voted, "intent_correct": voted["intent"] == truth["intent"],
                       "facility_correct": voted["target_facility"] == truth["target_facility"],
                       "valid_votes": sum(p is not None for p in predictions)})
    return events
