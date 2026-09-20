"""Resumable public-event Qwen predictions during natural M18 B episodes."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time
import urllib.error

from ocres.m18_bob import majority
from ocres.m18_prompt import build_prompt
from ocres.m18_qwen import append_jsonl, decode_response, read_jsonl, request_once, trim_prompt, utc


class LiveQwen:
    def __init__(self, rows, history, options, output, budget_file, group,
                 prompt_version="v2_mapfix", request_fn=request_once):
        self.rows, self.history, self.options = rows, history, options
        self.output, self.budget_file, self.group = Path(output), Path(budget_file), group
        self.request_fn = request_fn
        self.prompt_version = prompt_version
        records = read_jsonl(self.output / "responses.jsonl")
        self.completed = {r["id"]: r for r in records if r["kind"] == "completion"}
        self.attempts = Counter(r["id"] for r in records if r["kind"] == "attempt")
        self.budget_used = len(read_jsonl(self.budget_file))
        self.prompts = {r["id"]: r for r in read_jsonl(self.output / "prompts.jsonl")}

    def _one(self, item, fact_ids, vocabulary):
        request_id = item["id"]
        if request_id in self.completed:
            return self.completed[request_id]["parsed"]
        answer, errors, response = None, ["transport_exhausted"], None
        for attempt in range(self.attempts[request_id], 1+self.options["transport_retries"]):
            if self.budget_used >= self.options["request_limit"]:
                raise RuntimeError("approved M18 request cap reached")
            append_jsonl(self.budget_file, {"id": request_id, "attempt": attempt, "utc": utc()})
            append_jsonl(self.output / "responses.jsonl", {"kind": "attempt", "id": request_id,
                                                           "attempt": attempt, "utc": utc()})
            self.attempts[request_id] += 1
            self.budget_used += 1
            try:
                response = self.request_fn(item["messages"], self.options)
                answer, errors = decode_response(response, fact_ids, vocabulary)
                break  # do not repair invalid model output
            except urllib.error.HTTPError as exc:
                errors = [f"http_{exc.code}"]
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                errors = [type(exc).__name__]
            if attempt < self.options["transport_retries"]:
                time.sleep(4*(attempt+1))
        completed = {"kind": "completion", "id": request_id, "utc": utc(),
                     "parsed": answer, "valid": answer is not None, "errors": errors,
                     "raw_response": response}
        append_jsonl(self.output / "responses.jsonl", completed)
        self.completed[request_id] = completed
        return answer

    def predict(self, event, condition, seed, query_index):
        history = self.history[condition] if condition != "none" else []
        messages, _, vocabulary = build_prompt(event, history, self.rows, self.prompt_version)
        messages, fact_ids = trim_prompt(messages, self.options["max_input_tokens"])
        query_id = f"B:g{self.group}:s{seed}:c{condition}:q{query_index}"
        record = {"id": query_id, "messages": messages, "fact_ids": sorted(fact_ids)}
        if query_id in self.prompts:
            if self.prompts[query_id] != record:
                raise RuntimeError("B public prompt differs on resume")
        else:
            append_jsonl(self.output / "prompts.jsonl", record)
            self.prompts[query_id] = record
        votes = [self._one({"id": query_id+f":r{repeat}", "messages": messages},
                           fact_ids, vocabulary) for repeat in range(self.options["repeats"])]
        voted = majority(votes)
        print(json.dumps({"query": query_id, "intent": voted["intent"],
                          "valid_votes": sum(x is not None for x in votes),
                          "budget_used": self.budget_used}), flush=True)
        return voted, sum(x is not None for x in votes)
