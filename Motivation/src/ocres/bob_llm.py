"""LLM-decided Bob role controller.

Bob keeps the deterministic motor/role executor from `PredictionRoleBob`, but
instead of the fixed rule ("Alice FETCH => Bob serves"), it asks Qwen at each
task boundary which complementary job to take now:

    supply_one_onion  ~ go fetch/place onions (role "cook")
    serve_one_soup    ~ handle dish/pickup/deliver (role "serve")

The prompt only sees public state (pots, holds, positions, deliveries) plus the
already-received intent prediction. Decisions are cached by a state hash so a
re-run never repeats an API call.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

from ocres.generic_agents import PredictionRoleBob
from ocres.recipes import agent_held, pot_kinds, soup_cook_remaining

SYSTEM = (
    "你是Overcooked里的Bob。Alice是你的搭档。你要决定自己现在做哪份工作来最大化出汤数量：\n"
    "- supply_one_onion：去拿洋葱并把它们放进锅（做供料）\n"
    "- serve_one_soup：去拿盘子、盛汤、送到出餐口（做服务）\n"
    "原则：两人分工互补；若还有锅缺洋葱而Alice正在取料/下料，你通常应该做服务；"
    "但如果没有锅接近完成而多口锅仍缺料，留在供料位往往更好。只输出JSON。"
)


def state_summary(state, grid, me, prediction, deliveries):
    kinds = pot_kinds(state, grid)
    pots = "; ".join(
        f"锅{p}:{k}" + (f"(剩{rem}tick)" if (rem := soup_cook_remaining(state, grid, p)) is not None else "")
        for p, k in sorted(kinds.items()))
    held_me = agent_held(state, me) or "无"
    held_al = agent_held(state, 1 - me) or "无"
    return (
        f"锅态：{pots}\n"
        f"你手上：{held_me}；Alice手上：{held_al}\n"
        f"你位置：{tuple(state.players[me].position)}；Alice位置：{tuple(state.players[1-me].position)}\n"
        f"已交付：{deliveries}碗\n"
        f"你对Alice的预测：{prediction.get('intent')}（目标 {prediction.get('target_facility')}）"
    )


class LLMRoleBob(PredictionRoleBob):
    def __init__(self, grid, me=1, horizon=700, chat_fn=None, cache=None):
        super().__init__(grid, me, horizon)
        self.chat_fn = chat_fn
        self.cache = cache if cache is not None else {}
        self.cache_hits = 0
        self.decisions = 0
        self.trace = []

    def update_prediction(self, intent, facility="unknown"):
        self.prediction_updates += 1
        self.predicted_intent = intent
        self.predicted_facility = facility
        self.accepted_prediction_updates += 1
        return True

    def _decide_task(self, state, deliveries):
        summary = state_summary(state, self.grid, self.me, {"intent": self.predicted_intent,
                                                            "target_facility": self.predicted_facility}, deliveries)
        key = hashlib.sha1((summary + f"|{self.me}").encode()).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            task = self.cache[key]["task"]
        else:
            user = summary + "\n请输出 JSON：{\"task\":\"supply_one_onion 或 serve_one_soup\",\"reason\":\"≤20字\"}"
            out = None
            try:
                out = self.chat_fn(SYSTEM, user)
                task = out.get("task")
            except Exception:  # network already retried inside chat; fall back to rule
                task = None
            if task not in ("supply_one_onion", "serve_one_soup"):
                # rule fallback: PRE/HOLD (Alice waits) -> Bob supplies
                task = "supply_one_onion" if self.predicted_intent in ("PRE", "HOLD") else "serve_one_soup"
            self.cache[key] = {"task": task, "summary": summary, "raw": out}
            if getattr(self, "cache_path", None):
                pathlib.Path(self.cache_path).write_text(json.dumps(self.cache, ensure_ascii=False), encoding="utf-8")
        self.decisions += 1
        self.trace.append((int(state.timestep), task))
        self.commitment = task
        self._commitment_started = False
        self._last_seen_held = None
        return task

    def action(self, state, deliveries):
        if self.commitment is None:
            self._decide_task(state, deliveries)
        return super().action(state, deliveries)
