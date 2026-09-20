"""Hybrid LLM agents: the scripted FSM runs every tick as the default; the
model is consulted only at sparse decision points, keeping live-API calls low.

Decision points:
  - first tick of the episode;
  - the FSM has been idle (HOLD) for >= HOLD_TRIGGER ticks;
  - the current goal cannot make progress for >= STALL_TRIGGER ticks;
  - hard cadence cap (CADENCE ticks).
The model's choice is validated against the allowed goal vocabulary and the
deterministic Executor performs the atomic actions.
"""
from __future__ import annotations

import json

from ocres.executor import STAY
from ocres.recipes import agent_held, pot_kinds
from ocres.agents import CookAgent, HOLD, PRE

HOLD_TRIGGER = 6
STALL_TRIGGER = 8
CADENCE = 90

GOAL_NAMES = {
    "FETCH": "去洋葱台拿一个洋葱",
    "PLACE": "把手中的洋葱放进目标锅",
    "COOK_START": "锅已满3料，点火开始烹饪（空手 interact 锅）",
    "GET_DISH": "去取盘子",
    "PICKUP": "用盘子从做好的锅取汤",
    "DELIVER": "把汤送到出餐口交付",
    "PRE": "在目标锅附近持盘待命",
    "HOLD": "原地等待/不做动作",
}


def role_legal(held, role, kinds):
    ready = any(k == "ready" for k in kinds.values())
    items3 = any(k == "items3" for k in kinds.values())
    acc = any(k in ("empty", "items1", "items2") for k in kinds.values())
    if role == "serve":
        if held == "soup":
            return ["DELIVER"]
        if held == "dish":
            return ["PICKUP"] if ready else ["PRE", "HOLD"]
        if held == "onion":
            return ["PLACE"] if acc else ["HOLD"]
        if ready:
            return ["GET_DISH", "HOLD"]
        if any(k == "cooking" for k in kinds.values()):
            return ["GET_DISH", "PRE", "HOLD"]
        return ["HOLD"]
    # cook
    if held == "onion":
        return ["PLACE", "HOLD"]
    if held == "dish":
        return ["PICKUP", "HOLD"] if ready else ["HOLD"]
    if held == "soup":
        return ["DELIVER"]
    out = []
    if items3:
        out.append("COOK_START")
    if acc:
        out.append("FETCH")
    out.append("HOLD")
    return out


def state_summary(state, grid):
    kinds = pot_kinds(state, grid)
    pots = ", ".join(f"锅{x}:{k}" for x, k in sorted(kinds.items()))
    p0, p1 = state.players
    h0 = "无" if not p0.has_object() else p0.get_object().name
    h1 = "无" if not p1.has_object() else p1.get_object().name
    return (
        f"我方在{p0.position}面朝{p0.orientation}手拿{h0}；"
        f"伙伴在{p1.position}面朝{p1.orientation}手拿{h1}；"
        f"锅态:{pots}；已过{state.timestep}tick"
    )


SYSTEM_TMPL = (
    "你是 Overcooked 厨房的厨师({role})。{card}"
    "你可以选择子目标；动作由系统执行。"
    "只输出 JSON: {{\"goal\":目标名, \"target\":对象(无则null), \"reason\":短理由}}。"
    "目标名只能是给定可选项之一。"
)


class LLMCook:
    def __init__(self, grid, me, model, role_card="", label="LLM", role_fixed=None):
        self.grid = grid
        self.me = me
        self.model = model
        self.label = label
        self.role_card = role_card
        # parallel off => serial baseline; the model may still pick FETCH etc.
        self._fsm = CookAgent(grid, me, parallel_after_delay=None, role_fixed=role_fixed)
        self.intent = None
        self.intent_target = None
        self._hold = 0
        self._stall = 0
        self._since = 0
        self.call_count = 0
        self._last_act = None
        self._last_pos = None
        self._snap = None

    def _snapshot(self, state):
        return (agent_held(state, self.me), str(sorted((p, k) for p, k in pot_kinds(state, self.grid).items())))

    def _fsm_choice(self, state, deliveries):
        return self._fsm.choose_goal(state, deliveries)

    def _prompt_user(self, state, deliveries, fsm_goal, fsm_target):
        allowed = "; ".join(f"{g}:{GOAL_NAMES[g]}" for g in GOAL_NAMES)
        base = self._fsm_choice(state, deliveries)
        return (
            f"当前状态：{state_summary(state, self.grid)}\n"
            f"本局已交付 {deliveries} 碗；脚本基线此刻会选 {base[0]}({base[1]})。\n"
            f"可选子目标：{allowed}\n"
            "请选择此刻最合理的子目标。"
        )

    def _machine_target(self, state, goal, deliveries):
        """Concrete target cell for a goal, computed from state (never from the
        model's free text)."""
        kinds = pot_kinds(state, self.grid)
        me = state.players[self.me].position
        if goal == "FETCH":
            return self._fsm.onion_station(state)
        if goal == "PLACE":
            p = self._fsm.pick_accepting_pot(state, kinds)
            return p if p is not None else self.grid.pot_locs[0]
        if goal == "COOK_START":
            for p, k in sorted(kinds.items()):
                if k == "items3":
                    return p
            return self.grid.pot_locs[0]
        if goal == "PICKUP":
            cand = [p for p, k in kinds.items() if k == "ready"]
            return min(cand, key=lambda p: abs(p[0] - me[0]) + abs(p[1] - me[1])) if cand else self.grid.pot_locs[0]
        if goal == "PRE":
            cand = [p for p, k in kinds.items() if k in ("cooking", "ready")]
            return min(cand, key=lambda p: abs(p[0] - me[0]) + abs(p[1] - me[1])) if cand else self.grid.pot_locs[0]
        if goal == "DELIVER":
            return self.grid.serve_locs[0]
        if goal == "GET_DISH":
            return self.grid.dish_locs[0]
        return None

    def _apply_model(self, d, state, deliveries):
        g = d.get("goal")
        if g not in GOAL_NAMES:
            g, _ = self._fsm_choice(state, deliveries)
        self.intent = g
        self.intent_target = self._machine_target(state, g, deliveries)
        self._hold = self._stall = 0
        self._snap = self._snapshot(state)
        return self.intent, self.intent_target

    def _decide(self, state, deliveries):
        fsm_g, fsm_t = self._fsm_choice(state, deliveries)
        kinds = pot_kinds(state, self.grid)
        held = agent_held(state, self.me)
        legal = role_legal(held, self._fsm.role(deliveries), kinds)
        d = self.model.decide(
            {
                "role": self._fsm.role(deliveries),
                "summary": state_summary(state, self.grid),
                "deliveries": deliveries,
                "fsm": (fsm_g, str(fsm_t)),
                "allowed": list(GOAL_NAMES),
                "legal": legal,
                "_state": state,
                "_deliveries": deliveries,
            },
            role_card=self.role_card,
            label=self.label,
        )
        self.call_count += 1
        if d.get("goal") not in legal:
            d["goal"] = fsm_g if fsm_g in legal else (legal[0] if legal else "HOLD")
        return self._apply_model(d, state, deliveries)

    def action(self, state, deliveries):
        fsm_g, fsm_t = self._fsm_choice(state, deliveries)  # scripted default
        do_ask = False
        if self.intent is None:
            do_ask = True
        elif self.intent == HOLD:
            self._hold += 1
            if self._hold >= HOLD_TRIGGER:
                do_ask = True
        self._since += 1
        if self._since >= CADENCE:
            do_ask = True
            self._since = 0
        if do_ask:
            self._decide(state, deliveries)
        act = self._fsm.goal_action(state, self.intent, self.intent_target)
        if act is None and self.intent not in (HOLD, PRE):
            changed = self._snapshot(state) != self._snap
            if changed or self._stall >= STALL_TRIGGER:
                self._decide(state, deliveries)
                act = self._fsm.goal_action(state, self.intent, self.intent_target)
            else:
                self._stall += 1
        else:
            self._stall = 0
        return act, self.intent, self.intent_target


# ------------------------------------------------------------- models
class FakeModel:
    """Offline harness: delegate to a scripted FSM (rule baseline)."""

    def __init__(self, fsm):
        self.fsm = fsm

    def decide(self, ctx, role_card="", label=""):
        st = ctx.get("_state")
        dl = ctx.get("_deliveries")
        if st is None:
            return {"goal": "HOLD", "target": None}
        g, t = self.fsm.choose_goal(st, dl)
        return {"goal": g, "target": t}


class RemoteModel:
    """Qwen/Qwen3.5-9B via SiliconFlow OpenAI-compatible endpoint."""

    def __init__(self):
        from ocres import llm

        self.llm = llm

    def decide(self, ctx, role_card="", label=""):
        from ocres.llm import chat_json

        sys_txt = SYSTEM_TMPL.format(role=ctx["role"], card=role_card)
        pool = ctx.get("legal") or list(GOAL_NAMES)
        allowed = "; ".join(f"{g}:{GOAL_NAMES[g]}" for g in pool)
        extra = ""
        if ctx.get("ask_guess"):
            pool = "; ".join(f"{g}:{GOAL_NAMES[g]}" for g in (ctx.get("alice_pool") or list(GOAL_NAMES)))
            extra = f'\n额外：请输出 "alice_guess": 你判断 Alice(伙伴)此刻最可能在执行的子目标（只从这组【Alice可执行】目标中选一个：{pool}）。'
        user = (
            f"当前状态：{ctx['summary']}\n"
            f"本局已交付 {ctx['deliveries']} 碗。\n"
            f"此刻【你可执行】的子目标：{allowed}（请只从这些里选；HOLD=合理等待时也可选）\n"
            f"请选择此刻最合理的子目标。{extra}"
        )
        out = chat_json(
            [{"role": "system", "content": sys_txt}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=1500,
        )
        return out


class LLMBob(LLMCook):
    """Serve-side LLM cooperator that also *guesses Alice's next intent*.

    Its impression of Alice lives in role_card (injected each prompt).
    Per decision it stores (tick, guess) in self.guesses for later accuracy
    scoring against Alice's logged JSON self-report intent.
    """

    def __init__(self, grid, me, model, role_card="", label="LLM-Bob", impression_mode="static"):
        super().__init__(grid, me, model, role_card=role_card, label=label, role_fixed="serve")
        self.impression_mode = impression_mode
        self.guesses = []
        self._tick = 0
        from collections import deque

        self.obs = deque(maxlen=80)
        self._alice_prev = None
        self._pot_cook_prev = False

    def _observe(self, state):
        kinds = pot_kinds(state, self.grid)
        alice = state.players[0]
        apos = tuple(alice.position)
        cooking = any(k == "cooking" for k in kinds.values())
        acc = any(k in ("empty", "items1", "items2") for k in kinds.values())
        onion = self.grid.onion_locs[0]
        d_now = abs(apos[0] - onion[0]) + abs(apos[1] - onion[1])
        moving_to_onion = False
        if self._alice_prev is not None:
            moving_to_onion = d_now < self._alice_prev[1]
        self._alice_prev = (apos, d_now)
        if cooking and acc and not alice.has_object() and not self._pot_cook_prev:
            # a fresh 'pot cooking with another open pot' moment: sample whether
            # Alice heads to the onion station (parallel-fill) or not
            self.obs.append(1 if moving_to_onion else 0)
        self._pot_cook_prev = cooking

    def _card_now(self):
        if self.impression_mode != "auto" or len(self.obs) < 12:
            return self.role_card
        rate = sum(self.obs) / len(self.obs)
        from ocres.cards import updated_text

        return updated_text(rate, len(self.obs))

    def action(self, state, deliveries):
        self._tick = state.timestep
        self._observe(state)
        n_before = len(self.guesses)
        act, it, tg = super().action(state, deliveries)
        fresh = None
        if len(self.guesses) > n_before and self.guesses[-1][0] == self._tick:
            fresh = self.guesses[-1][1]
        return act, it, tg, {"bob_guess": fresh}

    def _decide(self, state, deliveries):
        kinds = pot_kinds(state, self.grid)
        held = agent_held(state, self.me)
        legal = role_legal(held, "serve", kinds)
        held_a = agent_held(state, 0)
        alice_pool = role_legal(held_a, "cook", kinds)
        fsm_g, fsm_t = self._fsm_choice(state, deliveries)
        d = self.model.decide(
            {
                "role": "serve",
                "summary": state_summary(state, self.grid),
                "deliveries": deliveries,
                "fsm": (fsm_g, str(fsm_t)),
                "allowed": list(GOAL_NAMES),
                "legal": legal,
                "alice_pool": alice_pool,
                "ask_guess": True,
            },
            role_card=self._card_now(),
            label=self.label,
        )
        self.call_count += 1
        if d.get("goal") not in legal:
            d["goal"] = fsm_g if fsm_g in legal else (legal[0] if legal else "HOLD")
        res = self._apply_model(d, state, deliveries)
        self._last_guess = d.get("alice_guess")
        self.guesses.append((self._tick, self._last_guess))
        return res
