"""v2 LLM player: partial-observation hybrid agent with partner intent board.

Both Alice and Bob are the same class. Capability/impression differences are
pure CONTEXT:
  experiences (Alice): top-k lessons injected before each decision
  bob_knows  (Bob):    None = unaware; list = aware (same E as Alice) with
                       instruction to simulate Alice-with-E before guessing.
A shared board (one-tick delayed) lets each player see the partner's recent
intentions, so division of labour is emergent, not scripted.
"""
from __future__ import annotations

from collections import deque

from ocres.executor import STAY
from ocres.recipes import agent_held, pot_kinds
from ocres.llm_agent import GOAL_NAMES
from ocres.sensor import visible_state, obs_text, MAP_PRIOR
from ocres.executor import Executor

HOLD_TRIGGER = 12
STALL_TRIGGER = 6
CADENCE = 50

GOAL_DESC = GOAL_NAMES


class LLMPlayer:
    def __init__(self, grid, me, chat, name="player", experiences=(), bob_knows=None, board=None, retrieve_k=None):
        self.grid = grid
        self.me = me
        self.chat = chat  # chat(system_text, user_text) -> dict (JSON object)
        self.name = name
        self.retrieve_k = retrieve_k if retrieve_k is not None else len(list(experiences))
        self.experiences = list(experiences)
        self.bob_knows = bob_knows  # if not None: this player is a guessing Bob
        self.exec = Executor(grid, me)
        self.board = board if board is not None else deque(maxlen=6)
        self.mem_pots = {}
        self.goal = None
        self.target = None
        self.intent = None
        self.intent_target = None
        self.call_count = 0
        self.cause = None
        self.alice_guess = None
        self.guesses = []  # (tick, guess)
        self._hold = self._stall = self._since = 0
        self._spin = 0
        self._prev_goal = None
        self._prev_pos = None
        self._snap = None
        self._pending = None
        self._decision_seq = 0
        self._decision_this_tick = None
        self._guess_for = None

    # ------------------------------------------------------------- helpers
    def _is_bob(self):
        return self.bob_knows is not None

    def _remaining(self, state):
        """Total onions still needed by all pots (truth used by the motor)."""
        kinds = pot_kinds(state, self.grid)
        tot = 0
        for k in kinds.values():
            if k == "empty":
                tot += 3
            elif k.startswith("items"):
                tot += 3 - int(k[5:])
        return tot

    def _fetch_feasible(self, state, seen, now):
        """Motor capacity filter: fetching is only legal if there is room for
        the onion and we are not racing the partner for the last slot."""
        rem = self._remaining(state)
        if rem < 1:
            return False
        op = state.players[1 - self.me]
        if agent_held(state, 1 - self.me) == "onion":
            return rem >= 2  # partner is bringing one; only fetch if two needed
        if rem == 1:
            # last onion: only the agent closer to an accepting pot should go
            acc = [p for p, k in pot_kinds(state, self.grid).items() if k in ("empty", "items1", "items2")]
            me = state.players[self.me].position
            dme = min(abs(p[0] - me[0]) + abs(p[1] - me[1]) for p in acc) if acc else 10 ** 6
            dop = min(abs(p[0] - op.position[0]) + abs(p[1] - op.position[1]) for p in acc) if acc else 10 ** 6
            return dme < dop or (dme == dop and self.me == 0)
        return True

    def _legal(self, held, seen, now, state):
        kinds = set()
        for k, v in seen.items():
            if isinstance(v, dict):
                kind = v["kind"]
                if kind == "cooking" and now - v.get("t", now) >= 22:
                    kinds.add("ready")
                    continue
            else:
                kind = v[0] if isinstance(v, tuple) else str(v)
            kinds.add(kind)
        if held == "soup":
            return ["DELIVER"]
        if held == "dish":
            return ["PICKUP"] if "ready" in kinds else ["PRE", "HOLD"]
        if held == "onion":
            return ["PLACE", "HOLD"]
        out = ["HOLD"]
        if self._fetch_feasible(state, seen, now):
            out.insert(0, "FETCH")
        if "items3" in kinds:
            out.append("COOK_START")
        if kinds & {"cooking", "ready"}:
            out.append("GET_DISH")
        return out

    def _machine_target(self, goal, state):
        kinds = pot_kinds(state, self.grid)  # motor may use truth for placement
        me = state.players[self.me].position
        if goal == "FETCH":
            locs = self.grid.onion_locs
            return locs[0]
        if goal in ("PLACE", "COOK_START"):
            for p in sorted(kinds):
                if goal == "PLACE" and kinds[p] in ("empty", "items1", "items2"):
                    return p
                if goal == "COOK_START" and kinds[p] == "items3":
                    return p
            return self.grid.pot_locs[0]
        if goal == "PICKUP":
            for p in sorted(kinds):
                if kinds[p] == "ready":
                    return p
            return self.grid.pot_locs[0]
        if goal == "GET_DISH":
            return self.grid.dish_locs[0]
        if goal == "DELIVER":
            return self.grid.serve_locs[0]
        if goal == "PRE":
            return self.grid.pot_locs[0]
        return None

    # ------------------------------------------------------------ prompts
    def _exp_lines(self):
        pool = getattr(self, "_cur_subs", None)
        if pool is None:
            pool = self.experiences
        out = []
        for i, e in enumerate(pool):
            if isinstance(e, dict):
                txt = (
                    f"经验[{e.get('id', i)}]：仅当局面确实符合【{e.get('situation', '')}】时，可参考该经验："
                    f"{e.get('decide', '')}。注意：先核对局面确实匹配，若与其他更紧急事项冲突则以当下实际为准，"
                    f"采纳时在cause里报告编号，不采纳填null。"
                )
            else:
                txt = str(e)
            out.append(txt)
        return out

    def _system(self):
        role_line = ""
        if self._is_bob():
            if self.bob_knows:
                role_line = (
                    "你是Bob，负责协助Alice并判断Alice的意图。你持有Alice最近被注入的经验库（如下），"
                    "Alice 注入了这些经验、能力可能因此提升。猜她意图时请用模拟法："
                    "“如果我是Alice且带着这些经验面对此刻局面，我会选什么意图”。\n"
                    "经验库：\n" + "\n".join(self._exp_lines())
                )
            else:
                role_line = "你是Bob，负责协助Alice并判断Alice的意图。你只知道Alice是一名普通厨师，按常识协助她。"
        else:
            if self.experiences:
                role_line = "你是Alice。你有一些过往积累的经验（如下）。决策时若当前局面符合某条经验，请遵循它并在cause里报告其编号。\n" + "\n".join(
                    self._exp_lines()
                )
            else:
                role_line = "你是Alice，一名厨师。凭基本做菜常识与观察行动。"
        role_line2 = ""
        if self.name == "Alice":
            role_line2 = "协作偏好：你主攻下料——往锅里放洋葱与点火(COOK_START)；锅要补料时由你优先负责取洋葱。"
        elif self._is_bob():
            role_line2 = "协作偏好：你主攻取盘/取汤/交付(DELIVER)与协助；锅要补料时通常由Alice负责，你仅在需要时搭手。"
        return (
            f"你在Overcooked厨房中扮演{self.name}。{role_line} {role_line2}\n"
            "做汤基本流程：锅需要洋葱时选FETCH取洋葱再PLACE放进去（共3个）；"
            "锅放满3个后空手interact点火(COOK_START)，煮约20tick；"
            "只有某锅在煮或已ready时才需要GET_DISH取盘；ready后用盘PICKUP取汤，再DELIVER到出餐口(每次交付+20分)。\n"
            "协作常识：留意伙伴最近意图分工——若伙伴已在取盘/取汤，你就补料/下料/等待；不要两人同时取盘；"
            "手上拿着盘时无法拿洋葱，所以锅还没开始煮时别提前拿盘。\n"
            "规则：动作由系统执行，你只需在决策点选择一个子目标并输出JSON。子目标含义：\n"
            + "\n".join(f"  {k}: {v}" for k, v in GOAL_NAMES.items())
            + "\n只输出JSON。"
        )

    def _alice_pool(self, state):
        """Possible intents of the partner (Alice) given her held item and the
        pot situation (Bob's partial knowledge of her hand)."""
        held_a = agent_held(state, 0)
        return self._legal(held_a, self.mem_pots, state.timestep, state)

    def _user(self, state, spin=0):
        vis = visible_state(state, self.grid, self.me, self.mem_pots)
        self.mem_pots = vis["seen_pots"]
        partner = None
        for e in self.board:
            if e[0] != self.name:
                partner = e
                break
        txt = obs_text(vis, now=state.timestep)
        legal = self._legal(vis["held"], vis["seen_pots"], state.timestep, state)
        # compass hint (static map sense, not a decision)
        me = tuple(state.players[self.me].position)
        hints = []
        for label, feat in [("左洋葱台", self.grid.onion_locs[0]), ("右洋葱台", self.grid.onion_locs[1]),
                            ("盘台", self.grid.dish_locs[0]), ("出餐口", self.grid.serve_locs[0])]:
            dx, dy = feat[0] - me[0], feat[1] - me[1]
            sx = "右" if dx > 0 else "左" if dx < 0 else ""
            sy = "下" if dy > 0 else "上" if dy < 0 else ""
            hints.append(f"{label}在{('你正' if not sx and not sy else '')}{sy}{sx}约{abs(dx) + abs(dy)}步")
        for p in self.grid.pot_locs:
            dx, dy = p[0] - me[0], p[1] - me[1]
            sx = "右" if dx > 0 else "左" if dx < 0 else ""
            sy = "下" if dy > 0 else "上" if dy < 0 else ""
            hints.append(f"锅{p}在{'你正' if not sx and not sy else ''}{sy}{sx}约{abs(dx) + abs(dy)}步")
        dir_line = "；".join(hints)
        parts = [
            f"{txt}",
            "方位：" + dir_line,
            f"伙伴最近意图：{partner[1:4] if partner else '暂无'}",
            f"此刻你可执行的子目标：{legal}（从这些里选；不确定选HOLD等待）",
        ]
        if spin >= 1:
            parts.append(f"注意：你上一次的选择({self.goal})没有取得进展，请这次改选一个能真正推进的目标。")
        extra = ""
        if self._is_bob():
            apool = self._alice_pool(state)
            extra = f'；额外输出"alice_guess": 你判断Alice此刻最可能在执行的子目标（只从这组【Alice可执行】集合里选：{apool}）'
        ask_cause = "，cause(若遵循了某条经验填其编号否则null)" if (not self._is_bob() and self.experiences) else ""
        return "\n".join(parts) + f"\n请输出JSON {{goal, target, reason(≤30字){ask_cause}{', alice_guess' if self._is_bob() else ''}}}。" + extra

    def _retrieve(self, vis_text):
        if not self.experiences or self._is_bob():
            self._cur_subs = self.experiences if self._is_bob() else []
            return
        stop = {"一个", "你", "锅", "的", "并", "且", "没有", "看到", "是"}
        sc = []
        for e in self.experiences:
            sit = str(e.get("situation", "")) if isinstance(e, dict) else str(e)
            toks = [w for w in sit.replace("，", " ").replace("(", " ").replace(")", " ").split() if len(w) >= 2 and w not in stop]
            sc.append((sum(vis_text.count(w) for w in toks), e))
        sc.sort(key=lambda x: -x[0])
        picked = [e for s, e in sc if s > 0][: self.retrieve_k]
        self._cur_subs = picked

    # -------------------------------------------------------- decision
    def _decide(self, state):
        vis = visible_state(state, self.grid, self.me, self.mem_pots)
        prev = getattr(self, "_prev_goal", None)
        pos = tuple(state.players[self.me].position)
        self._spin = (self._spin + 1) if (prev is not None and prev == self.goal and pos == getattr(self, "_prev_pos", None)) else 0
        self._retrieve(obs_text(vis, now=state.timestep))
        d = self.chat(self._system(), self._user(state, spin=self._spin))
        self._cur_subs = None
        self.call_count += 1
        g = d.get("goal")
        legal = self._legal(vis["held"], vis["seen_pots"], state.timestep, state)
        if g not in legal:
            g = "HOLD"
        if self._spin >= 2 and g == prev and prev != "HOLD" and len(legal) > 1:
            g = next((x for x in legal if x != prev), g)
        self._prev_goal, self._prev_pos = g, pos
        self.goal, self.target = g, self._machine_target(g, state)
        self.cause = d.get("cause")
        if self._is_bob():
            guess = d.get("alice_guess")
            if guess not in self._alice_pool(state):
                guess = None
            self.alice_guess = guess
            self.guesses.append((state.timestep, guess))
        self._decision_seq += 1
        self._decision_this_tick = self._decision_seq
        # Publish only actual decisions.  action() flushes this on the next
        # environment tick, so Bob cannot observe Alice's decision early.
        self._pending = (self.name, self.goal, str(self.target), self.cause, self._decision_seq)
        self._hold = self._stall = 0

    def _goal_complete(self, state):
        held = agent_held(state, self.me)
        kinds = pot_kinds(state, self.grid)
        g = self.goal
        if g == "FETCH":
            return held == "onion"
        if g == "PLACE":
            return held != "onion"
        if g == "GET_DISH":
            return held == "dish"
        if g == "PICKUP":
            return held == "soup"
        if g == "DELIVER":
            return held != "soup"
        if g == "COOK_START":
            return not any(k == "items3" for k in kinds.values())
        return False

    # ------------------------------------------------------------ tick
    def _any_accept(self, state):
        kinds = pot_kinds(state, self.grid)
        return any(k in ("empty", "items1", "items2") for k in kinds.values())

    def action(self, state, deliveries):
        # A decision made at tick t becomes visible at tick t+1.
        if self._pending is not None:
            self.board.appendleft(self._pending)
            self._pending = None
        self._decision_this_tick = None
        self._guess_for = None
        need = False
        guess_trigger = False
        if self._is_bob():
            ae = [e for e in self.board if e[0] == "Alice"]
            cur = ae[0] if ae else None
            if cur is not None and cur != getattr(self, "_last_alice", None):
                self._last_alice = cur
                need = True
                guess_trigger = True
                self._guess_for = cur[4]
        if self.goal is None:
            need = True
        elif self.goal == "HOLD":
            self._hold += 1
            if self._hold >= HOLD_TRIGGER:
                need = True
        self._since += 1
        if self._since >= CADENCE:
            need = True
            self._since = 0
        if need:
            self._decide(state)
        if self.goal in ("HOLD",):
            act = STAY
        elif self.goal == "PRE" or (self.goal == "PLACE" and agent_held(state, self.me) == "onion" and not self._any_accept(state)):
            # hold position off the pot stands so the serving partner is never blocked
            park = (3, 2)
            if state.players[self.me].position == park:
                act = STAY
            else:
                act = self.exec.stand_action(state, park) or STAY
        else:
            act = self.exec.interact_action(state, self.target) if self.target else None
            if self._goal_complete(state):
                act = None  # finished -> let the decision logic pick the next goal
            if act is None:
                self._stall += 1
                if self._stall >= STALL_TRIGGER or self._goal_complete(state):
                    self._decide(state)
                    act = self.exec.interact_action(state, self.target) if self.target else None
                    if act is None and self._stall >= STALL_TRIGGER:
                        # unjam: step to a nearby free cell instead of repeating
                        op = state.players[1 - self.me].position
                        me = state.players[self.me].position
                        for c in sorted(self.grid.passable):
                            if c != me and c != op and c != self.target:
                                act = self.exec.stand_action(state, c)
                                if act is not None:
                                    self._stall = 0
                                    break
            else:
                self._stall = 0
        self.intent, self.intent_target = self.goal, self.target
        # asymmetric back-off: same blocked move repeated at the same spot ->
        # yield one tick so the partner can pass
        pos = state.players[self.me].position
        if (
            act is not None
            and act != STAY
            and act == getattr(self, "_last_act", None)
            and pos == getattr(self, "_last_pos", None)
        ):
            self._rep = getattr(self, "_rep", 0) + 1
        else:
            self._rep = 0
        self._last_act, self._last_pos = act, pos
        if self._rep >= (3 if self.me == 0 else 4):
            self._rep = 0
            act = STAY
        return act, self.intent, self.intent_target, {
            "name": self.name,
            "guess": self.alice_guess if self._is_bob() else None,
            "cause": self.cause,
            "decision_event": self._decision_this_tick is not None,
            "decision_id": self._decision_this_tick,
            # Event fields are populated once.  The legacy guess/cause fields
            # above remain state snapshots for trace inspection only.
            "guess_event": self.alice_guess if (self._is_bob() and guess_trigger) else None,
            "guess_for": self._guess_for if guess_trigger else None,
            "cause_event": self.cause if (not self._is_bob() and self._decision_this_tick is not None) else None,
        }
