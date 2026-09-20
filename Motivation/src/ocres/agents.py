"""Deterministic macro agents for multi-pot onion soup (forced_coordination).

Roles (cook/serve) alternate per delivery. The cook fills pots and starts
cooking; the server preps a dish, picks up ready soup and delivers it.

Policy knobs (E-levels plug in here):
  serial_pot   : L0-like -- cook never starts a second pot while another pot
                 is cooking/ready (waits for it to drain). False = fill any
                 accepting pot (overlap, Lk-like).
  fetch_ahead  : not used on multi-pot (kept for API compatibility).
"""
from __future__ import annotations

from ocres.executor import Executor
from ocres.recipes import agent_held, pot_kinds, soup_cook_remaining
from ocres.grid import STAY

FETCH, PLACE, COOK_START, GET_DISH, PICKUP, DELIVER, PRE, HOLD = (
    "FETCH",
    "PLACE",
    "COOK_START",
    "GET_DISH",
    "PICKUP",
    "DELIVER",
    "PRE",
    "HOLD",
)


class CookAgent:
    def __init__(self, grid, me, parallel_after_delay=None, role_fixed=None):
        self.grid = grid
        self.me = me
        self.exec = Executor(grid, me)
        # None = serial (never start a second pot while one is cooking/ready);
        # int >= 0 = allow filling another pot once a cooking pot has been
        # cooking for at least that many ticks. This is the E-level knob.
        self.parallel_after_delay = parallel_after_delay
        self.role_fixed = role_fixed  # 'cook'/'serve'/None(=alternate by delivery)
        self._mem_pot = None  # pot chosen for the onion currently carried
        self.intent = None
        self.intent_target = None
        # deterministic parking cells that are never interaction stands
        stands = set()
        for loc in self.grid.onion_locs + self.grid.pot_locs + self.grid.dish_locs + self.grid.serve_locs:
            for s, _ in self.grid.interact_stand_cells(loc):
                stands.add(s)
        self.free_cells = sorted(self.grid.passable - stands)

    # ---- helpers -------------------------------------------------------
    def other(self, state):
        return state.players[1 - self.me]

    def _stand(self, state, cell):
        act = self.exec.stand_action(state, cell)
        return act if act is not None else STAY

    def _d(self, state, a, b):
        p = self.grid.bfs(a, b, ())
        return len(p) if p is not None else 10**6

    def dish_stand(self):
        d = self.grid.dish_locs[0]
        return d[0], d[1] - 1

    def pot_app(self, pot):
        """Approach (stand-from) cell below/next to a pot for facing it."""
        stand, face = self.grid.interact_stand_cells(pot)[0]
        return stand[0] - face[0], stand[1] - face[1]

    def onion_station(self, state):
        """Free onion station nearest to me."""
        me = state.players[self.me].position
        opos = self.other(state).position
        best, bd = None, 10**9
        for loc in self.grid.onion_locs:
            stands = [s for s, _ in self.grid.interact_stand_cells(loc)]
            if opos in stands:
                continue
            for s in stands:
                dd = self._d(state, me, s)
                if dd < bd:
                    bd, best = dd, loc
        return best if best is not None else self.grid.onion_locs[0]

    def pick_accepting_pot(self, state, kinds, need_at_least=1):
        """Best pot that accepts onions: most ingredients (fewest left),
        tie-break by distance; requires >= need_at_least more onions."""
        me = state.players[self.me].position
        cands = []
        for p, k in kinds.items():
            if k in ("empty", "items1", "items2"):
                items = 0 if k == "empty" else int(k[5:])
                if 3 - items >= need_at_least:
                    cands.append((3 - items, -self._d(state, me, p), p))
        cands.sort()
        return cands[0][2] if cands else None

    # ---- role decisions ------------------------------------------------
    def role(self, deliveries):
        if self.role_fixed is not None:
            return self.role_fixed
        return "cook" if self.me == (deliveries % 2) else "serve"

    def choose_goal(self, state, deliveries):
        held = agent_held(state, self.me)
        kinds = pot_kinds(state, self.grid)
        other_held = agent_held(state, 1 - self.me)
        self._cur_role = self.role(deliveries)
        if self._cur_role == "serve":
            return self._serve_goal(state, held, kinds, other_held)
        return self._cook_goal(state, held, kinds, other_held)

    def _serve_goal(self, state, held, kinds, other_held):
        ready = [p for p, k in kinds.items() if k == "ready"]
        cooking = [p for p, k in kinds.items() if k == "cooking"]
        if held == "soup":
            return DELIVER, self.grid.serve_locs[0]
        if held == "dish":
            if ready:
                me = state.players[self.me].position
                target = min(ready, key=lambda p: self._d(state, me, p))
                return PICKUP, target
            if cooking:
                # wait at the app cell of the soonest-to-finish cooking pot
                so = {p: soup_cook_remaining(state, self.grid, p) or 999 for p in cooking}
                target = min(so, key=so.get)
                return PRE, target
            return HOLD, self.dish_stand()
        if held == "onion":  # role switched mid-carry; finish placing
            target = self.pick_accepting_pot(state, kinds) or self._mem_pot
            if target is not None:
                return PLACE, target
            return HOLD, self.dish_stand()
        # empty handed
        if ready and other_held not in ("soup", "dish", "onion"):
            return GET_DISH, self.grid.dish_locs[0]
        if cooking:
            return GET_DISH, self.grid.dish_locs[0]
        return HOLD, self.dish_stand()

    def _cook_goal(self, state, held, kinds, other_held):
        station = self.onion_station(state)
        cooking = any(k in ("cooking", "items3") for k in kinds.values())
        ready = any(k == "ready" for k in kinds.values())
        if held == "dish":  # role switched mid-carry
            if ready:
                return PICKUP, min(kinds, key=lambda p: (kinds[p] != "ready", p))
            return HOLD, self.dish_stand()
        if held == "onion":
            target = self._mem_pot
            if target is None or kinds.get(target, "empty") not in ("empty", "items1", "items2"):
                target = self.pick_accepting_pot(state, kinds)
            if target is not None:
                self._mem_pot = target
                return PLACE, target
            return HOLD, station  # no pot accepts; wait with onion
        # empty handed
        self._mem_pot = None
        items3 = [p for p, k in kinds.items() if k == "items3"]
        if items3:
            return COOK_START, items3[0]
        if self._may_fill(state, kinds):
            accepting = self.pick_accepting_pot(state, kinds)
            if accepting is not None:
                self._mem_pot = accepting
                if other_held == "onion":
                    alt = self.pick_accepting_pot(state, kinds, need_at_least=2)
                    if alt is not None:
                        self._mem_pot = alt
                        return FETCH, station
                    return HOLD, station
                return FETCH, station
        return HOLD, station

    def _may_fill(self, state, kinds):
        """May the cook fetch onions for a fresh pot right now?"""
        if not any(k in ("empty", "items1", "items2") for k in kinds.values()):
            return False
        cooking = [p for p, k in kinds.items() if k == "cooking"]
        ready = [p for p, k in kinds.items() if k == "ready"]
        if not cooking and not ready:
            return True  # nothing running: fill freely
        if self.parallel_after_delay is None:
            return False  # serial: wait until the running pot drains
        delay = self.parallel_after_delay
        for p in cooking:
            s = state.get_object(p)
            cook_t = getattr(s, "cook_time", 20)
            started = cook_t - (soup_cook_remaining(state, self.grid, p) or cook_t)
            if started >= delay:
                return True
        if ready:  # a finished pot is about to be picked up; treat as ready to fill
            return True
        return False

    # ---- executor bridging ---------------------------------------------
    def goal_action(self, state, goal, target):
        if goal == FETCH:
            if agent_held(state, self.me) == "onion":
                return None
            return self.exec.interact_action(state, target)
        if goal == PLACE:
            if agent_held(state, self.me) != "onion":
                return None
            if pot_kinds(state, self.grid).get(target, "empty") not in ("empty", "items1", "items2"):
                return STAY
            return self.exec.interact_action(state, target)
        if goal == COOK_START:
            if pot_kinds(state, self.grid).get(target) != "items3":
                return None
            return self.exec.interact_action(state, target)
        if goal == GET_DISH:
            if agent_held(state, self.me) == "dish":
                return None
            return self.exec.interact_action(state, self.grid.dish_locs[0])
        if goal == PICKUP:
            if agent_held(state, self.me) == "soup":
                return None
            if pot_kinds(state, self.grid).get(target) != "ready":
                return None
            return self.exec.interact_action(state, target)
        if goal == DELIVER:
            if agent_held(state, self.me) != "soup":
                return None
            return self.exec.interact_action(state, self.grid.serve_locs[0])
        if goal == PRE:
            # wait near the pot area but OFF the (2,2) choke lane
            prefer = (3, 2)
            cell = min(self.free_cells, key=lambda c: abs(c[0] - prefer[0]) + abs(c[1] - prefer[1]))
            return self._stand(state, cell)
        if goal == HOLD:
            return self._hold_action(state, target)
        return STAY

    def _hold_action(self, state, pref):
        if getattr(self, "_park_cell", None) is None:
            role = getattr(self, "_cur_role", "cook")
            prefer = (3, 1) if role == "cook" else (5, 2)  # free, off the (2,2) choke
            me = state.players[self.me].position
            op = self.other(state).position
            best, bd = None, 10**9
            for c in self.free_cells:
                if c == op or c == me:
                    continue
                dd = abs(c[0] - prefer[0]) + abs(c[1] - prefer[1])
                if dd < bd:
                    bd, best = dd, c
            self._park_cell = best
        if self._park_cell is None or self._park_cell == state.players[self.me].position:
            return STAY
        act = self.exec.stand_action(state, self._park_cell)
        return act if act is not None else STAY

    # ---- per-tick API ---------------------------------------------------
    def action(self, state, deliveries):
        goal, target = self.choose_goal(state, deliveries)
        if goal != getattr(self, "_prev_goal", None):
            self._park_cell = None
        self._prev_goal = goal
        self.intent, self.intent_target = goal, target
        act = self.goal_action(state, goal, target)
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
            return STAY, self.intent, self.intent_target
        return act, self.intent, self.intent_target
