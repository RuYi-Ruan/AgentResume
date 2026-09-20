"""Gated Bob controller: take the serving role only when it is actually useful.

Same as `PredictionRoleBob`, except that a FETCH prediction (Alice goes to get
an onion => Bob should serve) is deferred while there is no pot ready, Bob is
not already holding a dish/soup, and no cooking pot is close to finishing.
Until the gate opens, Bob keeps supplying onions instead of waiting idle.
"""
from __future__ import annotations

from ocres.generic_agents import PredictionRoleBob
from ocres.recipes import agent_held, pot_kinds, soup_cook_remaining


class PredictionRoleBobGated(PredictionRoleBob):
    def __init__(self, grid, me=1, horizon=700, ready_window=10):
        super().__init__(grid, me, horizon)
        self.ready_window = ready_window
        self._serve_gate_open = False

    def update_prediction(self, intent, facility="unknown"):
        ok = super().update_prediction(intent, facility)
        if ok:
            self._serve_gate_open = False
        return ok

    def _serve_useful(self, state):
        """Leave the supply role only when nothing needs onions any more, or
        when we are already committed by what we hold / a pot is ready."""
        if agent_held(state, self.me) in ("dish", "soup"):
            return True
        kinds = pot_kinds(state, self.grid)
        if any(k == "ready" for k in kinds.values()):
            return True
        # still someone needs onions? then keep supplying
        return not any(k in ("empty", "items1", "items2") for k in kinds.values())

    def action(self, state, deliveries):
        if (
            self.commitment == "serve_one_soup"
            and not self._serve_gate_open
            and not self._serve_useful(state)
        ):
            # keep supplying until serving would actually help; remember the
            # original serve commitment and restore it once the gate opens
            self._set_role("cook")
            result = super(PredictionRoleBob, self).action(state, deliveries)
            held = agent_held(state, self.me)
            self._last_seen_held = held
            return result
        if self.commitment == "serve_one_soup" and not self._serve_gate_open:
            self._serve_gate_open = True
        return super().action(state, deliveries)
