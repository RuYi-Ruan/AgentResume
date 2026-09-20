"""Geometry-independent scripted macro agents for new-layout validation."""
from __future__ import annotations

from ocres.agents import COOK_START, DELIVER, FETCH, GET_DISH, HOLD, PICKUP, PLACE, PRE
from ocres.grid import STAY
from ocres.recipes import agent_held, pot_kinds
from ocres.trainable import INTENT_ID, MacroIntentPolicy, ObservationSpec, TrainableMacroAgent


class ScriptedMacroAgent(TrainableMacroAgent):
    """A role policy using the trainable agent's generic target resolver.

    It is an environment/control diagnostic, not the Alice model used in the
    formal experiment.  No facility coordinates or parking cells are hardcoded.
    """

    def __init__(self, grid, me, role, parallel, horizon=700):
        spec = ObservationSpec(
            width=len(grid.gw.terrain_mtx[0]),
            height=len(grid.gw.terrain_mtx),
            max_pots=len(grid.pot_locs),
            version="m16-layout-v1",
        )
        model = MacroIntentPolicy(spec.dim, hidden_dim=8)
        super().__init__(grid, me, model, spec=spec, horizon=horizon, max_goal_ticks=50)
        if role not in ("cook", "serve", "alternate"):
            raise ValueError(f"unknown role: {role}")
        self.role_mode = role
        self._active_role = "cook" if role == "alternate" and me == 0 else "serve" if role == "alternate" else role
        self.parallel = bool(parallel)
        self._safe_cells = self._derive_safe_parking_cells()
        self._parking_target = None

    def _derive_safe_parking_cells(self):
        transit = set()
        features = self.grid.onion_locs + self.grid.pot_locs + self.grid.dish_locs + self.grid.serve_locs
        for feature in features:
            for stand, face in self.grid.interact_stand_cells(feature):
                transit.add(stand)
                approach = stand[0] - face[0], stand[1] - face[1]
                if approach in self.grid.passable:
                    transit.add(approach)
        safe = sorted(self.grid.passable - transit)
        return safe or sorted(self.grid.passable)

    def _scripted_intent(self, state, role):
        held = agent_held(state, self.me)
        kinds = pot_kinds(state, self.grid)
        accepting = any(kind in ("empty", "items1", "items2") for kind in kinds.values())
        full = any(kind == "items3" for kind in kinds.values())
        cooking = any(kind == "cooking" for kind in kinds.values())
        ready = any(kind == "ready" for kind in kinds.values())

        # Always finish an object already held, even if roles later become dynamic.
        if held == "onion":
            return PLACE if accepting else HOLD
        if held == "dish":
            if ready:
                return PICKUP
            return PRE if cooking else HOLD
        if held == "soup":
            return DELIVER

        if role == "serve":
            if ready or cooking:
                return GET_DISH
            return HOLD

        if full:
            return COOK_START
        running = cooking or ready
        if accepting and (self.parallel or not running):
            return FETCH
        # A conservative cook does not freeze in place while another pot is
        # available. It moves to a safe monitoring area near the running pot.
        # This makes PRE and the parallel cook's FETCH share an observable
        # initial route where the layout permits it.
        if running:
            return PRE
        return HOLD

    def _decide(self, state, deliveries):
        self._active_role = (
            ("cook" if self.me == (deliveries % 2) else "serve")
            if self.role_mode == "alternate"
            else self.role_mode
        )
        intent = self._scripted_intent(state, self._active_role)
        self.begin_intent(state, deliveries, INTENT_ID[intent])
        self._parking_target = None
        if intent == PRE:
            self._parking_target = self._select_safe_parking_cell(state, self.intent_target)
        elif intent == HOLD:
            self._parking_target = self._select_safe_parking_cell(state, self._hold_preferred(state))
        return {"decision_id": self.decision_id, "policy": f"scripted_{self.role_mode}"}

    def _hold_preferred(self, state):
        # Stay put when the current cell is already safe; otherwise move only
        # as far as needed to vacate an interaction stand or its approach.
        # A serving partner waits near the shared workspace so it can react to
        # either side of the kitchen instead of idling at a random spawn.
        if self._active_role == "serve":
            width = len(self.grid.gw.terrain_mtx[0])
            height = len(self.grid.gw.terrain_mtx)
            return width // 2, height // 2
        return tuple(state.players[self.me].position)

    def _select_safe_parking_cell(self, state, preferred):
        occupied = tuple(state.players[1 - self.me].position)
        candidates = [cell for cell in self._safe_cells if cell != occupied]
        if not candidates:
            return tuple(state.players[self.me].position)
        position = tuple(state.players[self.me].position)
        return min(
            candidates,
            key=lambda cell: (
                abs(cell[0] - preferred[0]) + abs(cell[1] - preferred[1]),
                abs(cell[0] - position[0]) + abs(cell[1] - position[1]),
                cell,
            ),
        )

    def _safe_parking_action(self, state, preferred):
        if self._parking_target not in self.grid.passable:
            self._parking_target = self._select_safe_parking_cell(state, preferred)
        target = self._parking_target
        action = self.exec.stand_action(state, target)
        return action if action is not None else STAY

    def _execute(self, state, deliveries):
        if self.intent == PRE:
            return self._safe_parking_action(state, self.intent_target)
        if self.intent == HOLD:
            return self._safe_parking_action(state, self._hold_preferred(state))
        return super()._execute(state, deliveries)


class LayoutAwareTrainableAgent(TrainableMacroAgent):
    """Trainable policy adapter with parking derived from arbitrary geometry."""

    def __init__(self, *args, role_hint="cook", **kwargs):
        super().__init__(*args, **kwargs)
        self.role_hint = role_hint
        helper = ScriptedMacroAgent.__new__(ScriptedMacroAgent)
        helper.grid = self.grid
        self._safe_cells = ScriptedMacroAgent._derive_safe_parking_cells(helper)
        self._parking_target = None

    def _hold_preferred(self, state):
        if self.role_hint == "serve":
            width = len(self.grid.gw.terrain_mtx[0])
            height = len(self.grid.gw.terrain_mtx)
            return width // 2, height // 2
        return tuple(state.players[self.me].position)

    def _select_safe_parking_cell(self, state, preferred):
        occupied = tuple(state.players[1 - self.me].position)
        candidates = [cell for cell in self._safe_cells if cell != occupied]
        if not candidates:
            return tuple(state.players[self.me].position)
        position = tuple(state.players[self.me].position)
        return min(
            candidates,
            key=lambda cell: (
                abs(cell[0] - preferred[0]) + abs(cell[1] - preferred[1]),
                abs(cell[0] - position[0]) + abs(cell[1] - position[1]),
                cell,
            ),
        )

    def _safe_parking_action(self, state, preferred):
        if self._parking_target not in self.grid.passable:
            self._parking_target = self._select_safe_parking_cell(state, preferred)
        target = self._parking_target
        return self.exec.stand_action(state, target) or STAY

    def begin_intent(self, state, deliveries, choice):
        super().begin_intent(state, deliveries, choice)
        self._parking_target = None
        if self.intent == PRE:
            self._parking_target = self._select_safe_parking_cell(state, self.intent_target)
        elif self.intent == HOLD:
            self._parking_target = self._select_safe_parking_cell(state, self._hold_preferred(state))

    def _execute(self, state, deliveries):
        if self.intent == PRE:
            return self._safe_parking_action(state, self.intent_target)
        if self.intent == HOLD:
            return self._safe_parking_action(state, self._hold_preferred(state))
        return super()._execute(state, deliveries)


class PredictionRoleBob(ScriptedMacroAgent):
    """Common M16-V2 executor that makes one bounded task commitment."""

    def __init__(self, grid, me=1, horizon=700):
        super().__init__(grid, me, role="serve", parallel=True, horizon=horizon)
        self.predicted_intent = "FETCH"
        self.predicted_facility = "unknown"
        self.prediction_updates = 0
        self.commitment = None
        self._commitment_started = False
        self._last_seen_held = None
        self.completed_commitments = {"supply_one_onion": 0, "serve_one_soup": 0}
        self.accepted_prediction_updates = 0
        self.ignored_prediction_updates = 0
        self.route_yields = 0
        self.facility_coordinates = {}
        for prefix, locations in (
            ("onion", self.grid.onion_locs),
            ("pot", self.grid.pot_locs),
            ("dish", self.grid.dish_locs),
            ("serve", self.grid.serve_locs),
        ):
            for index, coordinate in enumerate(sorted(tuple(value) for value in locations)):
                self.facility_coordinates[f"{prefix}_{index}"] = coordinate

    def update_prediction(self, intent, facility="unknown"):
        """Accept a prediction only when no bounded task is in progress."""

        self.prediction_updates += 1
        if self.commitment is not None:
            self.ignored_prediction_updates += 1
            return False
        self.predicted_intent = intent
        self.predicted_facility = facility
        self.commitment = "supply_one_onion" if intent in (PRE, HOLD) else "serve_one_soup"
        self._commitment_started = False
        self._last_seen_held = None
        self.accepted_prediction_updates += 1
        return True

    def _execute(self, state, deliveries):
        action = super()._execute(state, deliveries)
        facility = self.facility_coordinates.get(self.predicted_facility)
        if facility is None or not isinstance(action, tuple) or action == STAY:
            return action
        own = tuple(state.players[self.me].position)
        alice = tuple(state.players[1 - self.me].position)
        paths = []
        for stand, _ in self.grid.interact_stand_cells(facility):
            path = self.grid.bfs(alice, stand, occupied=[own])
            if path:
                paths.append(path)
        if not paths:
            return action
        reserved_next = min(paths, key=lambda path: (len(path), path))[0]
        own_next = own[0] + action[0], own[1] + action[1]
        if own_next == reserved_next:
            self.route_yields += 1
            return STAY
        return action

    def _set_role(self, role):
        if role != self.role_mode:
            self.role_mode = role
            self.intent = None
            self._parking_target = None

    def _finish_completed_commitment(self, held):
        completed = (
            self._commitment_started
            and held is None
            and (
                (self.commitment == "supply_one_onion" and self._last_seen_held == "onion")
                or (self.commitment == "serve_one_soup" and self._last_seen_held == "soup")
            )
        )
        if completed:
            self.completed_commitments[self.commitment] += 1
            self.commitment = None
            self._commitment_started = False
            self._set_role("serve")

    def action(self, state, deliveries):
        held = agent_held(state, self.me)
        self._finish_completed_commitment(held)
        if self.commitment == "supply_one_onion":
            if not self._commitment_started and held in (None, "onion"):
                self._commitment_started = True
            self._set_role("cook" if self._commitment_started else "serve")
        elif self.commitment == "serve_one_soup":
            if not self._commitment_started and held in (None, "dish", "soup"):
                self._commitment_started = True
            self._set_role("serve")
        else:
            self._set_role("serve")
        result = super().action(state, deliveries)
        self._last_seen_held = held
        return result
