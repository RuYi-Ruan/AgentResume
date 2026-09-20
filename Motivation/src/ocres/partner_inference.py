"""Trainable action-conditioned inference and a fixed downstream Bob policy."""
from __future__ import annotations

import torch
from torch import nn

from ocres.agents import (
    COOK_START,
    DELIVER,
    FETCH,
    GET_DISH,
    HOLD,
    PICKUP,
    PLACE,
    PRE,
    CookAgent,
)
from ocres.grid import STAY
from ocres.recipes import pot_kinds


COOK_SIDE = {FETCH, PLACE, COOK_START}
SERVE_SIDE = {GET_DISH, PICKUP, DELIVER}


class PartnerInferenceNet(nn.Module):
    """Shared encoder with separate intent and facility classification heads."""

    def __init__(self, input_dim, intent_count, facility_count, hidden_dim=128):
        super().__init__()
        self.input_dim = int(input_dim)
        self.intent_count = int(intent_count)
        self.facility_count = int(facility_count)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.intent_head = nn.Linear(self.hidden_dim, self.intent_count)
        self.facility_head = nn.Linear(self.hidden_dim, self.facility_count)

    def forward(self, observation):
        hidden = self.encoder(observation)
        return self.intent_head(hidden), self.facility_head(hidden)


class PredictionAwareBob(CookAgent):
    """Fixed, interpretable cooperation policy driven by Bob's latest guess.

    The same controller is used in every condition.  Intent selects the
    complementary role; facility identity avoids duplicate facility use and
    reserves Alice's estimated next path cell.
    """

    def __init__(self, grid, me, facilities):
        super().__init__(grid, me, parallel_after_delay=0)
        self.facilities = facilities
        self.predicted_intent = None
        self.predicted_facility = "none"
        self.facility_avoidances = 0
        self.route_yields = 0
        self.blocked_yields = 0
        self._last_position = None
        self._last_action = STAY
        self._blocked_ticks = 0

    def update_prediction(self, intent, facility):
        self.predicted_intent = intent
        self.predicted_facility = facility

    def _predicted_coordinate(self):
        if self.predicted_facility == "none":
            return None
        try:
            facility_id = self.facilities.names.index(self.predicted_facility)
        except ValueError:
            return None
        for coordinate, candidate_id in self.facilities.coordinate_to_id.items():
            if candidate_id == facility_id:
                return coordinate
        return None

    def _set_complementary_role(self):
        if self.predicted_intent in COOK_SIDE:
            self.role_fixed = "serve"
        elif self.predicted_intent in SERVE_SIDE:
            self.role_fixed = "cook"
        else:
            self.role_fixed = None

    def _valid_alternatives(self, state, goal, avoided):
        kinds = pot_kinds(state, self.grid)
        if goal == FETCH:
            candidates = self.grid.onion_locs
        elif goal == PLACE:
            candidates = [p for p, kind in kinds.items() if kind in ("empty", "items1", "items2")]
        elif goal == COOK_START:
            candidates = [p for p, kind in kinds.items() if kind == "items3"]
        elif goal == GET_DISH:
            candidates = self.grid.dish_locs
        elif goal == PICKUP:
            candidates = [p for p, kind in kinds.items() if kind == "ready"]
        elif goal == DELIVER:
            candidates = self.grid.serve_locs
        elif goal == PRE:
            candidates = [p for p, kind in kinds.items() if kind in ("cooking", "ready")]
        else:
            candidates = []
        return [tuple(value) for value in candidates if tuple(value) != avoided]

    def _avoid_duplicate_facility(self, state, goal, target):
        avoided = self._predicted_coordinate()
        if avoided is None or tuple(target) != avoided:
            return target
        alternatives = self._valid_alternatives(state, goal, avoided)
        if not alternatives:
            return target
        me = tuple(state.players[self.me].position)
        chosen = min(alternatives, key=lambda value: (self._d(state, me, value), value))
        self.facility_avoidances += 1
        return chosen

    def _distance_to_interaction(self, state, agent_index, facility, occupied=()):
        start = tuple(state.players[agent_index].position)
        distances = []
        for stand, _ in self.grid.interact_stand_cells(facility):
            path = self.grid.bfs(start, stand, occupied=occupied)
            if path is not None:
                distances.append(len(path))
        return min(distances) if distances else 10**6

    def _would_enter_reserved_route(self, state, action, own_target):
        facility = self._predicted_coordinate()
        if facility is None or not isinstance(action, tuple) or action == STAY:
            return False
        alice = tuple(state.players[1 - self.me].position)
        own = tuple(state.players[self.me].position)
        stands = self.grid.interact_stand_cells(facility)
        paths = []
        for stand, _ in stands:
            path = self.grid.bfs(alice, stand, occupied=[own])
            if path:
                paths.append(path)
        if not paths:
            return False
        reserved = min(paths, key=lambda value: (len(value), value))[0]
        own_next = own[0] + action[0], own[1] + action[1]
        if own_target is None or own_next != reserved or tuple(own_target) not in self.facilities.coordinate_to_id:
            return False
        alice_distance = self._distance_to_interaction(state, 1 - self.me, facility, occupied=[own])
        bob_distance = self._distance_to_interaction(state, self.me, tuple(own_target), occupied=[alice])
        # Approved tie-break: only the agent farther from its own facility yields.
        return bob_distance > alice_distance

    def _blocked_yield_action(self, state):
        position = tuple(state.players[self.me].position)
        occupied = tuple(state.players[1 - self.me].position)
        candidates = [
            cell
            for cell in self.grid.passable
            if cell != occupied and abs(cell[0] - position[0]) + abs(cell[1] - position[1]) == 1
        ]
        if not candidates:
            return STAY
        stands = {
            stand
            for feature in self.grid.onion_locs + self.grid.pot_locs + self.grid.dish_locs + self.grid.serve_locs
            for stand, _ in self.grid.interact_stand_cells(feature)
        }
        chosen = max(
            candidates,
            key=lambda cell: (
                cell not in stands,
                abs(cell[0] - occupied[0]) + abs(cell[1] - occupied[1]),
                -cell[0],
                -cell[1],
            ),
        )
        return chosen[0] - position[0], chosen[1] - position[1]

    def action(self, state, deliveries):
        position = tuple(state.players[self.me].position)
        if (
            self._last_position == position
            and isinstance(self._last_action, tuple)
            and self._last_action != STAY
        ):
            self._blocked_ticks += 1
        else:
            self._blocked_ticks = 0
        self._set_complementary_role()
        goal, target = self.choose_goal(state, deliveries)
        target = self._avoid_duplicate_facility(state, goal, target)
        if goal != getattr(self, "_prev_goal", None):
            self._park_cell = None
        self._prev_goal = goal
        self.intent, self.intent_target = goal, target
        action = self.goal_action(state, goal, target)
        action = action if action is not None else STAY
        if self._would_enter_reserved_route(state, action, target):
            self.route_yields += 1
            action = STAY
        if self._blocked_ticks >= 3:
            yielded = self._blocked_yield_action(state)
            if yielded != STAY:
                action = yielded
                self.blocked_yields += 1
            self._blocked_ticks = 0
        self._last_position = position
        self._last_action = action
        return action, goal, target
