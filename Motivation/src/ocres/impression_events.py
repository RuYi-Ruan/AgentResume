"""Leakage-audited events for action-conditioned partner inference.

An event is emitted only when Alice's *macro intent* changes.  Its observation
contains the public state before and after the first primitive action plus that
action itself.  Alice's hidden intent and target are labels, never features.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ocres.grid import STAY
from ocres.trainable import ObservationSpec, _position, encode_log_observation


PRIMITIVE_ACTIONS = ((0, -1), (0, 1), (1, 0), (-1, 0), STAY, "interact")
ACTION_ID = {action: index for index, action in enumerate(PRIMITIVE_ACTIONS)}


def canonical_action(value):
    """Normalize actions loaded from object arrays into hashable values."""

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        value = tuple(value)
    if isinstance(value, tuple):
        value = tuple(int(part) for part in value)
    if value not in ACTION_ID:
        raise ValueError(f"unknown primitive action: {value!r}")
    return value


@dataclass(frozen=True)
class FacilityVocabulary:
    """Stable public names for static facilities; coordinates stay internal."""

    names: tuple[str, ...]
    coordinate_to_id: dict

    @classmethod
    def from_grid(cls, grid):
        entries = []
        for prefix, locations in (
            ("onion", grid.onion_locs),
            ("pot", grid.pot_locs),
            ("dish", grid.dish_locs),
            ("serve", grid.serve_locs),
        ):
            for index, coordinate in enumerate(sorted(tuple(p) for p in locations)):
                entries.append((f"{prefix}_{index}", coordinate))
        names = tuple(name for name, _ in entries) + ("none",)
        mapping = {coordinate: names.index(name) for name, coordinate in entries}
        return cls(names=names, coordinate_to_id=mapping)

    @property
    def none_id(self):
        return len(self.names) - 1

    def label(self, target):
        if target is None:
            return self.none_id
        return self.coordinate_to_id.get(_position(target), self.none_id)


def encode_observed_action(pre_row, action, post_row, deliveries=0, horizon=700, spec=None):
    """Encode only information Bob can observe after Alice's first action."""

    spec = spec or ObservationSpec()
    action_one_hot = np.zeros(len(PRIMITIVE_ACTIONS), dtype=np.float32)
    action_one_hot[ACTION_ID[canonical_action(action)]] = 1.0
    return np.concatenate(
        (
            encode_log_observation(pre_row, deliveries, horizon, spec),
            action_one_hot,
            encode_log_observation(post_row, deliveries, horizon, spec),
        )
    ).astype(np.float32, copy=False)


def action_observation_dim(spec=None):
    spec = spec or ObservationSpec()
    return 2 * spec.dim + len(PRIMITIVE_ACTIONS)


def extract_intent_change_events(logs, grid, horizon=700, spec=None):
    """Create one event per actual intent transition, after its first action.

    The last tick is skipped because its public post-action state is unavailable
    in the existing log format.  A fresh ``decision_id`` with an unchanged
    intent does not create another event.
    """

    spec = spec or ObservationSpec()
    facilities = FacilityVocabulary.from_grid(grid)
    events = []
    previous_intent = None
    deliveries = 0
    for index in range(max(0, len(logs) - 1)):
        row, post_row = logs[index], logs[index + 1]
        intent = str(row["intent0"])
        changed = intent != previous_intent
        previous_intent = intent
        if changed:
            alice_action = canonical_action(row["a"][0])
            events.append(
                {
                    "t": int(row["t"]),
                    "intent": intent,
                    "facility_id": facilities.label(row.get("target0")),
                    "facility": facilities.names[facilities.label(row.get("target0"))],
                    "alice_action": alice_action,
                    "x": encode_observed_action(
                        row, alice_action, post_row, deliveries=deliveries, horizon=horizon, spec=spec
                    ),
                }
            )
        if float(row.get("r", 0)) > 0:
            deliveries += 1
    return events, facilities
