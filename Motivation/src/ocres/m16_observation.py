"""Observation boundary for M16's partially observable LLM Bob."""
from __future__ import annotations


VIEW_RADIUS = 1  # a 3x3 egocentric view


def chebyshev_distance(first, second):
    return max(abs(first[0] - second[0]), abs(first[1] - second[1]))


def alice_visible_before_action(state, bob_index=1, radius=VIEW_RADIUS):
    """Whether Bob could actually witness Alice starting the action.

    Visibility is checked before the environment step.  Entering Bob's view
    afterwards is not enough to reveal Alice's previous held item or full
    first action, so such an event is intentionally not queried or scored.
    """

    alice_index = 1 - bob_index
    alice = tuple(state.players[alice_index].position)
    bob = tuple(state.players[bob_index].position)
    return chebyshev_distance(alice, bob) <= radius


def should_query_bob(state, intent_changed, bob_index=1, radius=VIEW_RADIUS):
    """No intent-change notification is exposed when Alice is out of view."""

    return bool(intent_changed and alice_visible_before_action(state, bob_index, radius))

