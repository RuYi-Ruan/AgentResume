"""Shared recipe/state helpers for cramped_room onion soup.

Mechanics (probe-verified):
  fetch onion from an onion dispenser (O) -> place into pot (3x, P)
  interact at pot with empty hands starts cooking (20 ticks)
  when soup is_ready, an agent holding a dish can pick it up ('soup')
  interact at the serving location delivers (+20 sparse reward)
"""
from __future__ import annotations

from ocres.grid import Grid, World

POT, DISH, SERVE = "pot", "dish", "serve"


def pot_pos(grid: Grid):
    return grid.pot_locs[0]


def dish_pos(grid: Grid):
    return grid.dish_locs[0]


def serve_pos(grid: Grid):
    return grid.serve_locs[0]


def soup_at(state, grid: Grid):
    p = pot_pos(grid)
    return state.get_object(p) if state.has_object(p) else None


def pot_state_kind(state, grid: Grid):
    """empty | items(k) | cooking | ready"""
    soup = soup_at(state, grid)
    if soup is None:
        return "empty"
    if soup.is_ready:
        return "ready"
    if soup.is_cooking:
        return "cooking"
    return f"items{len(soup.ingredients)}"


def pot_kinds(state, grid: Grid):
    """Map each pot cell -> its state kind."""
    out = {}
    for p in grid.pot_locs:
        soup = state.get_object(p) if state.has_object(p) else None
        if soup is None:
            out[p] = "empty"
        elif soup.is_ready:
            out[p] = "ready"
        elif soup.is_cooking:
            out[p] = "cooking"
        else:
            out[p] = f"items{len(soup.ingredients)}"
    return out


def soup_cook_remaining(state, grid: Grid, pot):
    s = state.get_object(pot) if state.has_object(pot) else None
    if s is None or not s.is_cooking:
        return None
    try:
        return s.cook_time_remaining
    except Exception:  # noqa: BLE001
        return None


def cook_remaining(state, grid: Grid):
    s = soup_at(state, grid)
    if s is None or not s.is_cooking:
        return None
    try:
        return s.cook_time_remaining
    except Exception:  # noqa: BLE001
        return None


def agent_held(state, i: int):
    o = state.players[i].held_object
    return o.name if o else None


def onion_ready_for_place(state, grid: Grid):
    """Can agent (any) currently add an onion to the pot?"""
    soup = soup_at(state, grid)
    return soup is None or (not soup.is_cooking and not soup.is_ready and len(soup.ingredients) < 3)
