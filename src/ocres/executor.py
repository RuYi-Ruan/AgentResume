"""Deterministic low-level executor: converts a target interaction into the
next atomic action for one agent, planning on floor cells and facing rules.

Agent actions are (dx, dy) moves, (0,0)=stay, or 'interact'.
A target is a static feature cell (pot/onion/dish/serve); the agent must stand
on a specific floor cell facing the feature before 'interact' has an effect.
"""
from __future__ import annotations

from ocres.grid import Grid, STAY

# Interaction anchors: feature cell -> (stand cell, facing dir). Verified by probe.
ANCHORS = {
    # (feature, stand, face)
}


def anchors_for(grid: Grid, target):
    """All (stand, face) pairs for interacting with `target`."""
    return grid.interact_stand_cells(target)


def feature_anchor(grid: Grid, target):
    """Preferred single anchor: pick stand whose 'approach' cell is floor; else first."""
    cands = anchors_for(grid, target)
    for stand, face in cands:
        app = (stand[0] - face[0], stand[1] - face[1])
        if app in grid.passable:
            return stand, face, app
    stand, face = cands[0]
    return stand, face, None


class Executor:
    """Per-tick resolver for one agent pursuing interaction or movement goals."""

    def __init__(self, grid: Grid, me: int):
        self.grid = grid
        self.me = me

    def other(self, state):
        return state.players[1 - self.me].position

    # -- pure movement ------------------------------------------------
    def move_action(self, state, goal_cell):
        """Return action to step toward goal_cell (None if already there)."""
        me = state.players[self.me].position
        if me == goal_cell:
            return None
        nxt = self.grid.step_toward(me, goal_cell, occupied=[self.other(state)])
        if nxt is None:
            return None
        return (nxt[0] - me[0], nxt[1] - me[1])

    # -- goal: stand at a specific cell (no interaction) --------------
    def stand_action(self, state, stand_cell):
        """Action to approach/stand on stand_cell; returns None when standing."""
        me = state.players[self.me]
        if me.position == stand_cell:
            return None
        return self.move_action(state, stand_cell)

    # -- goal: interact with a feature cell ----------------------------
    def interact_action(self, state, target):
        """Next action on the way to interacting with `target`; None if waiting."""
        stand, face, app = feature_anchor(self.grid, target)
        me = state.players[self.me]
        pos, ori = me.position, tuple(me.orientation)
        other_pos = self.other(state)
        if pos == stand and ori == face:
            return "interact"
        # route through approach cell so the final step sets the facing
        if app is not None:
            if pos == app:
                # at approach: step into stand once it is free
                return (stand[0] - pos[0], stand[1] - pos[1]) if stand != other_pos else None
            if app == other_pos:
                return None  # approach occupied -> wait, do not bounce
            a = self.move_action(state, app)
            if a is not None:
                return a
            return None
        # fallback: go to stand then rotate via approach cell when it frees
        if pos != stand:
            a = self.move_action(state, stand)
            if a is not None:
                return a
        return None  # waiting for space

    # -- goal: idle stay ------------------------------------------------
    @staticmethod
    def stay():
        return STAY
