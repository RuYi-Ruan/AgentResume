"""Grid geometry, pathfinding and interaction primitives for cramped_room.

Coordinates are (x=col, y=row), origin top-left, matching overcooked_ai.
Static feature cells (pot/onion/dish/serve/counter) are NOT walkable;
agents stand on floor cells adjacent to them and use 'interact'.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

DIRS = [(0, -1), (0, 1), (1, 0), (-1, 0)]  # up, down, right, left
STAY = (0, 0)


@dataclass
class Grid:
    gw: OvercookedGridworld
    layout: str
    passable: set = field(init=False)
    pot_locs: list = field(init=False)
    onion_locs: list = field(init=False)
    dish_locs: list = field(init=False)
    serve_locs: list = field(init=False)

    def __post_init__(self):
        terrain = self.gw.terrain_mtx
        self.passable = set()
        for y, row in enumerate(terrain):
            for x, ch in enumerate(row):
                if ch in (" ", "1", "2"):
                    self.passable.add((x, y))
        self.pot_locs = list(self.gw.get_pot_locations())
        self.onion_locs = list(self.gw.get_onion_dispenser_locations())
        self.dish_locs = list(self.gw.get_dish_dispenser_locations())
        self.serve_locs = list(self.gw.get_serving_locations())

    def bfs(self, start, goal, occupied=()):
        """Shortest path of floor cells start->goal avoiding occupied; None if blocked."""
        if start == goal:
            return []
        occ = set(occupied) - {start, goal}
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for d in DIRS:
                nxt = (cur[0] + d[0], cur[1] + d[1])
                if nxt in self.passable and nxt not in occ and nxt not in prev:
                    prev[nxt] = cur
                    if nxt == goal:
                        path = []
                        node = goal
                        while node != start:
                            path.append(node)
                            node = prev[node]
                        return path[::-1]
                    q.append(nxt)
        return None

    def step_toward(self, start, goal, occupied=()):
        """First floor cell to move into on the way to goal (None if at goal/blocked)."""
        path = self.bfs(start, goal, occupied)
        return path[0] if path else None

    def interact_stand_cells(self, target):
        """Floor stand cells adjacent to a static feature cell, with facing dir."""
        out = []
        for d in DIRS:
            stand = (target[0] + d[0], target[1] + d[1])
            if stand in self.passable:
                out.append((stand, tuple(-c for c in d)))  # face back toward target
        return out

    def orientation_toward(self, stand, target):
        return (target[0] - stand[0], target[1] - stand[1])


@dataclass
class World:
    """Shared env + grid access for a rollout session."""

    env: object
    grid: Grid
    horizon: int = 400

    @classmethod
    def make(cls, layout="cramped_room", horizon=400, start_state_fn=None, grid_rows=None):
        if grid_rows is None:
            gw = OvercookedGridworld.from_layout_name(layout)
            name = layout
        else:
            gw = OvercookedGridworld.from_grid(
                [list(r) for r in grid_rows], params_to_overwrite={"layout_name": "custom_twopot"}
            )
            name = "custom_twopot"
        envmod = __import__("overcooked_ai_py.mdp.overcooked_env", fromlist=["OvercookedEnv"])
        env = envmod.OvercookedEnv.from_mdp(gw, start_state_fn=start_state_fn, horizon=horizon)
        return cls(env=env, grid=Grid(gw=gw, layout=name), horizon=horizon)
