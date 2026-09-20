import sys
sys.path.insert(0, "src")
from ocres.grid import World
ROWS = [
    "XXPXPXX",   # y0: two pots at x2,x4 split by wall x3
    "O     O",   # y1: onions at both far corners
    "X 1 2 X",   # y2: players
    "X D S X",   # y3: dish x2, serve x4
    "XXXXXXX",   # y4
]
w = World.make(grid_rows=ROWS, horizon=100)
g = w.grid
s = w.env.state
print("passable:", sorted(g.passable))
for name, locs in [("pot", g.pot_locs), ("onion", g.onion_locs), ("dish", g.dish_locs), ("serve", g.serve_locs)]:
    for t in locs:
        print(f"{name}{t} stands:", g.interact_stand_cells(t))
print("players:", s.players_pos_and_or)
print("orders:", [tuple(o.ingredients) for o in s._all_orders])
# connectivity quick: one BFS through everything
left = min(g.passable); right = max(g.passable)
print("bfs across room:", g.bfs(left, right) is not None, len(g.bfs(left, right) or []))
