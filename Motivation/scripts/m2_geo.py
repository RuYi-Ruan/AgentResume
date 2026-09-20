import sys
sys.path.insert(0, "src")
from ocres.grid import Grid, World
w = World.make(layout="forced_coordination", horizon=50)
g = w.grid
print("passable sorted:", sorted(g.passable))
for name, locs in [("pot", g.pot_locs), ("onion", g.onion_locs), ("dish", g.dish_locs), ("serve", g.serve_locs)]:
    for t in locs:
        print(f"{name}{t} stands:", [(s, f) for s, f in g.interact_stand_cells(t)])
print("terrain rows:")
for row in g.gw.terrain_mtx:
    print("  ", "".join(str(c) for c in row))
