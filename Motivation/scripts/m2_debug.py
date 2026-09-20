import sys
sys.path.insert(0, "src")
from ocres.grid import World
from ocres.agents import CookAgent
from ocres.runner import run_episode
ROWS=["XXPXPXX","O     O","X 1 2 X","X D S X","XXXXXXX"]
w = World.make(grid_rows=ROWS, horizon=300)
alice = CookAgent(w.grid, me=0, parallel_after_delay=0)
partner = CookAgent(w.grid, me=1, parallel_after_delay=None)
logs, m = run_episode(w, [alice, partner], horizon=300)
print(m)
dels=[l["t"] for l in logs if l["r"]>0]
print("deliveries at:", dels)
st = dels[0] if dels else 0
for l in logs:
    if st <= l["t"] <= st+80:
        print(f"t={l['t']:3d} pots={l['pot']} i0={l['intent0']}@{str(l['target0']):>5} i1={l['intent1']}@{str(l['target1']):>5} p0={l['p0']} p1={l['p1']} h0={l['held0']} h1={l['held1']} a={l['a']} r={l['r']}")
