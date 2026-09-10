"""Two-pot custom layout pilot: L0(serial) vs Lk(parallel) + flip mining."""
import sys

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.agents import CookAgent
from ocres.runner import run_episode

HORIZON = 900
ROWS = [
    "XXPXPXX",
    "O     O",
    "X 1 2 X",
    "X D S X",
    "XXXXXXX",
]


def run_one(serial):
    w = World.make(grid_rows=ROWS, horizon=HORIZON)
    alice = CookAgent(w.grid, me=0, parallel_after_delay=serial)
    partner = CookAgent(w.grid, me=1, parallel_after_delay=None)  # fixed moderate partner
    logs, m = run_episode(w, [alice, partner], horizon=HORIZON)
    return logs, m


res = {}
for name, serial in [("L0", True), ("Lk", False)]:
    logs, m = run_one(serial)
    res[name] = (logs, m)
    print(f"{name} (serial={serial}): deliveries={m['deliveries']} reward={m['reward']} mean_gap={m['mean_gap']} max_stay=({m['max_stay0']},{m['max_stay1']})")

logs0 = res["L0"][0]
logsk = res["Lk"][0]
n = min(len(logs0), len(logsk))


def obs(l):
    return (l["p0"], l["p1"], l["held0"], l["held1"], l["pot"], tuple(l["a"]))


flips = []
i = 0
while i < n:
    j = i
    while j < n and obs(logs0[j]) == obs(logsk[j]):
        j += 1
    if j - i >= 2 and j < n:
        li, lk = logs0[j], logsk[j]
        if (li["p0"], li["p1"], li["held0"], li["held1"], li["pot"]) == (
            lk["p0"],
            lk["p1"],
            lk["held0"],
            lk["held1"],
            lk["pot"],
        ) and li["intent0"] != lk["intent0"]:
            flips.append((i, j, li["t"], li["intent0"], li["target0"], lk["intent0"], lk["target0"], li["pot"]))
    i = max(j, i + 1)

print(f"\nmatched-prefix intent flips (L0 vs Lk): {len(flips)}")
for f in flips[:20]:
    print(f"  pre[{f[0]}-{f[1]}] t={f[2]} pot={f[7]}: L0={f[3]}@{f[4]} vs Lk={f[5]}@{f[6]}")
