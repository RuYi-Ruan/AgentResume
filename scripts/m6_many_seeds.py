"""Generate L0/Lk episodes for many seeds (fixed two-pot layout)."""
import json
import pathlib
import random
import sys

sys.path.insert(0, "src")
from ocres import data
from ocres.data import TWO_POT
from ocres.agents import CookAgent
from ocres.runner import run_episode
from ocres.grid import World

HORIZON = 700
SEEDS = list(range(1, 9))
OUT = pathlib.Path("data/twopot_v1")
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = {"L0": None, "Lk": 0}


def randomize_start(w, seed):
    rng = random.Random(1000 + seed)
    a, b = rng.sample(sorted(w.grid.passable), 2)
    s = w.env.state
    s.players[0].update_pos_and_or(a, (1, 0))
    s.players[1].update_pos_and_or(b, (1, 0))
    s.timestep = 0


def run_one(delay, seed):
    w = World.make(grid_rows=TWO_POT, horizon=HORIZON)
    randomize_start(w, seed)
    alice = CookAgent(w.grid, me=0, parallel_after_delay=delay)
    partner = CookAgent(w.grid, me=1, parallel_after_delay=None)
    logs, m = run_episode(w, [alice, partner], horizon=HORIZON)
    return logs, m, w.env.game_stats


for label, delay in LEVELS.items():
    for seed in SEEDS:
        logs, m, stats = run_one(delay, seed)
        data.persist_episode(OUT, label, seed, logs, m, stats, delay, HORIZON, "twopot_v1")
        print(f"{label} seed{seed}: del={m['deliveries']} gap={m['mean_gap']}")
with open(OUT / "seeds8_meta.json", "w") as f:
    json.dump({"levels": list(LEVELS), "seeds": SEEDS, "horizon": HORIZON}, f, indent=1)
