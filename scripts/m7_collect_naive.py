"""v2 naive collection: no-experience Alice & unaware Bob, save logs + meta."""
import json
import pathlib
import sys
import time

sys.path.insert(0, "src")
from collections import deque

import numpy as np

from ocres.grid import World
from ocres.llm_player import LLMPlayer
from ocres.llm import chat_json
from ocres.runner import run_episode

ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
HORIZON = 300
NEP = 4
OUT = pathlib.Path("data/v2_naive")
OUT.mkdir(parents=True, exist_ok=True)


def remote_chat(system, user):
    return chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=1600,
    )


for ep in range(10, 10 + NEP):
    board = deque(maxlen=6)
    w = World.make(grid_rows=ROWS, horizon=HORIZON)
    alice = LLMPlayer(w.grid, me=0, chat=remote_chat, name="Alice", experiences=(), bob_knows=None, board=board)
    bob = LLMPlayer(w.grid, me=1, chat=remote_chat, name="Bob", experiences=(), bob_knows=False, board=board)
    t0 = time.time()
    logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
    arr = {k: np.array([l[k] for l in logs], dtype=object) for k in logs[0].keys()}
    np.savez_compressed(OUT / f"ep{ep}.npz", **arr)
    with open(OUT / f"ep{ep}.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": m, "calls": (alice.call_count, bob.call_count), "wall_s": round(time.time() - t0)}, f, default=str)
    print(f"ep{ep}: del={m['deliveries']} reward={m['reward']} calls={alice.call_count},{bob.call_count} wall={time.time()-t0:.0f}s")
