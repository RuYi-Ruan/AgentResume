"""v2 naive baseline: both LLM players, no experiences, cooperative only via
the intent board and 3x3 senses. Goal: at least one delivery (pipeline OK)."""
import sys
import time

sys.path.insert(0, "src")
from collections import deque

from ocres.grid import World
from ocres.llm_player import LLMPlayer
from ocres.llm import chat_json
from ocres.runner import run_episode

ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
HORIZON = 320


def remote_chat(system, user):
    return chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=1600,
    )


board = deque(maxlen=6)
w = World.make(grid_rows=ROWS, horizon=HORIZON)
alice = LLMPlayer(w.grid, me=0, chat=remote_chat, name="Alice", experiences=(), bob_knows=None, board=board)
bob = LLMPlayer(w.grid, me=1, chat=remote_chat, name="Bob", experiences=(), bob_knows=None, board=board)

t0 = time.time()
logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
print(f"metrics: deliveries={m['deliveries']} reward={m['reward']} mean_gap={m['mean_gap']} max_stay=({m['max_stay0']},{m['max_stay1']})")
print(f"calls: alice={alice.call_count} bob={bob.call_count} wall_s={time.time() - t0:.0f}")
if m["deliveries"] == 0:
    for l in logs[:45]:
        print(
            f"t={l['t']:3d} i0={l['intent0']}@{str(l['target0']):>7} i1={l['intent1']}@{str(l['target1']):>7} "
            f"p0={l['p0']} p1={l['p1']} h0={l['held0']} h1={l['held1']} g={l.get('info1') and (l['info1'] or {}).get('guess')}"
        )
