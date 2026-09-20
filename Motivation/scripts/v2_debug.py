"""v2 debug: short run, dump trace + first prompts for inspection."""
import sys

sys.path.insert(0, "src")
from collections import deque

from ocres.grid import World
from ocres.llm_player import LLMPlayer
from ocres.llm import chat_json
from ocres.runner import run_episode

ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
HORIZON = 180

calls = {"n": 0}


def remote_chat(system, user):
    calls["n"] += 1
    if calls["n"] <= 6:
        with open("data/v2_prompts.txt", "a", encoding="utf-8") as f:
            f.write(f"\n===== CALL {calls['n']} =====\n[SYS]\n{system}\n[USER]\n{user}\n")
    return chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=1600,
    )


board = deque(maxlen=6)
w = World.make(grid_rows=ROWS, horizon=HORIZON)
alice = LLMPlayer(w.grid, me=0, chat=remote_chat, name="Alice", board=board)
bob = LLMPlayer(w.grid, me=1, chat=remote_chat, name="Bob", board=board)
logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
print("metrics:", m)
print("calls:", alice.call_count, bob.call_count)
for l in logs[:55]:
    print(
        f"t={l['t']:3d} i0={l['intent0']}@{str(l['target0']):>6} i1={l['intent1']}@{str(l['target1']):>6} "
        f"p0={l['p0']} p1={l['p1']} h0={l['held0']} h1={l['held1']}"
    )
