"""M5 offline harness: run hybrid LLMCook with the FakeModel backend to
verify the decision-trigger loop, count calls, and record metrics.
The FakeModel returns the scripted FSM choices, so this should reproduce
the scripted baseline throughput while exercising the LLM plumbing.
"""
import pathlib
import sys

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.agents import CookAgent
from ocres.llm_agent import LLMCook, FakeModel
from ocres.runner import run_episode

ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
HORIZON = 900

w = World.make(grid_rows=ROWS, horizon=HORIZON)
ref_fsm = CookAgent(w.grid, me=0, parallel_after_delay=0)  # rule baseline used by Fake
alice = LLMCook(w.grid, me=0, model=FakeModel(ref_fsm), label="LLM-fake")
partner = CookAgent(w.grid, me=1, parallel_after_delay=None)

logs, m = run_episode(w, [alice, partner], horizon=HORIZON)
print("metrics:", {k: m[k] for k in ("deliveries", "reward", "mean_gap", "max_stay0", "max_stay1")})
print("LLM decision calls (agent0):", alice.call_count)
from collections import Counter  # noqa: E402

print("intent mix:", dict(Counter(l["intent0"] for l in logs)))
