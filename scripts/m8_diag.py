"""Diagnose why aware-Bob score drops: 1 rep each of unaware/aware with
Bob intent mix, parking at choke (2,2), and delivery timing."""
import sys
from collections import Counter

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

HORIZON = 260
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
E_TEXT = cards.ALICE_LK
UNAWARE = cards.BOB_STALE
AWARE = (
    "你对伙伴 Alice 的印象（重要：Alice 最近被注入了以下经验，能力可能因此提升，请据此判断她的意图）：\n"
    + E_TEXT
    + "\n猜她意图时请模拟：如果我是带着这些经验的 Alice，面对此刻局面我会选什么。"
)

for cond, bob_card in [("unaware", UNAWARE), ("aware", AWARE)]:
    model = RemoteModel()
    w = World.make(grid_rows=ROWS, horizon=HORIZON)
    alice = LLMCook(w.grid, me=0, model=model, role_card=E_TEXT, role_fixed="cook")
    bob = LLMBob(w.grid, me=1, model=model, role_card=bob_card)
    logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
    bob_ints = Counter(l["intent1"] for l in logs)
    alice_ints = Counter(l["intent0"] for l in logs)
    at_choke = Counter(str(l["p1"]) == "(2, 2)" for l in logs)[True]
    dish_wait = sum(1 for l in logs if l["held1"] == "dish" and l["held0"] == "onion")
    dels = [l["t"] for l in logs if l["r"] > 0]
    print(f"== {cond}: del={m['deliveries']} score={m['deliveries']*20} gap={m['mean_gap']} dels_t={dels}")
    print("   bob intent:", dict(bob_ints))
    print("   alice intent:", dict(alice_ints))
    print(f"   bob at choke(2,2) ticks: {at_choke}; bob-holds-dish while alice-holds-onion ticks: {dish_wait}")
