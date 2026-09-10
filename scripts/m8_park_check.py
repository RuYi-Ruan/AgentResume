"""Re-verify aware vs unaware scores with PRE parked off-choke (2 reps each)."""
import sys
from collections import Counter

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

HORIZON = 260
REPS = 2
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
E_TEXT = cards.ALICE_LK
UNAWARE = cards.BOB_STALE
AWARE = (
    "你对伙伴 Alice 的印象（重要：Alice 最近被注入了以下经验，能力可能因此提升，请据此判断她的意图）：\n"
    + E_TEXT
    + "\n猜她意图时请模拟：如果我是带着这些经验的 Alice，面对此刻局面我会选什么。"
)

for cond, bob_card in [("unaware", UNAWARE), ("aware", AWARE)]:
    rows = []
    for rep in range(REPS):
        model = RemoteModel()
        w = World.make(grid_rows=ROWS, horizon=HORIZON)
        alice = LLMCook(w.grid, me=0, model=model, role_card=E_TEXT, role_fixed="cook")
        bob = LLMBob(w.grid, me=1, model=model, role_card=bob_card)
        logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
        rows.append(m)
        at_choke = sum(1 for l in logs if str(l["p1"]) == "(2, 2)")
        print(f"{cond} rep{rep}: del={m['deliveries']} score={m['deliveries']*20} gap={m['mean_gap']} bob_choke={at_choke}")
    print(f"  -> {cond} mean_score={sum(r['deliveries'] for r in rows)*20/len(rows):.0f}")
