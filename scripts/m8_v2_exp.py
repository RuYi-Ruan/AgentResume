"""v2 experiment (motor-scaffold LLM): Alice E vs no-E x Bob unaware/aware.

Configs:
  before      : Alice novice (no E)          , Bob unaware
  after_unaware: Alice expert (E injected)   , Bob unaware (doesn't know E)
  after_aware : Alice expert (E injected)    , Bob aware (holds same E, told
                to simulate Alice-with-E before guessing)
Metrics: Alice steps, task score (deliveries*20), Bob alice_guess accuracy
(epoch window) overall + ambiguous subset (pot cooking & another accepts &
Alice actually chose FETCH -> parallel-fill).
"""
import json
import sys
import time

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

HORIZON = 260
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]

# E text shown to Alice (expert) and, for the aware Bob, handed over.
E_TEXT = cards.ALICE_LK
NOVICE_TEXT = cards.ALICE_L0
UNAWARE_BOB = cards.BOB_STALE
AWARE_BOB = (
    "你对伙伴 Alice 的印象（重要：Alice 最近被注入了以下经验，能力可能因此提升，请据此判断她的意图）：\n"
    + E_TEXT
    + "\n猜她意图时请模拟：如果我是带着这些经验的 Alice，面对此刻局面我会选什么。"
)

CONFIGS = {
    "before": dict(alice_card=NOVICE_TEXT, bob_card=UNAWARE_BOB, bob_aware=False),
    "after_unaware": dict(alice_card=E_TEXT, bob_card=UNAWARE_BOB, bob_aware=False),
    "after_aware": dict(alice_card=E_TEXT, bob_card=AWARE_BOB, bob_aware=True),
}

results = []
for cond, cfg in CONFIGS.items():
    for rep in range(REPS):
        model = RemoteModel()
        w = World.make(grid_rows=ROWS, horizon=HORIZON)
        alice = LLMCook(w.grid, me=0, model=model, role_card=cfg["alice_card"], role_fixed="cook")
        bob = LLMBob(w.grid, me=1, model=model, role_card=cfg["bob_card"])
        t0 = time.time()
        logs, m = run_episode(w, [alice, bob], horizon=HORIZON)

        # Bob guess epochs (fresh guess rows only)
        guess_at = {}
        for l in logs:
            g = (l.get("info1") or {}).get("bob_guess")
            if g is not None:
                guess_at[l["t"]] = g
        order = sorted(guess_at)
        total = exact = win = 0
        for i, t in enumerate(order):
            g = guess_at[t]
            end = order[i + 1] if i + 1 < len(order) else HORIZON
            total += 1
            if any(x["intent0"] == g for x in logs[t:end]):
                win += 1
        # ambiguous subset: pot cooking, another accepts, Alice picked FETCH
        amb = [l for l in logs if "cooking" in l["pot"] and l["intent0"] == "FETCH"]
        amb_hit = sum(1 for t, g in guess_at.items() if g == "FETCH" and any(t <= q < (order[order.index(t) + 1] if order.index(t) + 1 < len(order) else HORIZON) for q in [x["t"] for x in amb])) if amb else 0
        amb_n = len([1 for t in guess_at if any(t <= x["t"] < (order[order.index(t) + 1] if order.index(t) + 1 < len(order) else HORIZON) for x in amb)])
        score = m["deliveries"] * 20
        rows = {
            "cond": cond, "rep": rep, "steps": m["steps"], "deliveries": m["deliveries"],
            "score": score, "gap": m["mean_gap"], "acc_window": win / total if total else None,
            "calls": (alice.call_count, bob.call_count), "wall_s": round(time.time() - t0),
        }
        results.append(rows)
        print(f"{cond} rep{rep}: del={m['deliveries']} score={score} acc_w={win / total if total else None:.2%} steps={m['steps']} wall={rows['wall_s']}s")

with open("data/v2_results.json", "w") as f:
    json.dump(results, f, indent=1)
for cond in CONFIGS:
    rs = [r for r in results if r["cond"] == cond]
    print(cond, "mean_score=", sum(r["score"] for r in rs) / len(rs),
          "mean_acc=", round(sum(r["acc_window"] for r in rs) / len(rs), 3) if all(r["acc_window"] is not None for r in rs) else None)
