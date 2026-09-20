"""Extend v2 results: 2 more reps for after_unaware & after_aware, append,
then print pooled stats over all reps."""
import json
import sys
import time

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

HORIZON = 260
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
E_TEXT = cards.ALICE_LK
UNAWARE_BOB = cards.BOB_STALE
AWARE_BOB = (
    "你对伙伴 Alice 的印象（重要：Alice 最近被注入了以下经验，能力可能因此提升，请据此判断她的意图）：\n"
    + E_TEXT
    + "\n猜她意图时请模拟：如果我是带着这些经验的 Alice，面对此刻局面我会选什么。"
)

res_path = "data/v2_results.json"
results = json.load(open(res_path, encoding="utf-8"))
CONFIGS = {
    "after_unaware": UNAWARE_BOB,
    "after_aware": AWARE_BOB,
}
for cond, bob_card in CONFIGS.items():
    for rep in range(2):
        model = RemoteModel()
        w = World.make(grid_rows=ROWS, horizon=HORIZON)
        alice = LLMCook(w.grid, me=0, model=model, role_card=E_TEXT, role_fixed="cook")
        bob = LLMBob(w.grid, me=1, model=model, role_card=bob_card)
        t0 = time.time()
        logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
        guess_at = {}
        for l in logs:
            g = (l.get("info1") or {}).get("bob_guess")
            if g is not None:
                guess_at[l["t"]] = g
        order = sorted(guess_at)
        total = win = 0
        for i, t in enumerate(order):
            g = guess_at[t]
            end = order[i + 1] if i + 1 < len(order) else HORIZON
            total += 1
            if any(x["intent0"] == g for x in logs[t:end]):
                win += 1
        results.append({
            "cond": cond, "rep": f"ext{rep}", "steps": m["steps"],
            "deliveries": m["deliveries"], "score": m["deliveries"] * 20,
            "gap": m["mean_gap"], "acc_window": win / total if total else None,
            "calls": (alice.call_count, bob.call_count), "wall_s": round(time.time() - t0),
        })
        print(f"{cond} ext{rep}: del={m['deliveries']} score={m['deliveries']*20} acc_w={win/total if total else None:.2%}")

json.dump(results, open(res_path, "w"), indent=1)
print("\npooled:")
for cond in ["after_unaware", "after_aware"]:
    rs = [r for r in results if r["cond"] == cond]
    accs = [r["acc_window"] for r in rs if r["acc_window"] is not None]
    scores = [r["score"] for r in rs]
    dels = [r["deliveries"] for r in rs]
    print(f"{cond}: n={len(rs)} deliveries={dels} score_mean={sum(scores)/len(scores):.0f} acc_mean={sum(accs)/len(accs):.3f} acc_range=[{min(accs):.3f},{max(accs):.3f}]")
