"""M5 first live pilot: Alice=LLM cook (Lk experience), Bob=LLM serve with an
impression card (stale|current). Reports Bob's Alice-intent guess accuracy,
LLM call counts and team metrics."""
import sys
import time

sys.path.insert(0, "src")
from ocres.grid import World
from ocres.llm_agent import LLMCook, LLMBob, RemoteModel
from ocres.runner import run_episode
from ocres import cards

FLAVOR = sys.argv[1] if len(sys.argv) > 1 else "stale"
HORIZON = 260
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]

alice_card = cards.ALICE_LK
bob_card = cards.BOB_STALE if FLAVOR == "stale" else cards.BOB_CURRENT

model = RemoteModel()
w = World.make(grid_rows=ROWS, horizon=HORIZON)
alice = LLMCook(w.grid, me=0, model=model, role_card=alice_card, label="Alice", role_fixed="cook")
bob = LLMBob(w.grid, me=1, model=model, role_card=bob_card, label="Bob", )
t0 = time.time()
logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
dt = time.time() - t0

# ---- guess accuracy: compare Bob's alice_guess vs Alice logged intent0 ----
hits_t = hits_w = total = 0
misses = []
for l in logs:
    g = (l.get("info1") or {}).get("bob_guess")
    if not g:
        continue
    total += 1
    truth = l["intent0"]
    if g == truth:
        hits_t += 1
    # soft: Alice realised the guessed intent within the next 10 ticks
    if any(truth == g for truth in [x["intent0"] for x in logs[l["t"]: l["t"] + 11]]):
        hits_w += 1
    if g != truth:
        misses.append((l["t"], g, truth, l["pot"]))

print(f"\n== pilot flavor={FLAVOR} ==")
print(f"metrics: deliveries={m['deliveries']} reward={m['reward']} mean_gap={m['mean_gap']} max_stay=({m['max_stay0']},{m['max_stay1']})")
print(f"calls: alice={alice.call_count} bob={bob.call_count} wall_s={dt:.0f}")
if m["deliveries"] == 0:
    print("--- zero-delivery trace (first 45 ticks) ---")
    for l in logs[:45]:
        print(
            f"t={l['t']:3d} i0={l['intent0']:>10}@{str(l['target0']):>6} i1={l['intent1']:>10} "
            f"p0={l['p0']} p1={l['p1']} h0={l['held0']} h1={l['held1']} a0={str(l['a'][0]):>8} pots={l['pot']}"
        )
print(f"guess acc (exact t): {hits_t}/{total} = {hits_t / total if total else None:.2%}")
print(f"guess acc (<=10 ticks): {hits_w}/{total} = {hits_w / total if total else None:.2%}")
print("miss examples (t, bob_guess, alice_truth, pots):")
for t, g, truth, pots in misses[:6]:
    print("   ", t, g, "vs", truth, pots)
