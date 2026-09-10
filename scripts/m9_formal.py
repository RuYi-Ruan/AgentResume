"""v2 formal run (fixed motor): before (no E) vs after (E injected) x Bob
unaware vs aware; Alice experience E mined from naive trajectories."""
import json
import sys
import time

sys.path.insert(0, "src")
from collections import deque

from ocres.grid import World
from ocres.llm_player import LLMPlayer
from ocres.llm import chat_json
from ocres.metrics import score_intent_events
from ocres.runner import run_episode

HORIZON = 300
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]
E = json.load(open("data/v2_E.json", encoding="utf-8"))


def remote_chat(system, user):
    return chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=1600,
    )


def run_cond(label, alice_exp, bob_knows, reps):
    rows = []
    for rep in range(reps):
        board = deque(maxlen=6)
        w = World.make(grid_rows=ROWS, horizon=HORIZON)
        alice = LLMPlayer(w.grid, me=0, chat=remote_chat, name="Alice", experiences=alice_exp, bob_knows=None, board=board, retrieve_k=2)
        bob_exp = E if bob_knows else ()
        bob = LLMPlayer(w.grid, me=1, chat=remote_chat, name="Bob", experiences=bob_exp, bob_knows=bool(bob_knows), board=board)
        t0 = time.time()
        logs, m = run_episode(w, [alice, bob], horizon=HORIZON)
        intent_m = score_intent_events(logs)
        r = {"cond": label, "rep": rep, "deliveries": m["deliveries"], "score": m["deliveries"] * 20,
             "acc": intent_m["accuracy"], "alice_decisions": intent_m["alice_decisions"],
             "guessed_decisions": intent_m["guessed_decisions"], "cause_events": intent_m["cause_events"],
             "cause_acc": intent_m["cause_accuracy"],
             "calls": (alice.call_count, bob.call_count), "wall_s": round(time.time() - t0)}
        rows.append(r)
        accs = f"{r['acc']:.2%}" if r["acc"] is not None else "NA"
        cacc = f"cause_acc={r['cause_acc']:.2%}" if r["cause_acc"] is not None else "cause_acc=NA"
        print(f"{label} rep{rep}: del={m['deliveries']} score={r['score']} acc={accs} "
              f"decisions={r['alice_decisions']} guessed={r['guessed_decisions']} "
              f"cause_events={r['cause_events']} {cacc} wall={r['wall_s']}s")
    return rows


results = []
for label, exp, know in [("before", (), False), ("after_unaware", E, False), ("after_aware", E, True)]:
    results += run_cond(label, exp, know, REPS)
json.dump(results, open("data/v2_results_formal.json", "w"), indent=1)
print("\npooled:")
for cond in ["before", "after_unaware", "after_aware"]:
    rs = [r for r in results if r["cond"] == cond]
    accs = [r["acc"] for r in rs if r["acc"] is not None]
    print(f"{cond}: n={len(rs)} del={[r['deliveries'] for r in rs]} score_mean={sum(r['score'] for r in rs) / len(rs):.1f} acc_mean={sum(accs) / len(accs):.3f}")
