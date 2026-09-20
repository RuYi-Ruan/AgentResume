"""M4 Bob experiment v2 -- leave-one-seed-out, whole-episode test.

For each held-out Lk seed (test, whole episode):
  Stale   = book from all L0 episodes
  Oracle  = book from other Lk episodes (whole)
  Updated = book from L0 + other Lk episodes' first halves
Report overall accuracy and accuracy on level-divergence ticks (identical
observable context, different true intent) pooled over test seeds.
"""
import pathlib
import sys

sys.path.insert(0, "src")
from ocres import data
from ocres.bob import Book, accuracy, ctx_of, level_divergence_ticks

OUT = pathlib.Path("data/twopot_v1")
SEEDS = [1, 2, 3]

L0 = {s: data.load_episode(OUT, "L0", s)[0] for s in SEEDS}
LK = {s: data.load_episode(OUT, "Lk", s)[0] for s in SEEDS}

stale_book = Book()
for s in SEEDS:
    stale_book.add_episode(L0[s])

pool = {cond: {"agree": 0, "seen": 0} for cond in ["Stale", "Updated", "Oracle"]}
rows = []
for test_seed in SEEDS:
    test_logs = LK[test_seed]
    other = [s for s in SEEDS if s != test_seed]
    oracle = Book()
    for s in other:
        oracle.add_episode(LK[s])
    updated = Book()
    for s in SEEDS:
        updated.add_episode(L0[s])
    for s in other:
        updated.add_episode(LK[s][:450])

    div = level_divergence_ticks(L0[test_seed], LK[test_seed], 0, len(test_logs))
    for cond, book in [("Stale", stale_book), ("Updated", updated), ("Oracle", oracle)]:
        acc, n = accuracy(book, test_logs, window=(0, None))
        agree = seen = 0
        for t in div:
            c = ctx_of(test_logs[t])
            pred = book.predict(c)
            if pred is None:
                continue
            seen += 1
            agree += int(pred == test_logs[t]["intent0"])
        p = pool[cond]
        p["agree"] += agree
        p["seen"] += seen
        rows.append((test_seed, cond, acc, n, agree, seen))
        print(f"seed{test_seed} {cond:>7}: acc={acc:.3f} (n={n})  div={agree}/{seen}")

print("\npooled divergence accuracy:")
for cond in ["Stale", "Updated", "Oracle"]:
    p = pool[cond]
    print(f"  {cond}: {p['agree']}/{p['seen']} = {p['agree'] / p['seen'] if p['seen'] else None:.3f}")
