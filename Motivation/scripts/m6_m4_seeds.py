"""M4-style whole-episode comparison on 8 seeds (deterministic replicates).

Stale = book from all L0 episodes; Updated = + other Lk seeds' first halves;
Oracle = other Lk seeds. Accuracy on the held-out Lk seed (whole episode),
plus level-divergence ticks (same observable context, different true intent).
"""
import pathlib
import sys

sys.path.insert(0, "src")
from ocres import data
from ocres.bob import Book, accuracy, ctx_of, level_divergence_ticks

OUT = pathlib.Path("data/twopot_v1")
SEEDS = list(range(1, 9))
HALF = 350

L0 = {s: data.load_episode(OUT, "L0", s)[0] for s in SEEDS}
LK = {s: data.load_episode(OUT, "Lk", s)[0] for s in SEEDS}

stale = Book()
for s in SEEDS:
    stale.add_episode(L0[s])

div_pool = {c: {"agree": 0, "seen": 0} for c in ["Stale", "Updated", "Oracle"]}
rows = []
for ts in SEEDS:
    test = LK[ts]
    other = [s for s in SEEDS if s != ts]
    oracle = Book()
    for s in other:
        oracle.add_episode(LK[s])
    updated = Book()
    for s in SEEDS:
        updated.add_episode(L0[s])
    for s in other:
        updated.add_episode(LK[s][:HALF])
    div = level_divergence_ticks(L0[ts], LK[ts], 0, len(test))
    for cond, book in [("Stale", stale), ("Updated", updated), ("Oracle", oracle)]:
        acc, n = accuracy(book, test, window=(0, None))
        agree = seen = 0
        for t in div:
            c = ctx_of(test[t])
            p = book.predict(c)
            if p is None:
                continue
            seen += 1
            agree += int(p == test[t]["intent0"])
        div_pool[cond]["agree"] += agree
        div_pool[cond]["seen"] += seen
        rows.append((ts, cond, acc, n, agree, seen))
        print(f"seed{ts:02d} {cond:>7}: acc={acc:.3f} n={n} div={agree}/{seen}")

print("\nmean accuracy (over test seeds):")
for cond in ["Stale", "Updated", "Oracle"]:
    accs = [r[2] for r in rows if r[1] == cond and r[2] is not None]
    print(f"  {cond}: {sum(accs) / len(accs):.3f}")
print("pooled divergence accuracy:")
for cond in ["Stale", "Updated", "Oracle"]:
    p = div_pool[cond]
    print(f"  {cond}: {p['agree']}/{p['seen']} = {p['agree'] / p['seen'] if p['seen'] else None:.3f}")
