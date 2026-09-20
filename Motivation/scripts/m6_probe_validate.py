"""M6 offline probe validation: amplify the reinterpretation set F and test
stale-impression Bob against it.

F = contexts (alice cell, held, pots) that occur in the ambiguous class
    (a pot cooking AND another pot accepting onions AND Alice empty-handed)
    where the two levels' intents differ.
Measures:
  - |F| and confusion (L0 intent -> Lk intent)
  - Bob accuracy on F: stale book (L0-only) vs oracle book (Lk), bootstrapped
"""
import pathlib
import sys
from collections import Counter

sys.path.insert(0, "src")
from ocres import data
from ocres.bob import Book, ctx_of

OUT = pathlib.Path("data/twopot_v1")
SEEDS = [1, 2, 3]


def ambiguous_tick(l):
    pots = l["pot"]
    return ("cooking" in pots) and any(k in pots for k in ("'empty'", "'items1'", "'items2'")) and l["held0"] in (None, "None")


def intent_at(level_logs, ctx):
    cnt = Counter(l["intent0"] for l in level_logs if ctx_of(l) == ctx)
    if not cnt:
        return None, 0
    g, n = cnt.most_common(1)[0]
    return g, n


L0 = {s: data.load_episode(OUT, "L0", s)[0] for s in SEEDS}
LK = {s: data.load_episode(OUT, "Lk", s)[0] for s in SEEDS}
l0_all = [l for s in SEEDS for l in L0[s]]
lk_all = [l for s in SEEDS for l in LK[s]]

# ---- ambiguous contexts seen by the upgraded (Lk) Alice ---------------
amb_ctx = {}
for l in lk_all:
    if ambiguous_tick(l):
        c = ctx_of(l)
        amb_ctx.setdefault(c, []).append(l["intent0"])

flips = []
for c, intents in amb_ctx.items():
    i0, n0 = intent_at(l0_all, c)
    ik, nk = Counter(intents).most_common(1)[0]
    if i0 is not None and i0 != ik and n0 >= 1 and nk >= 2:
        flips.append((c, i0, ik, n0, nk))

print(f"ambiguous contexts (Lk): {len(amb_ctx)}")
print(f"reinterpretation set |F| (i_L0 != i_Lk): {len(flips)}")
from collections import defaultdict  # noqa: E402

conf = defaultdict(int)
for c, i0, ik, *_ in flips:
    conf[(i0, ik)] += 1
print("confusion L0->Lk:", dict(conf))
for c, i0, ik, n0, nk in flips[:6]:
    print(f"  ctx={c}: L0={i0}({n0}) vs Lk={ik}({n_k := nk})")

# ---- Bob books & accuracy on F ---------------------------------------
stale = Book()
for s in SEEDS:
    stale.add_episode(L0[s])
oracle = Book()
for s in SEEDS:
    oracle.add_episode(LK[s])


def acc_on(book, pairs):
    hit = tot = 0
    for c, i0, ik, *_ in pairs:
        p = book.predict(c)
        if p is None:
            continue
        tot += 1
        hit += int(p == ik)
    return hit, tot


for name, book in [("stale(L0-only)", stale), ("oracle(Lk)", oracle)]:
    h, t = acc_on(book, flips)
    print(f"Bob {name}: on |F| guesses Lk-truth {h}/{t} = {h / t if t else None:.2%}")

# bootstrap CI over contexts for oracle-stale difference
import random  # noqa: E402

rng = random.Random(42)
diffs = []
for _ in range(2000):
    sample = [flips[rng.randrange(len(flips))] for _ in range(len(flips))]
    h0, t0 = acc_on(stale, sample)
    h1, t1 = acc_on(oracle, sample)
    d = (h1 / t1 if t1 else 0) - (h0 / t0 if t0 else 0)
    diffs.append(d)
diffs.sort()
print(f"oracle-stale diff on F: median={diffs[len(diffs) // 2]:.3f} "
      f"95%CI=[{diffs[50]:.3f}, {diffs[-50]:.3f}]")
