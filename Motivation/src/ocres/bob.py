"""Bob: the two-card impression model (Resume minimal proxy).

Card 1 (level card) is implicit in WHICH experience book is consulted.
Card 2 (experience book) = per observed level's counts of
  context -> next Alice intent, where context = (alice cell, held, pot states)
    derived purely from OBSERVABLE behavior, never from Alice's logged intent.

Conditions:
  Stale   : book built only from L0 episodes (impression frozen in the past)
  Updated : book = stale + first half of the current Lk episode (incremental)
  Oracle  : book built from other Lk episodes (correct current level)
"""
from __future__ import annotations

from collections import Counter, defaultdict

SMOOTH = 0.3


def ctx_of(l):
    return (str(l["p0"]), l["held0"], str(l["pot"]))


class Book:
    def __init__(self):
        self.counts = defaultdict(Counter)
        self.total = Counter()

    def add_episode(self, logs):
        for l in logs:
            self.counts[ctx_of(l)][l["intent0"]] += 1
            self.total[ctx_of(l)] += 1

    def predict_proba(self, c):
        cnt = self.counts.get(c)
        if not cnt:
            return None  # unseen context
        denom = self.total[c] + SMOOTH * len(cnt)
        return {k: (v + SMOOTH) / denom for k, v in cnt.items()}

    def predict(self, c):
        p = self.predict_proba(c)
        return max(p, key=p.get) if p else None


def accuracy(book, logs, window=(0, None)):
    lo = window[0]
    hi = window[1] if window[1] else len(logs)
    n = agree = 0
    for l in logs[lo:hi]:
        pred = book.predict(ctx_of(l))
        if pred is None:
            continue
        n += 1
        agree += int(pred == l["intent0"])
    return agree / n if n else None, n


def level_divergence_ticks(logs0, logsk, lo, hi):
    """Ticks where L0 and Lk Alice intents differ at identical observable
    context (same positions/held/pots) -- the stale-impression trap set."""
    out = set()
    n = min(len(logs0), len(logsk))
    for t in range(lo, min(hi, n)):
        a, b = logs0[t], logsk[t]
        if (str(a["p0"]), str(a["p1"]), a["held0"], a["held1"], a["pot"]) == (
            str(b["p0"]),
            str(b["p1"]),
            b["held0"],
            b["held1"],
            b["pot"],
        ) and a["intent0"] != b["intent0"]:
            out.add(t)
    return out
