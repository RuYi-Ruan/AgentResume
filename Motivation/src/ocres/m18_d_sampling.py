"""Approved D-v2 ordinary-event sampler: exact uniqueness, not family uniqueness."""
from __future__ import annotations

from collections import Counter
import random


def choose_natural_d(rows, quota, max_per_episode, selection_seed):
    ordered = sorted(rows, key=lambda r: (r["source_seed"], r["source_t"], r["event_hash"]))
    random.Random(selection_seed).shuffle(ordered)
    selected, hashes, per_episode, excluded = [], set(), Counter(), Counter()
    for row in ordered:
        if row["event_hash"] in hashes:
            excluded["within_D_exact_duplicate"] += 1
            continue
        if per_episode[row["source_seed"]] >= max_per_episode:
            excluded["within_D_episode_cap"] += 1
            continue
        selected.append(row)
        hashes.add(row["event_hash"])
        per_episode[row["source_seed"]] += 1
        if len(selected) == quota:
            break
    return selected, dict(excluded)
