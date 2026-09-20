"""Approved M18 controlled mechanism snapshots and frozen split selection.

These snapshots are deliberately constructed, not labelled natural episodes.
The class distribution is reported rather than manufactured.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from ocres.m18_bob import PublicSensor, visible
from ocres.m18_data import select_balanced, strict_pair
from ocres.m18_training import randomized_world


def enumerate_snapshots(rows, models, spec, map_name, horizon=700):
    base = randomized_world(rows, 184000, horizon)
    clones = [randomized_world(rows, 184000, horizon) for _ in range(2)]
    positions = sorted(base.grid.passable)
    candidates, funnel = [], Counter()
    for pot_index, pot in enumerate(sorted(base.grid.pot_locs)):
        for alice_position in positions:
            for bob_position in positions:
                if alice_position == bob_position or not visible(alice_position, bob_position):
                    continue
                state = base.env.state
                state.players[0].update_pos_and_or(alice_position, (1, 0))
                state.players[1].update_pos_and_or(bob_position, (1, 0))
                state.timestep = 0
                state.objects.clear()
                state.add_object(SoupState.get_soup(pot, num_onions=3, cooking_tick=5))
                sensor = PublicSensor(base.grid)
                sensor.observe(state)
                funnel["legal_visible_snapshots"] += 1
                pair = strict_pair(clones, state, 0, sensor.memory, models, spec, map_name)
                if pair is None:
                    continue
                funnel["strict_intent_change_same_visible_action"] += 1
                pair.update(controlled_state=state.to_dict(), controlled_grid={
                    "cooking_pot_index": pot_index, "alice_position": list(alice_position),
                    "bob_position": list(bob_position), "orientation": [1, 0],
                    "cooking_tick": 5},
                    source_seed=int(pair["family"][:12], 16), source_t=0,
                    source_kind="controlled-m18-v1")
                candidates.append(pair)
    return candidates, dict(funnel)


def freeze_splits(candidates, history, group, formal_n=50, development_n=10):
    if formal_n != 50 or development_n != 10:
        raise ValueError("approved controlled quotas changed")
    history_hashes = {s["event_hash"] for phase in ("stale", "updated") for s in history[phase]}
    history_families = {s["family"] for phase in ("stale", "updated") for s in history[phase]}
    exact, representatives = set(), {}
    report = Counter()
    for row in sorted(candidates, key=lambda r: (r["event_hash"], r["family"])):
        if row["event_hash"] in history_hashes or row["family"] in history_families:
            report["overlap_visible_history"] += 1
        elif row["event_hash"] in exact:
            report["within_controlled_exact_duplicate"] += 1
        elif row["family"] in representatives:
            report["within_controlled_near_family_duplicate"] += 1
        else:
            exact.add(row["event_hash"])
            representatives[row["family"]] = row
    pool = sorted(representatives.values(), key=lambda r: (r["event_hash"], r["family"]))
    # Families are locked to partitions before class balancing. Unlike natural
    # sources, no controlled row is misrepresented as an episode seed.
    development_pool = pool[:development_n]
    formal_pool = pool[development_n:]
    development = select_balanced(development_pool, development_n, 18700+group)
    formal = select_balanced(formal_pool, formal_n, 18900+group)
    if {r["family"] for r in development} & {r["family"] for r in formal}:
        raise AssertionError("controlled family overlap between splits")
    return {"development": development, "formal": formal,
            "passed": len(development) == development_n and len(formal) == formal_n,
            "funnel": {**report, "strict_candidates": len(candidates),
                       "independent_near_families_after_history": len(pool),
                       "development_selected": len(development), "formal_selected": len(formal)},
            "pre_to_post": dict(Counter((r["scoring_only"]["pre"]["intent"],
                                         r["scoring_only"]["post"]["intent"]) for r in formal)),
            "post_intent_counts": dict(Counter(r["scoring_only"]["post"]["intent"] for r in formal))}


def public_rows(split):
    """The Qwen-facing file cannot include scoring_only or controlled_state."""
    return [{"event_hash": row["event_hash"], "family": row["family"],
             "public_event": row["event"]} for row in split]


def split_hash(split):
    payload = json.dumps([r["event_hash"] for r in split], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
