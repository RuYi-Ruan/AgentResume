"""No-API feasibility probe for a controlled M18 mechanism dataset.

Uses auditable prepared pot snapshots; never labels them natural episodes. This
is an alternative design probe, not an approved formal sample generator.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")
import torch
from overcooked_ai_py.mdp.overcooked_mdp import SoupState

from ocres.m18_bob import PublicSensor, event_hash, near_family, visible
from ocres.m18_data import strict_pair
from ocres.m18_training import group_spec, load_config, randomized_world
from ocres.trainable import load_policy_checkpoint


def main():
    torch.set_num_threads(1)
    config = load_config()
    group = group_spec(config, 0)
    rows = group["rows"]
    base = randomized_world(rows, 104000)
    checkpoints = "artifacts/m18/v1_waitfix_20260915/checkpoints/group_0/"
    models = []
    for stage in ("pre", "post"):
        model, spec, _ = load_policy_checkpoint(checkpoints + stage + ".pt")
        model.eval()
        models.append(model)
    worlds = [randomized_world(rows, 104000) for _ in range(2)]
    counts = Counter()
    rows_out = []
    exact, families = set(), set()
    positions = sorted(base.grid.passable)
    for pot in base.grid.pot_locs:
        for alice in positions:
            for bob in positions:
                if alice == bob or not visible(alice, bob):
                    continue
                state = base.env.state
                state.players[0].update_pos_and_or(alice, (1, 0))
                state.players[1].update_pos_and_or(bob, (1, 0))
                state.objects.clear()
                state.add_object(SoupState.get_soup(pot, num_onions=3, cooking_tick=5))
                sensor = PublicSensor(base.grid)
                sensor.observe(state)
                counts["enumerated"] += 1
                pair = strict_pair(worlds, state, 0, sensor.memory, models, spec, group["map"])
                if pair is None:
                    continue
                counts["strict"] += 1
                counts["post_"+pair["scoring_only"]["post"]["intent"]] += 1
                exact.add(pair["event_hash"])
                families.add(pair["family"])
                rows_out.append({"pot": list(pot), "alice": list(alice), "bob": list(bob),
                                 "event_hash": pair["event_hash"], "family": pair["family"],
                                 "post": pair["scoring_only"]["post"]["intent"]})
    history = json.loads(Path("data/m18/v1_waitfix_20260915/group_0/history/impressions.json").read_text(encoding="utf-8"))
    history_hashes = {row["event_hash"] for phase in ("stale", "updated") for row in history[phase]}
    history_families = {row["family"] for phase in ("stale", "updated") for row in history[phase]}
    after_history = [row for row in rows_out if row["event_hash"] not in history_hashes
                     and row["family"] not in history_families]
    family_representatives = {}
    for row in sorted(after_history, key=lambda value: value["event_hash"]):
        family_representatives.setdefault(row["family"], row)
    available = list(family_representatives.values())
    dev, formal = available[:10], available[10:60]
    print(json.dumps({"counts": dict(counts), "unique_exact": len(exact),
                      "unique_near_families": len(families), "after_history_rows": len(after_history),
                      "after_history_families": len(available), "development_capacity": len(dev),
                      "formal_capacity": len(formal),
                      "candidate_rows": rows_out[:15]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
