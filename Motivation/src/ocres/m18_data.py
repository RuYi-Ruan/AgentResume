"""M18 local source collection and strict pairing, without model API calls."""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import random

from ocres.grid import STAY
from ocres.m18_bob import (
    INTENTS, PublicBob, PublicSensor, PublicTrigger, event_hash, facilities,
    near_family, public_event,
)
from ocres.m18_training import AlternatingAlice, randomized_world, soup_count


def label_for(grid, intent, target):
    if intent == "HOLD":
        return {"intent": intent, "target_facility": "none"}
    target_name = next((name for name, item in facilities(grid).items()
                        if tuple(item["position"]) == tuple(target)), None) if target is not None else None
    if target_name is None:
        raise ValueError(f"intent {intent} has no facility target")
    return {"intent": intent, "target_facility": target_name}


def collect_episode(rows, seed, model, spec, horizon=700, paired_models=None, map_name="A"):
    """Real trajectories with public Bob. Optional interventions only in clones.

    Real episode never receives oracle task notifications. Scoring labels and
    candidate source snapshots are stored in separate fields from public facts.
    """
    world = randomized_world(rows, seed, horizon)
    alice = AlternatingAlice(world.grid, 0, model, spec, device="cpu", horizon=horizon, max_goal_ticks=50)
    bob, sensor, trigger = PublicBob(world.grid), PublicSensor(world.grid), PublicTrigger()
    decisions, pairs, replay, segments = [], [], [], []
    active = None
    blocked_segment = False
    last_semantic = None
    deliveries, boundary_count, semantic_count, visible_count = 0, 0, 0, 0
    trigger_rows = []
    pair_worlds = [randomized_world(rows, seed, horizon) for _ in range(2)] if paired_models else None
    before = sensor.observe(world.env.state)
    while not world.env.is_done():
        state = world.env.state
        outcome = alice.action(state, deliveries)
        action, intent, target, info = outcome
        label = label_for(world.grid, intent, target)
        semantic = (label["intent"], label["target_facility"])
        boundary = info is not None
        new_semantic = boundary and semantic != last_semantic
        if boundary:
            boundary_count += 1
            last_semantic = semantic
        if new_semantic:
            semantic_count += 1
        if paired_models and boundary and before["alice"] is not None:
            paired = strict_pair(pair_worlds, state, deliveries, sensor.memory, paired_models, spec, map_name)
            if paired is not None:
                paired.update(source_seed=seed, source_t=int(state.timestep), source_state=state.to_dict(),
                              source_deliveries=deliveries)
                pairs.append(paired)
        bob_action = bob.action(before)
        _, reward, _, _ = world.env.step((action if action is not None else STAY, bob_action))
        deliveries += soup_count(reward)
        after = sensor.observe(world.env.state)
        event = public_event(before, after)
        if new_semantic and event is not None:
            visible_count += 1
            decisions.append({"source_seed": seed, "source_t": before["t"],
                              "event": event, "event_hash": event_hash(event),
                              "family": near_family(map_name, event), "scoring_only": label})
        if trigger.consider(before, after):
            trigger_rows.append({"t": after["t"], "true_new_semantic": new_semantic,
                                 "position": after["bob"]["position"]})
        # Segments are continuous visibility intervals, not hidden task segments.
        if before["alice"] is None or after["alice"] is None:
            if active is not None and len(active["frames"]) >= 2:
                active["end_reason"] = "lost_visibility"
                segments.append(active)
            active, blocked_segment = None, False
        elif not blocked_segment:
            if active is None:
                active = {"source_seed": seed, "start_t": before["t"],
                          "frames": [deepcopy(before)], "actions": [],
                          "event_hash": event_hash(event), "family": near_family(map_name, event)}
            active["frames"].append(deepcopy(after))
            active["actions"].append(event["action"])
            if len(active["actions"]) >= 12:
                active["end_reason"] = "length_limit"
                segments.append(active)
                active, blocked_segment = None, True
        replay.append({"t": before["t"], "actions": [action, bob_action], "reward": reward,
                       "intent": label, "new_semantic": new_semantic})
        before = after
    if active is not None and len(active["frames"]) >= 2:
        active["end_reason"] = "episode_end"
        segments.append(active)
    return {"seed": seed, "deliveries": deliveries, "task_boundaries": boundary_count,
            "semantic_task_starts": semantic_count, "visible_task_starts": visible_count,
            "public_triggers": trigger_rows, "decisions": decisions, "pairs": pairs,
            "segments": segments, "replay": replay, "bob_block_events": bob.block_events}


def strict_pair(worlds, source, deliveries, memory, models, spec, map_name):
    records = []
    for world, model in zip(worlds, models):
        world.env.state = source.deepcopy()
        sensor = PublicSensor(world.grid)
        sensor.memory = deepcopy(memory)
        before = sensor.observe(world.env.state)
        actor = AlternatingAlice(world.grid, 0, model, spec, device="cpu", horizon=700, max_goal_ticks=50)
        action, intent, target, _ = actor.action(world.env.state, deliveries)
        label = label_for(world.grid, intent, target)
        world.env.step((action if action is not None else STAY, STAY))
        after = sensor.observe(world.env.state)
        event = public_event(before, after)
        records.append((label, event))
    if records[0][0]["intent"] == records[1][0]["intent"]:
        return None
    if records[0][1] is None or records[0][1] != records[1][1]:
        return None
    event = records[1][1]
    return {"event": event, "event_hash": event_hash(event), "family": near_family(map_name, event),
            "scoring_only": {"pre": records[0][0], "post": records[1][0]}}


def paired_histories(pre_episodes, post_episodes):
    old, updated, accepted_sources, rejected_sources = [], [], [], []
    for first, second in zip(pre_episodes, post_episodes):
        if first["seed"] != second["seed"]:
            raise ValueError("history source pairing mismatch")
        count = min(len(first["segments"]), len(second["segments"]), 2)
        if not count:
            rejected_sources.append(first["seed"])
            continue
        accepted_sources.append(first["seed"])
        old.extend(first["segments"][:count])
        updated.extend(second["segments"][:count])
    return {"stale": old, "updated": updated, "accepted_sources": accepted_sources,
            "rejected_sources": rejected_sources,
            "passed": len(old) >= 40 and len(accepted_sources) >= 20}


def disjoint_candidates(rows, occupied_hashes, occupied_families):
    candidates, counts = [], Counter()
    seen_hashes = set()
    for row in rows:
        if row["event_hash"] in occupied_hashes or row["family"] in occupied_families:
            counts["cross_partition_duplicate"] += 1
        elif row["event_hash"] in seen_hashes:
            counts["within_partition_exact_duplicate"] += 1
        else:
            candidates.append(row)
            seen_hashes.add(row["event_hash"])
    return candidates, dict(counts)


def select_balanced(rows, count, seed):
    """Fixed round-robin of existing post classes with unique episode/family.

    Does not relax uniqueness to hit N; shortage is a gate failure.
    """
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["scoring_only"]["post"]["intent"]].append(row)
    rng = random.Random(seed)
    for name in INTENTS:
        buckets[name].sort(key=lambda row: (row["source_seed"], row["source_t"], row["event_hash"]))
        rng.shuffle(buckets[name])
    selected, used_episodes, used_families = [], set(), set()
    while len(selected) < count:
        old_length = len(selected)
        for name in INTENTS:
            while buckets[name]:
                row = buckets[name].pop()
                if row["source_seed"] in used_episodes or row["family"] in used_families:
                    continue
                selected.append(row)
                used_episodes.add(row["source_seed"])
                used_families.add(row["family"])
                break
            if len(selected) == count:
                break
        if len(selected) == old_length:
            break
    return selected
