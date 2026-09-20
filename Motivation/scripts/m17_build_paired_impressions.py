"""Build old/new impressions from the frozen, disjoint M17 history probes.

The probe states were selected because the frozen checkpoints disagree after
an identical first move.  The recorded impression still contains only what
Bob sees: local frames, realized actions and outcomes.  Hidden intent and
target labels are never copied from ``scoring_only``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pathlib
import sys

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

from ocres.generic_agents import LayoutAwareTrainableAgent
from ocres.grid import STAY, World
from ocres.layouts import AMBIGUOUS_KITCHEN_V2
from ocres.m17_protocol import VisibleSegmentRecorder, realized_partner_action
from ocres.sensor import visible_state
from ocres.trainable import load_policy_checkpoint
from m17_build_observable_impressions import (
    LocalShadowObserver,
    interaction_stands,
    player_frame,
    public_frame,
    visible_outcome,
)
from m17_prepare_fixed_events import controlled_world


SPLITS = pathlib.Path("data/m17_fixed_event_splits.json")
PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m17_paired_impressions.json")


def run_probe(row, model, spec, horizon, max_visible_steps):
    probe = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=horizon)
    cooking_pot = sorted(probe.grid.pot_locs)[int(row["group"]["cooking_pot_index"])]
    before = row["public_event"]["before"]
    alice_position = tuple(before["alice_visible_position"])
    bob_position = tuple(before["bob_position"])
    world = controlled_world(cooking_pot, alice_position, bob_position, horizon)
    alice = LayoutAwareTrainableAgent(
        world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
    )
    bob = LocalShadowObserver(world.grid)
    onion_stands = interaction_stands(world.grid, world.grid.onion_locs)
    pot_stands = interaction_stands(world.grid, world.grid.pot_locs)
    recorder = VisibleSegmentRecorder()
    memory = {}

    while world.env.state.timestep < horizon and not world.env.is_done():
        before_state = world.env.state
        before_obs = visible_state(before_state, world.grid, 1, memory)
        memory = before_obs["seen_pots"]
        before_alice = player_frame(before_state, 0)
        alice_action, _, _, _ = alice.action(before_state, 0)
        bob_action = bob.action(before_obs)
        world.env.step((alice_action or STAY, bob_action or STAY))
        after_state = world.env.state
        after_obs = visible_state(after_state, world.grid, 1, memory)
        memory = after_obs["seen_pots"]

        if before_obs["partner_visible"] is None or after_obs["partner_visible"] is None:
            if recorder.active is None:
                recorder.start({
                    "t": int(before_state.timestep),
                    "position": list(before_alice["position"]),
                    "orientation": list(before_alice["orientation"]),
                    "held": before_alice["held"],
                    "action_result": {"kind": "leaves_view"},
                    "bob_position": list(before_obs["pos"]),
                    "visible_tiles": list(before_obs["tiles"]),
                    "remembered_pots": dict(before_obs["seen_pots"]),
                })
            recorder.close("lost_visibility")
            break

        after_alice = player_frame(after_state, 0)
        action_result = realized_partner_action(before_alice, after_alice)
        frame = public_frame(after_state, action_result, after_obs)
        if recorder.active is None:
            recorder.start(frame)
        else:
            recorder.observe(frame)
        outcome = visible_outcome(frame, onion_stands, pot_stands)
        if outcome is not None:
            recorder.close(outcome)
            break
        if len(recorder.active["frames"]) >= max_visible_steps:
            never_held_onion = all(item["held"] != "onion" for item in recorder.active["frames"])
            never_at_onion = all(
                tuple(item["position"]) not in onion_stands
                for item in recorder.active["frames"]
            )
            recorder.close(
                "remained_empty_without_fetch"
                if never_held_onion and never_at_onion
                else "visible_timeout_no_clue"
            )
            break

    if recorder.active is not None:
        recorder.close("episode_ended")
    segment = recorder.completed[0]
    segment.update({
        "seed": int(row["seed"]),
        "episode_id": str(row["episode_id"]),
        "event_id": str(row["event_id"]),
    })
    return segment


def summarize(segments):
    outcomes = Counter(row["outcome"] for row in segments)
    complete_names = {
        "seen_holding_onion", "seen_at_onion", "seen_waiting_at_pot",
        "remained_empty_without_fetch",
    }
    complete = sum(outcomes[name] for name in complete_names)
    return {
        "segments": len(segments),
        "complete_relevant": complete,
        "outcome_counts": dict(outcomes),
        "mean_visible_steps": (
            sum(row["visible_steps"] for row in segments) / len(segments) if segments else 0.0
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=pathlib.Path, default=SPLITS)
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--max-visible-steps", type=int, default=20)
    parser.add_argument("--minimum-complete", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    split_data = json.loads(args.splits.read_text(encoding="utf-8"))
    history_rows = split_data["history_probe"]["events"]
    pre_model, pre_spec, pre_meta = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, post_meta = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    pre = [run_probe(row, pre_model, pre_spec, args.horizon, args.max_visible_steps) for row in history_rows]
    post = [run_probe(row, post_model, post_spec, args.horizon, args.max_visible_steps) for row in history_rows]
    pre_summary = summarize(pre)
    post_summary = summarize(post)
    passed = all(
        row["complete_relevant"] >= args.minimum_complete
        for row in (pre_summary, post_summary)
    )
    result = {
        "milestone": "M17 disjoint paired-history impression gate",
        "passed": passed,
        "qwen_calls": 0,
        "source_split": "history_probe only",
        "segment_rule": "visibility loss immediately ends the probe segment",
        "forbidden_sources_in_saved_segments": [
            "Alice intent", "Alice target", "formal test events", "out-of-view continuation"
        ],
        "minimum_complete_per_type": args.minimum_complete,
        "checkpoint_meta": {"pre": pre_meta, "post": post_meta},
        "pre": {"summary": pre_summary, "segments": pre},
        "post": {"summary": post_summary, "segments": post},
        "none": {"summary": None, "segments": []},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "qwen_calls": 0,
        "minimum_complete_per_type": args.minimum_complete,
        "pre": pre_summary,
        "post": post_summary,
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
