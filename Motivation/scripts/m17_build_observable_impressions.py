"""Build M17 impressions from strictly consecutive Bob-visible behavior.

No LLM is called.  Alice's hidden intent and target are never read.  A segment
ends immediately when Alice leaves Bob's 3x3 view, and actions are reconstructed
from the state before and after the environment step.
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
from ocres.grid import STAY
from ocres.m17_protocol import VisibleSegmentRecorder, realized_partner_action
from ocres.recipes import agent_held
from ocres.sensor import visible_state
from ocres.trainable import load_policy_checkpoint
from m16_v2_audit_checkpoints import randomized_world


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m17_observable_impressions.json")


class LocalShadowObserver:
    """A fixed probe policy using only Bob's current local observation.

    When Alice is visible, Bob steps toward her current cell.  If Alice moves,
    Bob usually enters the cell she just vacated; if she stays, the environment
    safely blocks Bob.  When Alice is not visible, Bob returns to a central
    passable cell.  No Alice intent, target or out-of-view position is read.
    """

    def __init__(self, grid):
        self.grid = grid
        width = len(grid.gw.terrain_mtx[0])
        height = len(grid.gw.terrain_mtx)
        center = width // 2, height // 2
        self.center = min(
            grid.passable,
            key=lambda cell: (
                abs(cell[0] - center[0]) + abs(cell[1] - center[1]),
                cell,
            ),
        )

    def action(self, observation):
        start = tuple(observation["pos"])
        target = (
            tuple(observation["partner_visible"])
            if observation["partner_visible"] is not None
            else self.center
        )
        step = self.grid.step_toward(start, target)
        if step is None:
            return STAY
        return step[0] - start[0], step[1] - start[1]


def interaction_stands(grid, facilities):
    return {
        stand
        for facility in facilities
        for stand, _ in grid.interact_stand_cells(facility)
    }


def player_frame(state, index):
    player = state.players[index]
    return {
        "position": list(player.position),
        "orientation": list(player.orientation),
        "held": agent_held(state, index),
    }


def public_frame(state, action_result, bob_observation):
    frame = player_frame(state, 0)
    frame.update({
        "t": int(state.timestep),
        "action_result": action_result,
        "bob_position": list(bob_observation["pos"]),
        "visible_tiles": list(bob_observation["tiles"]),
        "remembered_pots": dict(bob_observation["seen_pots"]),
    })
    return frame


def visible_outcome(frame, onion_stands, pot_stands):
    position = tuple(frame["position"])
    held = frame["held"]
    if held == "onion":
        return "seen_holding_onion"
    if held is None and position in onion_stands:
        return "seen_at_onion"
    if (
        held is None
        and position in pot_stands
        and frame["action_result"]["kind"] == "stay"
    ):
        return "seen_waiting_at_pot"
    return None


def collect_history(model, spec, seeds, horizon, max_visible_steps):
    all_segments = []
    per_seed = []
    totals = Counter()
    for seed in seeds:
        world = randomized_world(seed, horizon)
        center = LocalShadowObserver(world.grid).center
        if tuple(world.env.state.players[0].position) == center:
            center = min(
                (cell for cell in world.grid.passable if cell != tuple(world.env.state.players[0].position)),
                key=lambda cell: (
                    abs(cell[0] - LocalShadowObserver(world.grid).center[0])
                    + abs(cell[1] - LocalShadowObserver(world.grid).center[1]),
                    cell,
                ),
            )
        world.env.state.players[1].update_pos_and_or(center, (0, -1))
        alice = LayoutAwareTrainableAgent(
            world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
        )
        # The same local shadow/central probe policy is used for pre and post.
        bob = LocalShadowObserver(world.grid)
        onion_stands = interaction_stands(world.grid, world.grid.onion_locs)
        pot_stands = interaction_stands(world.grid, world.grid.pot_locs)
        memory = {}
        deliveries = 0
        recorder = VisibleSegmentRecorder()
        armed = True
        seed_segments = []

        while world.env.state.timestep < horizon and not world.env.is_done():
            before_state = world.env.state
            before_obs = visible_state(before_state, world.grid, 1, memory)
            memory = before_obs["seen_pots"]
            before_alice = player_frame(before_state, 0)
            alice_action, _, _, _ = alice.action(before_state, deliveries)
            bob_action = bob.action(before_obs)
            _, reward, _, _ = world.env.step((alice_action or STAY, bob_action or STAY))
            if reward > 0:
                deliveries += 1

            after_state = world.env.state
            after_obs = visible_state(after_state, world.grid, 1, memory)
            memory = after_obs["seen_pots"]
            before_visible = before_obs["partner_visible"] is not None
            after_visible = after_obs["partner_visible"] is not None

            if not (before_visible and after_visible):
                closed = recorder.observe(None)
                if closed is not None:
                    seed_segments.append(closed)
                    totals["lost_visibility"] += 1
                armed = True
                continue

            after_alice = player_frame(after_state, 0)
            action_result = realized_partner_action(before_alice, after_alice)
            frame = public_frame(after_state, action_result, after_obs)
            # Impression formation uses general, directly observed empty-hand
            # behavior. Requiring Bob to have already visited every pot would
            # silently select only a tiny and unrepresentative subset.
            eligible_before = before_alice["held"] is None

            if recorder.active is not None:
                recorder.observe(frame)
                outcome = visible_outcome(frame, onion_stands, pot_stands)
                if outcome is None and len(recorder.active["frames"]) >= max_visible_steps:
                    never_held_onion = all(item["held"] != "onion" for item in recorder.active["frames"])
                    never_at_onion = all(
                        tuple(item["position"]) not in onion_stands
                        for item in recorder.active["frames"]
                    )
                    outcome = (
                        "remained_empty_without_fetch"
                        if never_held_onion and never_at_onion
                        else "visible_timeout_no_clue"
                    )
                if outcome is not None:
                    closed = recorder.close(outcome)
                    seed_segments.append(closed)
                    totals[outcome] += 1
                    armed = False

            if not armed and after_alice["held"] is not None:
                armed = True
            if recorder.active is None and armed and eligible_before:
                recorder.start(frame)
                totals["segments_started"] += 1

        closed = recorder.close("episode_ended")
        if closed is not None:
            seed_segments.append(closed)
            totals["episode_ended"] += 1
        all_segments.extend(seed_segments)
        per_seed.append(
            {
                "seed": int(seed),
                "episode_id": f"history-{int(seed)}",
                "deliveries": deliveries,
                "segments": seed_segments,
            }
        )

    complete = [
        segment
        for segment in all_segments
        if segment["outcome"] in {
            "seen_holding_onion", "seen_at_onion", "seen_waiting_at_pot",
            "remained_empty_without_fetch",
        }
    ]
    totals["segments_total"] = len(all_segments)
    totals["segments_complete_relevant"] = len(complete)
    return {"totals": dict(totals), "segments": all_segments, "per_seed": per_seed}


def fingerprint(history):
    totals = Counter(history["totals"])
    segments = history["segments"]
    first_actions = Counter(segment["frames"][0]["action_result"]["kind"] for segment in segments)
    return {
        "segments_total": len(segments),
        "segments_complete_relevant": totals["segments_complete_relevant"],
        "first_action_counts": dict(first_actions),
        "outcome_counts": {
            name: totals[name]
            for name in (
                "seen_holding_onion",
                "seen_at_onion",
                "seen_waiting_at_pot",
                "remained_empty_without_fetch",
                "lost_visibility",
                "visible_timeout_no_clue",
                "episode_ended",
            )
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(9701, 9765)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--max-visible-steps", type=int, default=20)
    parser.add_argument("--minimum-complete", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, pre_meta = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, post_meta = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    pre = collect_history(pre_model, pre_spec, args.seeds, args.horizon, args.max_visible_steps)
    post = collect_history(post_model, post_spec, args.seeds, args.horizon, args.max_visible_steps)
    pre_fingerprint = fingerprint(pre)
    post_fingerprint = fingerprint(post)
    passed = all(
        item["segments_complete_relevant"] >= args.minimum_complete
        for item in (pre_fingerprint, post_fingerprint)
    )
    result = {
        "milestone": "M17 observable-impression data gate",
        "passed": passed,
        "qwen_calls": 0,
        "observer": "same fixed local-shadow/central probe policy in both conditions",
        "view": "3x3 Chebyshev radius 1",
        "segment_rule": "visibility loss immediately ends a segment",
        "action_rule": "action reconstructed from before/after public state",
        "forbidden_sources": ["Alice macro intent", "Alice target", "out-of-view continuation"],
        "seeds": args.seeds,
        "minimum_complete_per_type": args.minimum_complete,
        "checkpoint_meta": {"pre": pre_meta, "post": post_meta},
        "pre": {"fingerprint": pre_fingerprint, "history": pre},
        "post": {"fingerprint": post_fingerprint, "history": post},
        "none": {"fingerprint": None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "passed": passed,
        "minimum_complete_per_type": args.minimum_complete,
        "pre": pre_fingerprint,
        "post": post_fingerprint,
        "qwen_calls": 0,
    }, ensure_ascii=False, indent=2))
    print("results:", args.output)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
