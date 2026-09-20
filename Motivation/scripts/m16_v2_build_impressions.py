"""Build M16-V2 impression text from Bob-visible history only."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
sys.path.insert(0, "src")

from ocres.generic_agents import LayoutAwareTrainableAgent, ScriptedMacroAgent
from ocres.grid import STAY
from ocres.sensor import visible_state
from ocres.trainable import load_policy_checkpoint
from m16_v2_audit_checkpoints import randomized_world


PRE_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_pre.pt")
POST_CHECKPOINT = pathlib.Path("artifacts/checkpoints/m16_v2_alice_post.pt")
OUTPUT = pathlib.Path("data/m16_v2_impressions.json")


def interaction_stands(grid, facilities):
    return {
        stand
        for facility in facilities
        for stand, _ in grid.interact_stand_cells(facility)
    }


def collect_visible_history(model, spec, seeds, horizon):
    totals = Counter()
    per_seed = []
    all_episodes = []
    for seed in seeds:
        world = randomized_world(seed, horizon)
        alice = LayoutAwareTrainableAgent(
            world.grid, 0, model, spec, device="cpu", horizon=horizon, role_hint="cook"
        )
        bob = ScriptedMacroAgent(world.grid, 1, "serve", True, horizon)
        onion_stands = interaction_stands(world.grid, world.grid.onion_locs)
        pot_stands = interaction_stands(world.grid, world.grid.pot_locs)
        memory = {}
        deliveries = 0
        local = Counter()
        episodes = []
        pending = None
        armed = True
        while world.env.state.timestep < horizon and not world.env.is_done():
            state = world.env.state
            observation = visible_state(state, world.grid, 1, memory)
            memory = observation["seen_pots"]
            alice_action, _, _, _ = alice.action(state, deliveries)
            bob_action, _, _, _ = bob.action(state, deliveries)
            if observation["partner_visible"] is not None:
                local["visible_ticks"] += 1
                alice_player = state.players[0]
                held = alice_player.get_object().name if alice_player.has_object() else None
                if held is None:
                    local["empty_ticks"] += 1
                    if alice_action == STAY:
                        local["empty_stay"] += 1
                    elif alice_action == "interact":
                        local["empty_interact"] += 1
                    elif isinstance(alice_action, tuple):
                        local["empty_move"] += 1
                    if tuple(alice_player.position) in onion_stands:
                        local["empty_at_onion"] += 1
                    if tuple(alice_player.position) in pot_stands:
                        local["empty_at_pot"] += 1
                elif held == "onion":
                    local["holding_onion"] += 1

                remembered = [entry["kind"] for entry in memory.values()]
                knows_running = any(kind in ("cooking", "ready") for kind in remembered)
                knows_accepting = any(kind in ("empty", "items1", "items2") for kind in remembered)
                if held is None and knows_running and knows_accepting:
                    local["known_choice_ticks"] += 1
                    if alice_action == STAY:
                        local["known_choice_stay"] += 1
                    elif alice_action == "interact":
                        local["known_choice_interact"] += 1
                    elif isinstance(alice_action, tuple):
                        local["known_choice_move"] += 1

                # Build partner memories from visible sequences only.  We never
                # read Alice's macro label or target.  Similar consecutive ticks
                # are collapsed into one episode and re-arm only after Bob sees
                # a meaningful external change.
                choice_now = held is None and knows_running and knows_accepting
                if pending is not None and state.timestep > pending["start_t"]:
                    elapsed = int(state.timestep - pending["start_t"])
                    outcome = None
                    if pending["first_action"] == "stay" and alice_action == STAY:
                        outcome = "later_seen_continued_stay"
                    elif held == "onion":
                        outcome = "later_seen_holding_onion"
                    elif held is None and tuple(alice_player.position) in onion_stands:
                        outcome = "later_seen_at_onion"
                    elif (
                        held is None
                        and tuple(alice_player.position) in pot_stands
                        and alice_action == STAY
                    ):
                        outcome = "later_seen_waiting_at_pot"
                    elif elapsed >= 20:
                        outcome = "no_later_visible_clue"
                    if outcome is not None:
                        episode = {**pending, "elapsed": elapsed, "outcome": outcome}
                        episodes.append(episode)
                        local[f"episode_outcome_{outcome}"] += 1
                        pending = None
                        armed = False

                if not armed and (held is not None or not choice_now):
                    armed = True
                if pending is None and armed and choice_now:
                    if alice_action == STAY:
                        first_action = "stay"
                    elif alice_action == "interact":
                        first_action = "interact"
                    elif isinstance(alice_action, tuple):
                        first_action = "move"
                    else:
                        first_action = "other"
                    pending = {
                        "seed": int(seed),
                        "start_t": int(state.timestep),
                        "first_action": first_action,
                    }
                    local["memory_episodes_started"] += 1
                    local[f"episode_first_{first_action}"] += 1

            if pending is not None and state.timestep - pending["start_t"] >= 20:
                episode = {
                    **pending,
                    "elapsed": int(state.timestep - pending["start_t"]),
                    "outcome": "no_later_visible_clue",
                }
                episodes.append(episode)
                local["episode_outcome_no_later_visible_clue"] += 1
                pending = None
                armed = False

            _, reward, _, _ = world.env.step((alice_action or STAY, bob_action or STAY))
            if reward > 0:
                deliveries += 1
        totals.update(local)
        all_episodes.extend(episodes)
        per_seed.append({"seed": int(seed), "deliveries": deliveries, "episodes": episodes, **dict(local)})
    return {"totals": dict(totals), "episodes": all_episodes, "per_seed": per_seed}


def percent(part, total):
    return 100.0 * part / total if total else 0.0


def impression_text(history):
    values = Counter(history["totals"])
    episodes = len(history["episodes"])
    episode_values = Counter(episode["first_action"] for episode in history["episodes"])
    first_move = episode_values["move"]
    first_stay = episode_values["stay"]
    later_onion = values["episode_outcome_later_seen_holding_onion"] + values["episode_outcome_later_seen_at_onion"]
    later_wait = values["episode_outcome_later_seen_waiting_at_pot"] + values["episode_outcome_later_seen_continued_stay"]
    unknown = values["episode_outcome_no_later_visible_clue"]
    return (
        "以下是我在相似局面前后亲眼看到的伙伴记忆；中间看不见的过程没有补写，也不假定属于同一任务：\n"
        f"- 我记得 {episodes} 个相似片段：当时我知道一口锅正在烹饪、另有锅还能补料，"
        "并看见 Alice 空手。\n"
        f"- 这些片段开始时，我看见她移动 {first_move} 次、停留 {first_stay} 次。\n"
        f"- 在之后再次可见的画面中，我看到她拿着洋葱或到达洋葱台 {later_onion} 次；"
        f"看到她继续空手停留或停在锅边 {later_wait} 次；另外 {unknown} 次在 20 步内没有再看到足够线索。\n"
        "这些只是有限视野下真实发生过的外在行为，不代表我知道她当时的内心目标。"
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-checkpoint", type=pathlib.Path, default=PRE_CHECKPOINT)
    parser.add_argument("--post-checkpoint", type=pathlib.Path, default=POST_CHECKPOINT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(9701, 9717)))
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    pre_model, pre_spec, _ = load_policy_checkpoint(args.pre_checkpoint, map_location="cpu")
    post_model, post_spec, _ = load_policy_checkpoint(args.post_checkpoint, map_location="cpu")
    if pre_spec != post_spec:
        raise ValueError("pre/post checkpoint observation specs differ")
    pre = collect_visible_history(pre_model, pre_spec, args.seeds, args.horizon)
    post = collect_visible_history(post_model, post_spec, args.seeds, args.horizon)
    result = {
        "milestone": "M16-V2 observable episode impression generation",
        "source_constraint": "Only behavior witnessed inside Bob's 3x3 view is summarized; gaps remain unknown.",
        "forbidden_sources": ["Alice intent labels", "Alice targets", "checkpoint names in text", "out-of-view behavior"],
        "seeds": args.seeds,
        "pre": {"history": pre, "text": impression_text(pre)},
        "post": {"history": post, "text": impression_text(post)},
        "none": {"text": "Alice 是我的合作伙伴；我没有她个人行为习惯的历史记录。"},
        "qwen_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value["text"] for key, value in result.items() if key in ("pre", "post", "none")}, ensure_ascii=False, indent=2))
    print("results:", args.output)


if __name__ == "__main__":
    main()
