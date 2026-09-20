"""Controlled A-score continuation, beginning after the frozen visible action."""
from __future__ import annotations

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

from ocres.grid import STAY
from ocres.m18_bob import PublicBob, PublicSensor, public_event
from ocres.m18_training import AlternatingAlice, randomized_world, soup_count


def continue_controlled(row, rows, model, spec, prediction, ticks=200, bob_class=PublicBob):
    world = randomized_world(rows, 185000+int(row["event_hash"][:8], 16), 700)
    world.env.state = OvercookedState.from_dict(row["controlled_state"])
    sensor = PublicSensor(world.grid)
    before = sensor.observe(world.env.state)
    alice = AlternatingAlice(world.grid, 0, model, spec, device="cpu", horizon=700, max_goal_ticks=50)
    bob = bob_class(world.grid)
    first, intent, target, _ = alice.action(world.env.state, 0)
    world.env.step((first if first is not None else STAY, STAY))
    after = sensor.observe(world.env.state)
    if public_event(before, after) != row["event"]:
        raise AssertionError("replayed public event differs from frozen A")
    bob.update_prediction(prediction, after["t"])
    soups, reward_total, reward_ticks = 0, 0, []
    for _ in range(ticks):
        if world.env.is_done():
            raise AssertionError("continuation ended before approved tick horizon")
        state = world.env.state
        obs = sensor.observe(state)
        action, _, _, _ = alice.action(state, soups)
        bob_action = bob.action(obs)
        _, reward, _, _ = world.env.step((action if action is not None else STAY, bob_action))
        soups += soup_count(reward)
        reward_total += reward
        if reward:
            reward_ticks.append(int(world.env.state.timestep))
        sensor.observe(world.env.state)
    return {"soups": soups, "reward": reward_total, "reward_ticks": reward_ticks,
            "bob_block_events": bob.block_events, "ticks": ticks,
            "initial_post_intent": intent, "initial_post_target": list(target) if target is not None else None}
