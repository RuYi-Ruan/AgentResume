"""M18 B natural collaboration with public-only query triggering."""
from __future__ import annotations

from ocres.grid import STAY
from ocres.m18_bob import PublicBob, PublicSensor, PublicTrigger, public_event
from ocres.m18_data import label_for
from ocres.m18_training import AlternatingAlice, randomized_world, soup_count


def run_b_episode(rows, seed, model, spec, config, condition, predict=None, bob_class=PublicBob):
    """predict(event, condition, seed, query_index) never receives hidden truth."""
    world = randomized_world(rows, seed, config["horizon"])
    alice = AlternatingAlice(world.grid, 0, model, spec, device="cpu",
                             horizon=config["horizon"], max_goal_ticks=config["max_goal_ticks"])
    bob = bob_class(world.grid, ttl=config["evaluation"]["prediction_ttl"])
    sensor = PublicSensor(world.grid)
    trigger = PublicTrigger(cooldown=config["evaluation"]["B_trigger_cooldown"],
                            limit=config["evaluation"]["B_max_queries"])
    before = sensor.observe(world.env.state)
    deliveries, score_300, score_700, delivery_t = 0, 0, 0, []
    questions = []
    semantic_starts, missed_visible_starts, last_task = 0, 0, None
    while not world.env.is_done():
        state = world.env.state
        alice_action, intent, target, info = alice.action(state, deliveries)
        truth = label_for(world.grid, intent, target)
        semantic = (truth["intent"], truth["target_facility"])
        new_task = info is not None and semantic != last_task
        if info is not None:
            last_task = semantic
        if new_task:
            semantic_starts += 1
        bob_action = bob.action(before)
        old_t = before["t"]
        _, reward, _, _ = world.env.step((alice_action if alice_action is not None else STAY, bob_action))
        added = soup_count(reward)
        if added:
            delivery_t.extend([int(world.env.state.timestep)]*added)
        deliveries += added
        score_700 += reward
        if old_t < config["evaluation"]["B_primary_ticks"]:
            score_300 += reward
        after = sensor.observe(world.env.state)
        fired = trigger.consider(before, after)
        if fired:
            observed = public_event(before, after)
            if observed is None:
                raise AssertionError("public trigger had no public event")
            query_index = len(questions)
            if condition == "oracle":
                vote, valid_votes = truth, 3
            else:
                if predict is None:
                    raise ValueError("Qwen condition requires a public event predictor")
                vote, valid_votes = predict(observed, condition, seed, query_index)
            bob.update_prediction(vote, after["t"])
            questions.append({"query_index": query_index, "public_t": after["t"],
                              "truth": truth, "prediction": vote, "valid_votes": valid_votes,
                              "intent_correct": vote["intent"] == truth["intent"],
                              "facility_correct": vote["target_facility"] == truth["target_facility"],
                              "true_new_task": new_task, "event": observed})
        elif new_task and public_event(before, after) is not None:
            missed_visible_starts += 1
        before = after
    return {"seed": seed, "condition": condition, "score_first_300": score_300,
            "score_full_700": score_700, "soups": deliveries,
            "time_to_five_soups": delivery_t[4] if len(delivery_t) >= 5 else None,
            "time_to_five_censored": len(delivery_t) < 5,
            "delivery_ticks": delivery_t, "public_queries": questions,
            "query_count": len(questions), "semantic_task_starts": semantic_starts,
            "visible_starts_without_query": missed_visible_starts,
            "bob_block_events": bob.block_events}
