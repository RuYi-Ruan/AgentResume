from copy import deepcopy
import unittest

from ocres.m18_bob import (
    PublicBob, PublicSensor, PublicTrigger, event_hash, facilities, majority,
    public_event, validate_response,
)
from ocres.m18_training import group_spec, load_config, randomized_world
from overcooked_ai_py.mdp.overcooked_mdp import SoupState


class M18BobTests(unittest.TestCase):
    def setUp(self):
        self.rows = group_spec(load_config(), 0)["rows"]
        self.world = randomized_world(self.rows, 10)

    def test_hidden_world_change_does_not_change_bob_input_or_action(self):
        state = self.world.env.state
        state.players[1].update_pos_and_or((5, 3), (1, 0))
        state.players[0].update_pos_and_or((1, 1), (1, 0))
        before = PublicSensor(self.world.grid).observe(state)
        state.players[0].update_pos_and_or((9, 5), (1, 0))
        state.add_object(SoupState.get_soup((2, 0), num_onions=3, cooking_tick=5))
        after = PublicSensor(self.world.grid).observe(state)
        self.assertEqual(before, after)
        self.assertEqual(PublicBob(self.world.grid).action(before), PublicBob(self.world.grid).action(after))

    def test_all_intents_validate_without_private_mask(self):
        vocab = facilities(self.world.grid)
        targets = {"FETCH":"onion_A", "PLACE":"pot_A", "COOK_START":"pot_A",
                   "GET_DISH":"dish_A", "PICKUP":"pot_A", "DELIVER":"serve_A",
                   "PRE":"pot_A", "HOLD":"none", "unknown":"unknown"}
        for intent, target in targets.items():
            value = {"intent":intent, "target_facility":target, "confidence":0.5,"used_fact_ids":["C1"]}
            self.assertTrue(validate_response(value, {"C1"}, vocab)[0])
        value["used_fact_ids"] = ["hidden"]
        self.assertFalse(validate_response(value, {"C1"}, vocab)[0])
        value["intent"] = ["FETCH"]
        self.assertFalse(validate_response(value, {"C1"}, vocab)[0])

    def test_prediction_changes_public_task_without_truth_and_expires(self):
        grid = self.world.grid
        obs = PublicSensor(grid).observe(self.world.env.state)
        obs["bob"]["held"] = None
        obs["remembered_pots"] = {
            "pot_A": {"kind": "empty", "seen_t": 0},
            "pot_B": {"kind": "cooking", "seen_t": 0},
        }
        obs["visible_pots"] = {}
        bob = PublicBob(grid)
        bob.action(obs)
        self.assertEqual(bob.task, "FETCH")
        bob.update_prediction({"intent":"FETCH", "target_facility":"unknown"}, 0)
        bob.action(obs)
        self.assertEqual(bob.task, "GET_DISH")
        obs["t"] = 12
        bob.action(obs)
        self.assertEqual(bob.task, "FETCH")

    def test_vote_does_not_drop_invalid_or_split_intents(self):
        fetch = {"intent":"FETCH", "target_facility":"onion_A"}
        pre = {"intent":"PRE", "target_facility":"pot_A"}
        self.assertEqual(majority([fetch, pre, None])["intent"], "unknown")
        self.assertEqual(majority([fetch, fetch, pre]), fetch)
        split = {"intent":"FETCH", "target_facility":"onion_B"}
        self.assertEqual(majority([fetch, split, pre]), {"intent":"FETCH", "target_facility":"unknown"})

    def test_hash_ignores_absolute_time_and_trigger_does_not_repeat_straight_walk(self):
        state = self.world.env.state
        state.players[1].update_pos_and_or((5, 3), (1, 0))
        state.players[0].update_pos_and_or((4, 3), (1, 0))
        before = PublicSensor(self.world.grid).observe(state)
        after = deepcopy(before)
        after["t"] = 1
        event = public_event(before, after)
        later = deepcopy(event)
        for frame in (later["before"], later["after"]):
            frame["t"] += 40
            for value in frame["remembered_pots"].values():
                value["seen_t"] += 40
        self.assertEqual(event_hash(event), event_hash(later))
        trigger = PublicTrigger()
        self.assertTrue(trigger.consider(before, after))
        for tick in range(2, 30):
            before, after = deepcopy(after), deepcopy(after)
            after["t"] = tick
            self.assertFalse(trigger.consider(before, after))


if __name__ == "__main__":
    unittest.main()
