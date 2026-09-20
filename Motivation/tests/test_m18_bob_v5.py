import unittest

from ocres.m18_bob import PublicSensor
from ocres.m18_bob_v5 import PublicBobV5
from ocres.m18_training import group_spec, load_config, randomized_world


class M18BobV5Tests(unittest.TestCase):
    def setUp(self):
        rows = group_spec(load_config(), 0)["rows"]
        self.world = randomized_world(rows, 11)
        self.obs = PublicSensor(self.world.grid).observe(self.world.env.state)
        self.obs["bob"]["held"] = None
        self.obs["visible_pots"] = {}
        self.obs["remembered_pots"] = {
            "pot_A": {"kind": "items3", "seen_t": 0},
            "pot_B": {"kind": "cooking", "seen_t": 0},
            "pot_C": {"kind": "empty", "seen_t": 0},
        }

    def test_cook_start_prediction_selects_dish_preparation(self):
        bob = PublicBobV5(self.world.grid)
        bob.update_prediction({"intent": "COOK_START", "target_facility": "pot_A"}, self.obs["t"])
        bob.action(self.obs)
        self.assertEqual(bob.task, "GET_DISH")

    def test_serving_prediction_selects_ingredient_pipeline(self):
        bob = PublicBobV5(self.world.grid)
        bob.update_prediction({"intent": "DELIVER", "target_facility": "serve_A"}, self.obs["t"])
        bob.action(self.obs)
        self.assertEqual(bob.task, "FETCH")

    def test_held_object_work_is_not_overridden(self):
        self.obs["bob"]["held"] = "onion"
        bob = PublicBobV5(self.world.grid)
        bob.update_prediction({"intent": "DELIVER", "target_facility": "serve_A"}, self.obs["t"])
        bob.action(self.obs)
        self.assertEqual(bob.task, "PLACE")


if __name__ == "__main__":
    unittest.main()
