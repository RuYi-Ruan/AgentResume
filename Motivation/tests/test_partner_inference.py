import unittest


class PartnerInferenceTest(unittest.TestCase):
    def test_m16_prediction_selects_one_bounded_task(self):
        from ocres.generic_agents import PredictionRoleBob
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2

        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        bob = PredictionRoleBob(world.grid, horizon=20)
        bob.update_prediction("PRE", "pot_0")
        self.assertEqual(bob.commitment, "supply_one_onion")
        self.assertEqual(bob.predicted_facility, "pot_0")
        bob.action(world.env.state, 0)
        self.assertEqual(bob.role_mode, "cook")
        accepted = bob.update_prediction("FETCH", "onion_0")
        self.assertFalse(accepted)
        self.assertEqual(bob.commitment, "supply_one_onion")
        self.assertEqual(bob.predicted_facility, "pot_0")
        self.assertEqual(bob.ignored_prediction_updates, 1)
        bob.action(world.env.state, 0)
        self.assertEqual(bob.role_mode, "cook")

    def test_m16_supply_commitment_ends_after_one_onion_is_placed(self):
        from ocres.generic_agents import PredictionRoleBob
        from ocres.grid import World
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2

        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        bob = PredictionRoleBob(world.grid, horizon=20)
        bob.update_prediction("PRE")
        bob._commitment_started = True
        bob._last_seen_held = "onion"
        bob.action(world.env.state, 0)
        self.assertIsNone(bob.commitment)
        self.assertEqual(bob.role_mode, "serve")

    def test_intent_prediction_selects_complementary_role(self):
        from ocres.data import TWO_POT
        from ocres.grid import World
        from ocres.impression_events import FacilityVocabulary
        from ocres.partner_inference import PredictionAwareBob

        world = World.make(grid_rows=TWO_POT, horizon=20)
        bob = PredictionAwareBob(world.grid, 1, FacilityVocabulary.from_grid(world.grid))
        bob.update_prediction("FETCH", "onion_0")
        bob._set_complementary_role()
        self.assertEqual(bob.role_fixed, "serve")
        bob.update_prediction("DELIVER", "serve_0")
        bob._set_complementary_role()
        self.assertEqual(bob.role_fixed, "cook")

    def test_facility_prediction_selects_other_equivalent_facility(self):
        from ocres.data import TWO_POT
        from ocres.grid import World
        from ocres.impression_events import FacilityVocabulary
        from ocres.partner_inference import PredictionAwareBob

        world = World.make(grid_rows=TWO_POT, horizon=20)
        facilities = FacilityVocabulary.from_grid(world.grid)
        bob = PredictionAwareBob(world.grid, 1, facilities)
        pots = sorted(world.grid.pot_locs)
        avoided_name = facilities.names[facilities.coordinate_to_id[pots[0]]]
        bob.update_prediction("PLACE", avoided_name)
        chosen = bob._avoid_duplicate_facility(world.env.state, "PLACE", pots[0])
        self.assertEqual(chosen, pots[1])


if __name__ == "__main__":
    unittest.main()
