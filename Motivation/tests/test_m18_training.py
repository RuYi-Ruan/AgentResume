import unittest

from ocres.m18_training import (
    AlternatingAlice, collect_expert, episode_seeds, geometry_report, group_spec,
    load_config, observation_spec, randomized_world, soup_count,
)
from ocres.trainable import INTENT_ID, MacroIntentPolicy
from overcooked_ai_py.mdp.overcooked_mdp import SoupState


class M18TrainingTests(unittest.TestCase):
    def test_sources_are_disjoint(self):
        config = load_config()
        all_seeds = []
        for group in range(9):
            for partition in config["seed_offsets"]:
                all_seeds.extend(episode_seeds(config, group, partition))
        self.assertEqual(len(all_seeds), len(set(all_seeds)))

    def test_layouts_and_legal_reproducible_start(self):
        config = load_config()
        for rows in config["maps"].values():
            report = geometry_report(rows)
            self.assertTrue(report["connected"] and report["accessible"])
            self.assertEqual(observation_spec(rows).dim, 182)
            a = randomized_world(rows, 123)
            b = randomized_world(rows, 123)
            self.assertTrue(a.env.state.time_independent_equal(b.env.state))
            self.assertNotEqual(a.env.state.players[0].position, a.env.state.players[1].position)

    def test_role_hint_follows_public_deliveries(self):
        rows = group_spec(load_config(), 0)["rows"]
        world = randomized_world(rows, 123)
        spec = observation_spec(rows)
        actor = AlternatingAlice(world.grid, 0, MacroIntentPolicy(spec.dim, 8), spec)
        actor.begin_intent(world.env.state, 0, INTENT_ID["HOLD"])
        self.assertEqual(actor.role_hint, "cook")
        actor.begin_intent(world.env.state, 1, INTENT_ID["HOLD"])
        self.assertEqual(actor.role_hint, "serve")

    def test_two_simultaneous_deliveries_count_twice(self):
        self.assertEqual(soup_count(40), 2)
        with self.assertRaises(ValueError):
            soup_count(1)

    def test_ready_soup_does_not_reset_in_progress_waiting_route(self):
        rows = group_spec(load_config(), 0)["rows"]
        world = randomized_world(rows, 123)
        state = world.env.state
        state.add_object(SoupState.get_soup((2, 0), num_onions=3, cooking_tick=20))
        spec = observation_spec(rows)
        actor = AlternatingAlice(world.grid, 0, MacroIntentPolicy(spec.dim, 8), spec)
        actor.begin_intent(state, 0, INTENT_ID["PRE"])
        actor._goal_ticks = 1
        actor._parking_target = (5, 3)
        state.players[0].update_pos_and_or((1, 1), (1, 0))
        self.assertFalse(actor.macro_complete(state, 0))
        state.players[0].update_pos_and_or((5, 3), (1, 0))
        self.assertTrue(actor.macro_complete(state, 0))

    def test_expert_labels_are_pre_action_and_not_repeated_ticks(self):
        rows = group_spec(load_config(), 0)["rows"]
        samples, metrics, trace = collect_expert(rows, 100001, horizon=80, keep_trace=True)
        boundaries = [row for row in trace if row["decision0"] is not None]
        self.assertEqual(len(samples), len(boundaries))
        self.assertLess(len(samples), len(trace))
        self.assertTrue(all(mask[label] for _, label, mask in samples))
        self.assertEqual(len({row["decision0"]["decision_id"] for row in boundaries}), len(samples))
        self.assertEqual(metrics["reward"], sum(row["reward"] for row in trace))

    def test_c_map_wait_deliver_oscillation_regression(self):
        rows = load_config()["maps"]["C"]
        _, metrics, trace = collect_expert(rows, 700001, keep_trace=True)
        # The failing trace previously alternated two states until tick 700,
        # with the final delivery at tick 384. Verify recovery, not a score target.
        self.assertTrue(any(t > 400 for t in metrics["delivery_ticks"]))
        self.assertGreater(sum(metrics["cycle_yields"]), 0)
        positions = [(tuple(row["position0"]), tuple(row["position1"])) for row in trace[-20:]]
        self.assertGreater(len(set(positions)), 2)


if __name__ == "__main__":
    unittest.main()
