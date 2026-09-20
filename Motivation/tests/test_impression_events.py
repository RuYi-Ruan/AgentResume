import unittest

import numpy as np

from ocres.data import TWO_POT
from ocres.grid import World


class ImpressionEventTest(unittest.TestCase):
    def setUp(self):
        from ocres.impression_events import FacilityVocabulary

        self.world = World.make(grid_rows=TWO_POT, horizon=20)
        self.facilities = FacilityVocabulary.from_grid(self.world.grid)

    @staticmethod
    def row(t, intent, target, action=(1, 0), p0=(1, 2), held0=None):
        return {
            "t": t,
            "a": [action, (0, 0)],
            "intent0": intent,
            "target0": target,
            "p0": p0,
            "p1": (5, 2),
            "held0": held0,
            "held1": None,
            "pot": "[((2, 0), 'empty'), ((4, 0), 'empty')]",
            "r": 0,
        }

    def test_facilities_have_names_not_coordinate_outputs(self):
        self.assertIn("pot_0", self.facilities.names)
        self.assertIn("pot_1", self.facilities.names)
        self.assertIn("none", self.facilities.names)
        for name in self.facilities.names:
            self.assertNotIn("(", name)

    def test_only_intent_changes_create_events(self):
        from ocres.impression_events import extract_intent_change_events

        onion = sorted(self.world.grid.onion_locs)[0]
        pot = sorted(self.world.grid.pot_locs)[0]
        logs = [
            self.row(0, "FETCH", onion),
            self.row(1, "FETCH", onion, p0=(2, 2)),
            self.row(2, "FETCH", onion, p0=(3, 2)),
            self.row(3, "PLACE", pot, held0="onion"),
            self.row(4, "PLACE", pot, held0="onion"),
        ]
        events, _ = extract_intent_change_events(logs, self.world.grid, horizon=20)
        self.assertEqual([event["intent"] for event in events], ["FETCH", "PLACE"])

    def test_feature_contains_first_action_and_post_state(self):
        from ocres.impression_events import ACTION_ID, action_observation_dim, extract_intent_change_events
        from ocres.trainable import ObservationSpec

        onion = sorted(self.world.grid.onion_locs)[0]
        logs = [
            self.row(0, "FETCH", onion, action=(1, 0), p0=(1, 2)),
            self.row(1, "FETCH", onion, p0=(2, 2)),
        ]
        events, _ = extract_intent_change_events(logs, self.world.grid, horizon=20)
        feature = events[0]["x"]
        spec = ObservationSpec()
        self.assertEqual(feature.shape, (action_observation_dim(spec),))
        action_slice = feature[spec.dim : spec.dim + len(ACTION_ID)]
        self.assertEqual(int(np.argmax(action_slice)), ACTION_ID[(1, 0)])
        self.assertEqual(float(action_slice.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
