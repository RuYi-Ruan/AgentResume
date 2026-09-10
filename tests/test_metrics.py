import unittest

from ocres.metrics import score_intent_events


class IntentEventMetricsTest(unittest.TestCase):
    def test_persistent_snapshots_are_counted_once(self):
        logs = [
            {"t": 0, "intent0": "FETCH", "info0": {"decision_event": True, "decision_id": 1, "cause_event": "E1"}, "info1": {}},
            {"t": 1, "intent0": "FETCH", "info0": {"decision_event": False, "decision_id": None, "cause": "E1"},
             "info1": {"guess": "FETCH", "guess_event": "FETCH", "guess_for": 1}},
            {"t": 2, "intent0": "FETCH", "info0": {"decision_event": False, "decision_id": None, "cause": "E1"},
             "info1": {"guess": "FETCH", "guess_event": None, "guess_for": None}},
        ]
        got = score_intent_events(logs)
        self.assertEqual(got["alice_decisions"], 1)
        self.assertEqual(got["guessed_decisions"], 1)
        self.assertEqual(got["cause_events"], 1)
        self.assertEqual(got["accuracy"], 1.0)
        self.assertEqual(got["cause_accuracy"], 1.0)

    def test_unpaired_and_invalid_guesses_are_not_scored(self):
        logs = [
            {"t": 0, "intent0": "HOLD", "info0": {"decision_event": True, "decision_id": 1, "cause_event": None}, "info1": {}},
            {"t": 1, "intent0": "HOLD", "info0": {}, "info1": {"guess_event": None, "guess_for": 1}},
        ]
        got = score_intent_events(logs)
        self.assertEqual(got["alice_decisions"], 1)
        self.assertEqual(got["guessed_decisions"], 0)
        self.assertIsNone(got["accuracy"])


if __name__ == "__main__":
    unittest.main()
