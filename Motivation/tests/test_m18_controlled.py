import unittest

from ocres.m18_controlled import freeze_splits, public_rows
from ocres.m18_bob import event_hash


class ControlledSplitTests(unittest.TestCase):
    def test_public_file_has_no_scoring_or_snapshot(self):
        source = {"event_hash": "A", "family": "B", "event": {"before": 1},
                  "scoring_only": {"post": {"intent": "FETCH"}},
                  "controlled_state": {"secret": "marker"}}
        self.assertEqual(public_rows([source]), [{"event_hash":"A", "family":"B",
                                                  "public_event":{"before":1}}])

    def test_freeze_splits_never_reuses_a_family(self):
        candidates = []
        for index in range(80):
            candidates.append({"event_hash": f"h{index:03}", "family": f"f{index:03}",
                               "source_seed": index, "source_t": 0, "scoring_only": {
                                   "pre": {"intent":"PRE"}, "post": {"intent":"FETCH"}}})
        history = {"stale": [{"event_hash":"h000", "family":"f000"}], "updated": []}
        split = freeze_splits(candidates, history, 0)
        self.assertTrue(split["passed"])
        self.assertEqual(len(split["formal"]), 50)
        self.assertFalse({r["family"] for r in split["development"]} &
                         {r["family"] for r in split["formal"]})


if __name__ == "__main__":
    unittest.main()
