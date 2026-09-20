import json
import unittest

from ocres.m18_prompt import build_prompt, retrieval_score, top_history


ROWS = ["XOPXDXPXDXX", "X         X", "X X     X X", "S  1   2  S", "X X     X X", "X         X", "XXSOXPDXSXX"]


class PromptTests(unittest.TestCase):
    def test_prompt_discloses_only_public_fields(self):
        frame = {"t": 12, "bob": {"position": [4, 2], "orientation": [1, 0], "held": None},
                 "alice": {"position": [4, 1], "orientation": [1, 0], "held": None},
                 "visible_pots": {}, "remembered_pots": {}}
        event = {"before": frame, "after": {**frame, "t": 13}, "action": {"kind": "stay"}}
        messages, ids, vocab = build_prompt(event, [], ROWS)
        payload = json.loads(messages[1]["content"])
        self.assertEqual(ids, {"C1", "C2", "C3"})
        self.assertIn("pot_A", vocab)
        self.assertNotIn("scoring_only", messages[1]["content"])
        self.assertNotIn("source_seed", messages[1]["content"])
        self.assertNotIn("12", json.dumps(payload["facts"]))
        with self.assertRaises(ValueError):
            build_prompt({**event, "scoring_only": {"intent": "FETCH"}}, [], ROWS)

    def test_static_map_markers_are_not_fake_player_positions(self):
        frame = {"t": 12, "bob": {"position": [4, 2], "orientation": [1, 0], "held": None},
                 "alice": {"position": [4, 1], "orientation": [1, 0], "held": None},
                 "visible_pots": {}, "remembered_pots": {}}
        event = {"before": frame, "after": {**frame, "t": 13}, "action": {"kind": "stay"}}
        messages, _, vocab = build_prompt(event, [], ROWS, "v2_mapfix")
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["known_map"][3], "S         S")
        self.assertEqual(vocab["pot_B"]["position"], [5, 6])


if __name__ == "__main__":
    unittest.main()
