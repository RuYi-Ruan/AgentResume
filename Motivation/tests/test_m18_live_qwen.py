import json
from pathlib import Path
import tempfile
import unittest

from ocres.m18_live_qwen import LiveQwen


ROWS = ["XOPXDXPXDXX", "X         X", "X X     X X", "S  1   2  S", "X X     X X", "X         X", "XXSOXPDXSXX"]


class LiveQwenTests(unittest.TestCase):
    def test_public_prompt_and_replay_do_not_repeat_paid_calls(self):
        frame = {"t": 12, "bob": {"position": [4, 2], "orientation": [1, 0], "held": None},
                 "alice": {"position": [4, 1], "orientation": [1, 0], "held": None},
                 "visible_pots": {}, "remembered_pots": {}}
        event = {"before": frame, "after": {**frame, "t": 13}, "action": {"kind": "stay"}}
        options = {"model": "mock", "repeats": 3, "max_input_tokens": 4096,
                   "request_limit": 10, "transport_retries": 2}
        def mock_request(messages, options):
            return {"choices": [{"message": {"content": json.dumps({
                "intent": "unknown", "target_facility": "unknown", "confidence": .1,
                "used_fact_ids": ["C1"]})}}]}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            output, budget = folder/"B", folder/"budget.jsonl"
            first = LiveQwen(ROWS, {"stale": [], "updated": []}, options,
                             output, budget, 0, request_fn=mock_request)
            vote, valid = first.predict(event, "none", 106000, 0)
            self.assertEqual((vote["intent"], valid), ("unknown", 3))
            second = LiveQwen(ROWS, {"stale": [], "updated": []}, options,
                              output, budget, 0, request_fn=mock_request)
            self.assertEqual(second.predict(event, "none", 106000, 0), (vote, 3))
            self.assertEqual(len(budget.read_text(encoding="utf-8").splitlines()), 3)
            prompt = json.loads((output/"prompts.jsonl").read_text(encoding="utf-8"))
            content = json.loads(prompt["messages"][1]["content"])
            self.assertNotIn("1", content["known_map"][3])
            self.assertNotIn("2", content["known_map"][3])
            self.assertNotIn("truth", prompt["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
