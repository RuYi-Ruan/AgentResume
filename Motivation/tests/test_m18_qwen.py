import json
from pathlib import Path
import tempfile
import unittest

from ocres.m18_qwen import decode_response, run_requests, summarize, trim_prompt


class QwenProtocolTests(unittest.TestCase):
    def test_invalid_output_is_not_repaired(self):
        vocab = {"onion_A": {"kind": "onion"}}
        response = {"choices": [{"message": {"content": json.dumps({
            "intent": "FETCH", "target_facility": "onion_A", "confidence": .7,
            "used_fact_ids": ["C1"], "secret": "no"})}}]}
        parsed, errors = decode_response(response, {"C1"}, vocab)
        self.assertIsNone(parsed)
        self.assertEqual(errors, ["invalid_fields"])

    def test_input_cap_never_drops_current_fact(self):
        messages = [{"role": "system", "content": "short"},
                    {"role": "user", "content": json.dumps({"facts": {
                        "C1": {"a": 1}, "H1": "x"*100, "H2": "y"*100}})}]
        trimmed, ids = trim_prompt(messages, 60)
        self.assertIn("C1", ids)
        self.assertNotIn("H1", ids)
        self.assertNotIn("H2", ids)

    def test_three_votes_resume_and_single_budget_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            def request(messages, options):
                return {"choices": [{"message": {"content": json.dumps({
                    "intent": "FETCH", "target_facility": "onion_A", "confidence": .7,
                    "used_fact_ids": ["C1"]})}}], "usage": {"total_tokens": 10}}
            options = {"request_limit": 10, "transport_retries": 2}
            plan = [{"id": f"t:r{i}", "event_index": 0, "event_hash": "abc",
                     "condition": "stale", "repeat": i, "messages": [],
                     "fact_ids": ["C1"], "vocabulary": {"onion_A": {"kind": "onion"}}}
                    for i in range(3)]
            output, budget = folder/"out.jsonl", folder/"budget.jsonl"
            run_requests(plan, options, output, budget, request)
            run_requests(plan, options, output, budget, request)
            self.assertEqual(len(budget.read_text().splitlines()), 3)
            truth = [{"scoring_only": {"post": {"intent": "FETCH", "target_facility": "onion_A"}}}]
            self.assertTrue(summarize(plan, output, truth)[0]["intent_correct"])


if __name__ == "__main__":
    unittest.main()
