import json
import pathlib
import sys
import unittest


class M17QwenPromptTest(unittest.TestCase):
    def test_prompt_builder_never_reads_scoring_only_payload(self):
        scripts = pathlib.Path(__file__).resolve().parents[1] / "scripts"
        sys.path.insert(0, str(scripts))
        from m17_qwen_fixed import build_messages
        from ocres.grid import World
        from ocres.impression_events import FacilityVocabulary
        from ocres.layouts import AMBIGUOUS_KITCHEN_V2

        world = World.make(grid_rows=AMBIGUOUS_KITCHEN_V2, horizon=20)
        vocabulary = FacilityVocabulary.from_grid(world.grid)
        event = {
            "public_event": {
                "before": {"tiles": ["伙伴(空手)"], "bob_position": [1, 1]},
                "realized_alice_action": {"kind": "move", "delta": [1, 0]},
                "after": {"tiles": ["空地"], "bob_position": [1, 1]},
            },
            "scoring_only": {"sentinel": "SECRET_GROUND_TRUTH_12345"},
        }
        impressions = {
            "pre": {
                "segments": [{
                    "event_id": "history-1",
                    "outcome": "remained_empty_without_fetch",
                    "visible_steps": 1,
                    "frames": [{
                        "t": 1,
                        "position": [2, 1],
                        "held": None,
                        "action_result": {"kind": "move", "delta": [1, 0]},
                        "bob_position": [1, 1],
                    }],
                }]
            }
        }
        messages, fact_ids = build_messages(event, "pre", impressions, vocabulary)
        encoded = json.dumps(messages, ensure_ascii=False)
        self.assertNotIn("SECRET_GROUND_TRUTH_12345", encoded)
        self.assertEqual(fact_ids, {"M1", "F1", "F2", "F3", "H1", "H2"})


if __name__ == "__main__":
    unittest.main()
