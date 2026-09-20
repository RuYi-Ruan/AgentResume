import unittest


class M17ProtocolTest(unittest.TestCase):
    def test_blocked_command_is_recorded_as_stay(self):
        from ocres.m17_protocol import realized_partner_action

        before = {"position": (2, 2), "orientation": (1, 0), "held": None}
        after = {"position": (2, 2), "orientation": (1, 0), "held": None}
        self.assertEqual(realized_partner_action(before, after), {"kind": "stay"})

    def test_realized_move_uses_position_delta(self):
        from ocres.m17_protocol import realized_partner_action

        before = {"position": (2, 2), "orientation": (1, 0), "held": None}
        after = {"position": (3, 2), "orientation": (1, 0), "held": None}
        self.assertEqual(
            realized_partner_action(before, after),
            {"kind": "move", "delta": [1, 0]},
        )

    def test_visibility_gap_closes_segment_immediately(self):
        from ocres.m17_protocol import VisibleSegmentRecorder

        recorder = VisibleSegmentRecorder()
        recorder.start({"t": 4, "position": [2, 2]})
        closed = recorder.observe(None)
        self.assertEqual(closed["outcome"], "lost_visibility")
        self.assertEqual(closed["end_t"], 4)
        self.assertIsNone(recorder.active)
        recorder.observe({"t": 9, "position": [5, 5]})
        self.assertEqual(len(recorder.completed[0]["frames"]), 1)

    def test_dataset_partitions_must_be_disjoint(self):
        from ocres.m17_protocol import validate_partition_provenance

        row = {"seed": 1, "episode_id": "e1", "event_id": "x"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_partition_provenance({"history": [row], "test": [dict(row)]})

    def test_llm_response_accepts_unseen_target_as_prediction(self):
        from ocres.m17_protocol import validate_intent_response

        value = {
            "intent": "FETCH",
            "target_facility": "onion_1",
            "confidence": 0.7,
            "used_fact_ids": ["F1", "H3"],
        }
        valid, errors = validate_intent_response(
            value,
            allowed_fact_ids={"F1", "H3"},
            allowed_facilities={"pot_0", "onion_1"},
        )
        self.assertTrue(valid, errors)

    def test_llm_response_rejects_invented_fact_and_free_text(self):
        from ocres.m17_protocol import validate_intent_response

        value = {
            "intent": "PRE",
            "target_facility": "pot_0",
            "confidence": 0.8,
            "used_fact_ids": ["F9"],
            "reason": "我猜她会去锅边",
        }
        valid, errors = validate_intent_response(
            value, allowed_fact_ids={"F1"}, allowed_facilities={"pot_0"}
        )
        self.assertFalse(valid)
        self.assertTrue(any(item.startswith("unknown_fact_ids") for item in errors))
        self.assertTrue(any(item.startswith("extra_fields") for item in errors))


if __name__ == "__main__":
    unittest.main()
