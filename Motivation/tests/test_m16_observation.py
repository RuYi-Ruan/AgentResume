import unittest
from types import SimpleNamespace


def state(alice, bob):
    return SimpleNamespace(players=[SimpleNamespace(position=alice), SimpleNamespace(position=bob)])


class M16ObservationTest(unittest.TestCase):
    def test_visible_intent_change_triggers_one_query(self):
        from ocres.m16_observation import should_query_bob

        self.assertTrue(should_query_bob(state((3, 3), (4, 3)), intent_changed=True))

    def test_unseen_intent_change_is_not_revealed(self):
        from ocres.m16_observation import should_query_bob

        self.assertFalse(should_query_bob(state((1, 1), (5, 5)), intent_changed=True))

    def test_visibility_without_intent_change_does_not_repeat_query(self):
        from ocres.m16_observation import should_query_bob

        self.assertFalse(should_query_bob(state((3, 3), (4, 3)), intent_changed=False))

