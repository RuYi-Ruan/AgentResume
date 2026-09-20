import unittest
from collections import deque

from ocres.grid import World
from ocres.llm_player import LLMPlayer
from ocres.runner import run_episode


ROWS = ["XXPXPXX", "O     O", "X 1 2 X", "X D S X", "XXXXXXX"]


class DecisionEventFlowTest(unittest.TestCase):
    def test_bob_guesses_once_on_the_tick_after_alice_decides(self):
        board = deque(maxlen=6)
        world = World.make(grid_rows=ROWS, horizon=4)

        def alice_chat(_system, _user):
            return {"goal": "HOLD", "target": None, "reason": "test", "cause": None}

        def bob_chat(_system, _user):
            return {"goal": "HOLD", "target": None, "reason": "test", "alice_guess": "HOLD"}

        alice = LLMPlayer(world.grid, 0, alice_chat, name="Alice", board=board)
        bob = LLMPlayer(world.grid, 1, bob_chat, name="Bob", bob_knows=False, board=board)
        logs, _ = run_episode(world, [alice, bob], horizon=4)

        alice_events = [row for row in logs if (row["info0"] or {}).get("decision_event")]
        bob_events = [row for row in logs if (row["info1"] or {}).get("guess_event") is not None]
        self.assertEqual([row["t"] for row in alice_events], [0])
        self.assertEqual([row["t"] for row in bob_events], [1])
        self.assertEqual(bob_events[0]["info1"]["guess_for"], alice_events[0]["info0"]["decision_id"])


if __name__ == "__main__":
    unittest.main()
