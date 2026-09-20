import unittest
from types import SimpleNamespace


class ExecutorFallbackTest(unittest.TestCase):
    def test_rotates_toward_edge_feature_without_approach_cell(self):
        from ocres.executor import Executor

        class Grid:
            passable = {(2, 1)}

            @staticmethod
            def interact_stand_cells(_target):
                return [((2, 1), (0, 1))]

        state = SimpleNamespace(
            players=[
                SimpleNamespace(position=(2, 1), orientation=(1, 0)),
                SimpleNamespace(position=(9, 9), orientation=(0, 1)),
            ]
        )
        action = Executor(Grid(), 0).interact_action(state, (2, 2))
        self.assertEqual(action, (0, 1))


if __name__ == "__main__":
    unittest.main()
