import unittest

import numpy as np

from ocres.mlp import MLPClassifier


class MLPClassifierTest(unittest.TestCase):
    def test_learns_a_small_separable_problem(self):
        x = np.array([[-2, -1], [-1, -2], [-2, -2], [1, 2], [2, 1], [2, 2]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        model = MLPClassifier(2, 2, hidden_dim=8, seed=7).fit(x, y, epochs=250)
        self.assertGreaterEqual(np.mean(model.predict(x) == y), 0.99)


if __name__ == "__main__":
    unittest.main()
