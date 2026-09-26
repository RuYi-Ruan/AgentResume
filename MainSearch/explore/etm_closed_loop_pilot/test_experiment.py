import unittest

import numpy as np

from MainSearch.explore.etm_closed_loop_pilot.run_experiment import (
    AGENTS,
    Config,
    PairwiseETM,
    categorical_from_uniform,
    role_targets,
    run_experiment,
    run_seed,
)


class ClosedLoopPilotTest(unittest.TestCase):
    def test_targets_are_permutations(self) -> None:
        targets = role_targets()
        for context in range(6):
            self.assertEqual(
                {int(targets[agent][context]) for agent in AGENTS}, {0, 1, 2}
            )

    def test_pairwise_models_are_directional(self) -> None:
        model = PairwiseETM(Config(seeds=1, rounds=10))
        for _ in range(20):
            model.update("A", "B", 0, 2)
        self.assertGreater(model.distribution("A", "B", 0)[2], 0.8)
        self.assertTrue(np.allclose(model.distribution("C", "B", 0), 1.0 / 3.0))

    def test_categorical_boundary(self) -> None:
        probs = np.array([0.2, 0.3, 0.5])
        self.assertEqual(categorical_from_uniform(probs, 0.1), 0)
        self.assertEqual(categorical_from_uniform(probs, 0.4), 1)
        self.assertEqual(categorical_from_uniform(probs, 0.9), 2)

    def test_seed_is_deterministic(self) -> None:
        config = Config(seeds=1, rounds=30, warmup=5)
        self.assertEqual(run_seed(7, config), run_seed(7, config))

    def test_smoke_experiment(self) -> None:
        config = Config(
            seeds=3,
            rounds=90,
            window=30,
            warmup=15,
            permutation_samples=200,
        )
        windows, summary = run_experiment(config)
        self.assertEqual(len(windows), 3 * 4)
        self.assertEqual(summary["condition_rounds"], 3 * 90 * 4)


if __name__ == "__main__":
    unittest.main()
