import unittest

import numpy as np

from MainSearch.explore.etm_gradual_learning_pilot.run_experiment import (
    Config,
    TeammateModel,
    posterior_intent,
    run_experiment,
    run_seed,
)


class PilotTest(unittest.TestCase):
    def test_posterior_uses_teammate_prior(self) -> None:
        prior = np.array([0.05, 0.90, 0.05])
        posterior = posterior_intent(prior, cue=0, cue_accuracy=0.55)
        self.assertEqual(int(np.argmax(posterior)), 1)

    def test_dynamic_model_updates_by_identity(self) -> None:
        config = Config(seeds=1, rounds=30)
        model = TeammateModel(config)
        for _ in range(20):
            model.update("B", context=0, role=2)
        self.assertGreater(model.distribution("B", 0)[2], 0.8)
        self.assertTrue(np.allclose(model.distribution("C", 0), 1.0 / 3.0))

    def test_seed_has_all_conditions_each_round(self) -> None:
        config = Config(seeds=1, rounds=20)
        rows = run_seed(0, config)
        self.assertEqual(len(rows), config.rounds * 4)
        for round_index in range(config.rounds):
            round_rows = [row for row in rows if row["round"] == round_index]
            self.assertEqual(len(round_rows), 4)
            self.assertEqual(
                len({row["partner_capability"] for row in round_rows}), 1
            )

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
        self.assertEqual(summary["events"], 3 * 90 * 4)
        self.assertEqual(summary["status"], "development_pilot")


if __name__ == "__main__":
    unittest.main()
