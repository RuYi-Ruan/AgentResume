import unittest

import gymnasium as gym
import lbforaging  # noqa: F401
import numpy as np

from MainSearch.explore.lbf_etm_route_pilot.run_experiment import (
    Config,
    ETM,
    configure_episode,
    execute_targets,
    expected_collection_value,
    run_experiment,
)


class LBFETMRouteTest(unittest.TestCase):
    def test_dynamic_model_updates_one_relationship(self) -> None:
        model = ETM(Config(seeds=1, episodes=10))
        for _ in range(20):
            model.update("A", "B", 2)
        self.assertGreater(model.distribution("A", "B")[2], 0.8)
        self.assertTrue(np.allclose(model.distribution("C", "B"), 1.0 / 3.0))

    def test_expected_value_rewards_valid_coalitions(self) -> None:
        teammate_distributions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
        value = expected_collection_value(0, 1, teammate_distributions)
        self.assertAlmostEqual(value, 0.5)

    def test_real_lbf_collects_split_coalitions(self) -> None:
        config = Config(seeds=1, episodes=1)
        env = gym.make(config.environment_id, disable_env_checker=True)
        try:
            env.reset(seed=5)
            foods = configure_episode(env)
            outcome = execute_targets(env, foods, [0, 1, 1])
            self.assertEqual(outcome["collected_value"], 3)
            self.assertAlmostEqual(outcome["team_reward"], 0.5)
        finally:
            env.close()

    def test_smoke_run(self) -> None:
        config = Config(
            seeds=2,
            episodes=30,
            window=10,
            warmup=5,
            epsilon_decay_episodes=25,
            permutation_samples=200,
        )
        windows, summary = run_experiment(config)
        self.assertEqual(len(windows), 3 * 3)
        self.assertEqual(summary["condition_episodes"], 2 * 30 * 3)


if __name__ == "__main__":
    unittest.main()
