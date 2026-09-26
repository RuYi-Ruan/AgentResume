import unittest

import gymnasium as gym
import lbforaging  # noqa: F401
import numpy as np

from MainSearch.explore.lbf_gradual_learning_gate.run_experiment import (
    Config,
    adjacent_slots,
    epsilon_at,
    execute_aligned_episode,
    food_positions,
    plan_joint_path,
    run_experiment,
)


class LBFGradualLearningGateTest(unittest.TestCase):
    def test_epsilon_decreases_smoothly(self) -> None:
        config = Config(seeds=1, episodes=20, epsilon_decay_episodes=10)
        values = [epsilon_at(index, config) for index in range(12)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))
        self.assertAlmostEqual(values[-1], config.epsilon_end)

    def test_three_agent_environment_and_plan(self) -> None:
        config = Config(seeds=1, episodes=1)
        env = gym.make(config.environment_id, disable_env_checker=True)
        try:
            env.reset(seed=11)
            unwrapped = env.unwrapped
            foods = food_positions(unwrapped.field)
            self.assertEqual(len(unwrapped.players), 3)
            self.assertEqual(len(foods), 3)
            target = foods[0]
            self.assertGreaterEqual(len(adjacent_slots(target, unwrapped.field)), 3)
            plan = plan_joint_path(
                [tuple(player.position) for player in unwrapped.players],
                target,
                unwrapped.field,
            )
            self.assertIsNotNone(plan)
        finally:
            env.close()

    def test_aligned_agents_collect_real_food(self) -> None:
        config = Config(seeds=1, episodes=1)
        env = gym.make(config.environment_id, disable_env_checker=True)
        try:
            env.reset(seed=23)
            result = execute_aligned_episode(env, 0, config)
            self.assertEqual(result["planner_ok"], 1)
            self.assertEqual(result["collected"], 1)
            self.assertGreater(result["reward"], 0)
        finally:
            env.close()

    def test_smoke_run(self) -> None:
        config = Config(
            seeds=2,
            episodes=30,
            window=10,
            epsilon_decay_episodes=25,
        )
        windows, summary = run_experiment(config)
        self.assertEqual(len(windows), 3)
        self.assertEqual(summary["episodes"], 60)
        self.assertIn("gate_pass", summary)


if __name__ == "__main__":
    unittest.main()
