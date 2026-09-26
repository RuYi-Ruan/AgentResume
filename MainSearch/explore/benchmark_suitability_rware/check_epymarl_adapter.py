"""Check that EPyMARL's Gymma adapter accepts native four-agent RWARE."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "deps"))
sys.path.insert(0, str(HERE.parent / "benchmark_suitability_smaclite" / "epymarl_ref" / "src"))

import numpy as np  # noqa: E402
from envs.gymma import GymmaWrapper  # noqa: E402


def main():
    env = GymmaWrapper(
        key="rware:rware-tiny-4ag-v2",
        time_limit=500,
        pretrained_wrapper=None,
        seed=0,
        common_reward=True,
        reward_scalarisation="sum",
    )
    rng = np.random.default_rng(2026)
    try:
        obs, _ = env.reset(seed=0)
        assert env.n_agents == 4 and len(obs) == 4
        total_reward = 0.0
        for step in range(1, 501):
            actions = [int(rng.integers(0, env.get_total_actions())) for _ in range(4)]
            obs, reward, terminated, truncated, _ = env.step(actions)
            assert len(obs) == 4 and isinstance(reward, float)
            total_reward += reward
            if terminated or truncated:
                break
        result = {
            "environment": "rware-tiny-4ag-v2",
            "adapter": "EPyMARL GymmaWrapper",
            "num_agents": env.n_agents,
            "observation_size": env.get_obs_size(),
            "state_size": env.get_state_size(),
            "action_count": env.get_total_actions(),
            "steps": step,
            "team_deliveries": int(total_reward),
        }
        print(result, flush=True)
        output = HERE / "results" / "adapter_smoke.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
