"""Read-only rollout diagnosis for the five non-shared QMIX policies.

This evaluates each saved checkpoint in separate episodes. It never switches
policies within an episode and never updates a model.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import numpy as np
import smaclite  # noqa: F401
import torch


HERE = Path(__file__).resolve().parent
SRC = HERE / "epymarl_ref" / "src"
sys.path.insert(0, str(SRC))
from modules.agents.rnn_ns_agent import RNNNSAgent  # noqa: E402

RUN_DIR = HERE / "epymarl_ref" / "results" / "sacred" / "qmix_ns" / "2s3z" / "1"
MODEL_ROOT = HERE / "qmix_ns_300k_seed0" / "models" / "qmix_ns_300k_seed0_2s3z_2026-09-24_00-46-38"
STEPS = (100325, 160424, 200453, 220460, 240497, 260503, 280546, 300017)


def load_policy(step, obs_size, n_actions, n_agents):
    args = SimpleNamespace(n_agents=n_agents, n_actions=n_actions, hidden_dim=64, use_rnn=False)
    model = RNNNSAgent(obs_size, args)
    weights = torch.load(MODEL_ROOT / str(step) / "agent.th", map_location="cpu", weights_only=True)
    model.load_state_dict(weights)
    model.eval()
    return model


@torch.no_grad()
def choose(model, obs, avail):
    x = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    hidden = model.init_hidden().unsqueeze(0)
    q, _ = model(x, hidden)
    q = q.masked_fill(torch.as_tensor(np.asarray(avail) == 0), -torch.inf)
    return q.argmax(dim=1).numpy().astype(np.int16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--output", type=Path, default=HERE / "qmix_ns_checkpoint_diagnostic")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    torch.set_num_threads(1)
    metrics = json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8"))
    logged_wins = dict(zip(metrics["test_battle_won_mean"]["steps"], metrics["test_battle_won_mean"]["values"]))
    env = TimeLimit(gym.make("smaclite/2s3z-v0", seed=0, use_cpp_rvo2=False), max_episode_steps=150)
    base = env.unwrapped
    n_agents, obs_size, n_actions = base.n_agents, base.obs_size, base.n_actions
    rows = {"obs": [], "avail": [], "actions": [], "alive": [], "step": [], "seed": [], "time": []}
    summaries = []
    models = {}
    try:
        for step in args.steps:
            model = load_policy(step, obs_size, n_actions, n_agents)
            models[step] = model
            wins, rewards, lengths = [], [], []
            counts = np.zeros((n_agents, n_actions), dtype=np.int64)
            survived = np.zeros(n_agents, dtype=np.int64)
            for seed in range(args.episodes):
                env.reset(seed=seed)
                total = 0.0
                for t in range(150):
                    obs = np.asarray(base.get_obs(), dtype=np.float32)
                    avail = np.asarray(base.get_avail_actions(), dtype=np.int8)
                    alive = np.asarray([i in base.agents for i in range(n_agents)], dtype=np.int8)
                    actions = choose(model, obs, avail)
                    for i in range(n_agents):
                        if alive[i]:
                            counts[i, actions[i]] += 1
                    for name, value in (("obs", obs), ("avail", avail), ("actions", actions), ("alive", alive)):
                        rows[name].append(value)
                    rows["step"].append(step)
                    rows["seed"].append(seed)
                    rows["time"].append(t)
                    _, reward, done, truncated, info = env.step(actions.tolist())
                    total += float(reward)
                    if done or truncated:
                        break
                wins.append(int(info["battle_won"]))
                rewards.append(total)
                lengths.append(t + 1)
                survived += np.asarray([i in base.agents for i in range(n_agents)], dtype=np.int64)
            win_rate = float(np.mean(wins))
            reference = logged_wins.get(step)
            if args.episodes == 50 and reference is not None and abs(win_rate - reference) > 1e-9:
                raise RuntimeError(f"rollout mismatch at {step}: reproduced {win_rate}, training log {reference}")
            per_agent = []
            for i in range(n_agents):
                n = int(counts[i].sum())
                per_agent.append({
                    "id": i, "alive_decisions": n,
                    "stop_pct": round(100 * counts[i, 1] / n, 2),
                    "move_pct": round(100 * counts[i, 2:6].sum() / n, 2),
                    "attack_pct": round(100 * counts[i, 6:].sum() / n, 2),
                    "survived_episodes": int(survived[i]),
                })
            summaries.append({"step": step, "games": args.episodes, "wins": int(sum(wins)),
                              "mean_return": round(float(np.mean(rewards)), 4),
                              "mean_length": round(float(np.mean(lengths)), 3), "agents": per_agent})
            print(f"{step}: wins {sum(wins)}/{args.episodes}; return {np.mean(rewards):.3f}", flush=True)
    finally:
        env.close()

    packed = {name: np.asarray(values) for name, values in rows.items()}
    switch = []
    for previous, current in zip(args.steps[:-1], args.steps[1:]):
        per_agent = []
        for i in range(n_agents):
            # Same pooled local observations and valid-action masks for both policies.
            ids = np.flatnonzero(packed["alive"][:, i] == 1)
            if len(ids) > 1600:
                ids = ids[np.linspace(0, len(ids) - 1, 1600, dtype=int)]
            obs = packed["obs"][ids, i]
            avail = packed["avail"][ids, i]
            before = policy_actions(models[previous].agents[i], obs, avail)
            after = policy_actions(models[current].agents[i], obs, avail)
            per_agent.append(round(float(np.mean(before != after)), 4))
        switch.append({"from": previous, "to": current, "action_switch_rate_by_agent": per_agent})

    result = {"status": "completed_offline_diagnostic", "episodes_per_checkpoint": args.episodes,
              "checkpoints": summaries, "fixed_observation_switches": switch,
              "note": "Checkpoints evaluated in separate episodes; no policy changes within an episode. Action fractions exclude dead agents. Fixed-observation switches use pooled states, not partner-intent labels."}
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "trajectories.npz", **packed)
    print(f"saved: {args.output}", flush=True)


@torch.no_grad()
def policy_actions(agent, obs, avail):
    answers = []
    for start in range(0, len(obs), 256):
        x = torch.as_tensor(obs[start:start + 256], dtype=torch.float32)
        h = agent.init_hidden().expand(len(x), -1)
        q, _ = agent(x, h)
        q = q.masked_fill(torch.as_tensor(avail[start:start + 256] == 0), -torch.inf)
        answers.append(q.argmax(dim=1).numpy())
    return np.concatenate(answers)


if __name__ == "__main__":
    main()
