"""Five independently learning PPO agents on official MPE2 Simple Spread."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from mpe2 import simple_spread_v3
from torch import nn


@dataclass
class Config:
    seed: int = 0
    steps: int = 10_000
    num_envs: int = 16
    horizon: int = 25
    eval_interval: int = 10_000
    eval_episodes: int = 64
    epochs: int = 4
    minibatch: int = 200
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2


class Policy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)


def make_env():
    return simple_spread_v3.parallel_env(N=5, local_ratio=0.0, max_cycles=25, continuous_actions=False)


def team_metrics(env, observation, policies, horizon):
    names = env.possible_agents
    total_return = 0.0
    for _ in range(horizon):
        actions = {}
        with torch.no_grad():
            for idx, name in enumerate(names):
                logits, _ = policies[idx](torch.as_tensor(observation[name], dtype=torch.float32))
                actions[name] = int(logits.argmax().item())
        observation, rewards, terminated, truncated, _ = env.step(actions)
        total_return += float(np.mean(list(rewards.values())))
        if all(terminated.get(name, False) or truncated.get(name, False) for name in names):
            break
    world = env.unwrapped.world
    agent_positions = np.stack([a.state.p_pos for a in world.agents])
    landmark_positions = np.stack([l.state.p_pos for l in world.landmarks])
    distances = np.linalg.norm(agent_positions[:, None] - landmark_positions[None], axis=-1)
    nearest = distances.min(axis=0)
    return total_return, float(nearest.mean()), int(np.all(nearest < 0.15))


def evaluate(policies, config, step):
    returns, distances, covers = [], [], []
    env = make_env()
    for episode in range(config.eval_episodes):
        observation, _ = env.reset(seed=500_000 + episode)
        ret, dist, cover = team_metrics(env, observation, policies, config.horizon)
        returns.append(ret)
        distances.append(dist)
        covers.append(cover)
    env.close()
    return {
        "step": step,
        "mean_return": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "mean_nearest_landmark_distance": float(np.mean(distances)),
        "full_cover_episodes": int(sum(covers)),
        "eval_episodes": config.eval_episodes,
    }


def collect(envs, observations, policies, config, update_index):
    names = envs[0].possible_agents
    data = [{key: [] for key in ("obs", "action", "logp", "value", "reward", "done")} for _ in names]
    episode_returns = np.zeros(config.num_envs, dtype=np.float32)
    for tick in range(config.horizon):
        action_matrix = np.zeros((len(names), config.num_envs), dtype=np.int64)
        for idx, name in enumerate(names):
            obs_batch = torch.as_tensor(np.stack([obs[name] for obs in observations]), dtype=torch.float32)
            with torch.no_grad():
                logits, values = policies[idx](obs_batch)
                dist = torch.distributions.Categorical(logits=logits)
                actions = dist.sample()
                logp = dist.log_prob(actions)
            data[idx]["obs"].append(obs_batch)
            data[idx]["action"].append(actions)
            data[idx]["logp"].append(logp)
            data[idx]["value"].append(values)
            action_matrix[idx] = actions.numpy()
        next_observations = []
        reward_batch = np.zeros(config.num_envs, dtype=np.float32)
        done_batch = np.zeros(config.num_envs, dtype=np.float32)
        for env_index, env in enumerate(envs):
            actions = {name: int(action_matrix[idx, env_index]) for idx, name in enumerate(names)}
            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            reward_batch[env_index] = float(np.mean(list(rewards.values())))
            done_batch[env_index] = float(all(terminated.get(name, False) or truncated.get(name, False) for name in names))
            if done_batch[env_index]:
                next_obs, _ = env.reset(seed=config.seed * 10_000_000 + update_index * config.num_envs + env_index)
            next_observations.append(next_obs)
        reward_tensor = torch.as_tensor(reward_batch)
        done_tensor = torch.as_tensor(done_batch)
        episode_returns += reward_batch
        for agent_data in data:
            agent_data["reward"].append(reward_tensor)
            agent_data["done"].append(done_tensor)
        observations = next_observations
    return data, observations, float(episode_returns.mean())


def ppo_update(policies, optimizers, data, config):
    for policy, optimizer, agent_data in zip(policies, optimizers, data):
        arrays = {key: torch.stack(values) for key, values in agent_data.items()}
        advantages = torch.zeros_like(arrays["reward"])
        running = torch.zeros(config.num_envs)
        for tick in reversed(range(config.horizon)):
            next_value = arrays["value"][tick + 1] if tick + 1 < config.horizon else torch.zeros(config.num_envs)
            mask = 1.0 - arrays["done"][tick]
            delta = arrays["reward"][tick] + config.gamma * mask * next_value - arrays["value"][tick]
            running = delta + config.gamma * config.gae_lambda * mask * running
            advantages[tick] = running
        returns = advantages + arrays["value"]
        flat_obs = arrays["obs"].reshape(-1, arrays["obs"].shape[-1])
        flat_actions = arrays["action"].flatten()
        flat_old_logp = arrays["logp"].flatten()
        flat_adv = advantages.flatten()
        flat_returns = returns.flatten()
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        for _ in range(config.epochs):
            for indices in torch.randperm(flat_obs.shape[0]).split(config.minibatch):
                logits, values = policy(flat_obs[indices])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(flat_actions[indices])
                ratio = (logp - flat_old_logp[indices]).exp()
                objective = torch.minimum(
                    ratio * flat_adv[indices],
                    ratio.clamp(1 - config.clip, 1 + config.clip) * flat_adv[indices],
                )
                loss = -objective.mean() + 0.5 * nn.functional.mse_loss(values, flat_returns[indices]) - 0.01 * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = Config(seed=args.seed, steps=args.steps, num_envs=args.num_envs, eval_interval=args.eval_interval, eval_episodes=args.eval_episodes)
    if config.steps % (config.num_envs * config.horizon):
        parser.error("--steps must be divisible by num_envs * horizon (default 400)")
    torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    envs = [make_env() for _ in range(config.num_envs)]
    observations = [env.reset(seed=config.seed * 100_000 + idx)[0] for idx, env in enumerate(envs)]
    names = envs[0].possible_agents
    policies = [Policy(len(observations[0][name]), envs[0].action_space(name).n) for name in names]
    optimizers = [torch.optim.Adam(policy.parameters(), lr=config.learning_rate) for policy in policies]
    start = time.perf_counter()
    evaluations = [evaluate(policies, config, 0)]
    print(f"step=0 eval_return={evaluations[0]['mean_return']:.3f} distance={evaluations[0]['mean_nearest_landmark_distance']:.3f}", flush=True)
    training = []
    updates = config.steps // (config.num_envs * config.horizon)
    for update in range(1, updates + 1):
        samples, observations, mean_return = collect(envs, observations, policies, config, update)
        ppo_update(policies, optimizers, samples, config)
        step = update * config.num_envs * config.horizon
        training.append({"step": step, "sampled_mean_return": mean_return})
        if step % config.eval_interval == 0 or step == config.steps:
            result = evaluate(policies, config, step)
            evaluations.append(result)
            print(f"step={step} eval_return={result['mean_return']:.3f} distance={result['mean_nearest_landmark_distance']:.3f} full_cover={result['full_cover_episodes']}/{config.eval_episodes} elapsed_s={time.perf_counter()-start:.1f}", flush=True)
    for env in envs:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "training.csv", training)
    write_csv(args.output / "evaluations.csv", evaluations)
    torch.save([policy.state_dict() for policy in policies], args.output / "final_policies.pt")
    summary = {"status": "completed", "scope": "development learning gate; no ETM", "config": asdict(config), "elapsed_seconds": time.perf_counter() - start, "evaluations": evaluations}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
