"""Development gate: three learned primitive-action policies in LBF."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import lbforaging  # noqa: F401
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    seed: int = 0
    updates: int = 60
    episodes_per_update: int = 24
    eval_episodes: int = 100
    eval_interval: int = 10
    ppo_epochs: int = 4
    minibatch: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    shaping_coeff: float = 0.05
    environment_id: str = "Foraging-5x5-3p-2f-coop-v3"


class Actor(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 6),
        )

    def forward(self, obs):
        return torch.distributions.Categorical(logits=self.net(obs))


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * 3, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, obs):
        return self.net(obs.flatten(start_dim=-2)).squeeze(-1)


def make_env(config):
    return gym.make(config.environment_id, disable_env_checker=True)


def state_tensor(observations):
    return torch.as_tensor(np.stack(observations), dtype=torch.float32) / 5.0


def potential(env):
    world = env.unwrapped
    foods = np.argwhere(world.field > 0)
    if len(foods) == 0:
        return 0.0
    positions = np.asarray([player.position for player in world.players])
    total_distances = np.abs(foods[:, None, :] - positions[None, :, :]).sum(axis=(1, 2))
    return -float(total_distances.min())


def play_training_episode(env, actors, critic, config, reset_seed):
    obs, _ = env.reset(seed=reset_seed)
    phi = potential(env)
    rows = []
    raw_total = 0.0
    while True:
        state = state_tensor(obs)
        with torch.no_grad():
            dists = [actor(state[index]) for index, actor in enumerate(actors)]
            actions = torch.stack([dist.sample() for dist in dists])
            logp = sum(dist.log_prob(actions[index]) for index, dist in enumerate(dists))
            value = critic(state.unsqueeze(0))[0]
        next_obs, rewards, terminated, truncated, _ = env.step(tuple(int(x) for x in actions))
        done = bool(terminated or truncated)
        next_phi = 0.0 if done else potential(env)
        raw_reward = float(np.sum(rewards))
        shaped_reward = raw_reward + config.shaping_coeff * (config.gamma * next_phi - phi)
        rows.append((state, actions, logp, value, shaped_reward))
        raw_total += raw_reward
        obs, phi = next_obs, next_phi
        if done:
            break
    advantages = []
    running = 0.0
    next_value = 0.0
    for state, actions, logp, value, reward in reversed(rows):
        running = reward + config.gamma * next_value - float(value) + config.gamma * config.gae_lambda * running
        advantages.append(running)
        next_value = float(value)
    advantages.reverse()
    return rows, advantages, raw_total


def update(actors, critic, optimizer, data, config):
    obs = torch.stack([row[0] for row in data])
    actions = torch.stack([row[1] for row in data])
    old_logp = torch.stack([row[2] for row in data]).detach()
    old_value = torch.stack([row[3] for row in data]).detach()
    advantage = torch.as_tensor([row[5] for row in data], dtype=torch.float32)
    returns = advantage + old_value
    normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    for _ in range(config.ppo_epochs):
        for indices in torch.randperm(len(data)).split(config.minibatch):
            dists = [actor(obs[indices, agent]) for agent, actor in enumerate(actors)]
            logp = sum(dist.log_prob(actions[indices, agent]) for agent, dist in enumerate(dists))
            entropy = sum(dist.entropy() for dist in dists)
            ratio = (logp - old_logp[indices]).exp()
            surrogate = torch.minimum(
                ratio * normalized_advantage[indices],
                ratio.clamp(0.8, 1.2) * normalized_advantage[indices],
            )
            value = critic(obs[indices])
            loss = -surrogate.mean() + 0.5 * nn.functional.mse_loss(value, returns[indices]) - 0.005 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(critic.parameters()) + [p for actor in actors for p in actor.parameters()], 0.5)
            optimizer.step()


def evaluate(config, actors, update_number):
    env = make_env(config)
    totals = []
    successes = 0
    try:
        for episode in range(config.eval_episodes):
            obs, _ = env.reset(seed=70_000 + config.seed * 10_000 + episode)
            total = 0.0
            while True:
                state = state_tensor(obs)
                with torch.no_grad():
                    actions = tuple(int(actor(state[index]).probs.argmax()) for index, actor in enumerate(actors))
                obs, rewards, terminated, truncated, _ = env.step(actions)
                total += float(np.sum(rewards))
                if terminated or truncated:
                    break
            totals.append(total)
            successes += total > 0
    finally:
        env.close()
    return {
        "checkpoint": update_number,
        "mean_raw_team_reward": float(np.mean(totals)),
        "positive_episodes": int(successes),
        "episodes": config.eval_episodes,
    }


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=24)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "development_seed0")
    args = parser.parse_args()
    config = Config(
        seed=args.seed,
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
    )
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    env = make_env(config)
    initial_obs, _ = env.reset(seed=config.seed)
    actors = [Actor(len(initial_obs[0])) for _ in range(3)]
    critic = Critic(len(initial_obs[0]))
    optimizer = torch.optim.Adam(
        list(critic.parameters()) + [p for actor in actors for p in actor.parameters()],
        lr=config.learning_rate,
    )
    evaluations = [evaluate(config, actors, 0)]
    training = []
    try:
        for update_number in range(1, config.updates + 1):
            data = []
            raw_returns = []
            environment_steps = 0
            for episode in range(config.episodes_per_update):
                index = (update_number - 1) * config.episodes_per_update + episode
                rows, advantages, raw_total = play_training_episode(
                    env, actors, critic, config, config.seed * 1_000_000 + index
                )
                data.extend((*row, advantage) for row, advantage in zip(rows, advantages))
                raw_returns.append(raw_total)
                environment_steps += len(rows)
            update(actors, critic, optimizer, data, config)
            training.append({
                "update": update_number,
                "episodes_seen": update_number * config.episodes_per_update,
                "environment_steps_in_update": environment_steps,
                "mean_sampled_raw_reward": float(np.mean(raw_returns)),
                "positive_sampled_episodes": int(np.count_nonzero(np.asarray(raw_returns) > 0)),
            })
            if update_number % config.eval_interval == 0:
                result = evaluate(config, actors, update_number)
                evaluations.append(result)
                print(f"update {update_number}: fixed eval {result['mean_raw_team_reward']:.3f}, positive {result['positive_episodes']}/{config.eval_episodes}", flush=True)
    finally:
        env.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(args.output_dir / "training.csv", training)
    save_csv(args.output_dir / "evaluations.csv", evaluations)
    torch.save([actor.state_dict() for actor in actors], args.output_dir / "actors.pt")
    summary = {
        "status": "development_not_etm",
        "config": asdict(config),
        "evaluations": evaluations,
        "initial_reward": evaluations[0]["mean_raw_team_reward"],
        "final_reward": evaluations[-1]["mean_raw_team_reward"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
