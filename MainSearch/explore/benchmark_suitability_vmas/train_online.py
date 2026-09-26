"""Independent PPO gate: can three VMAS agents improve continuously?"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import vmas
from torch import nn


@dataclass(frozen=True)
class Config:
    scenario: str = "transport"
    seed: int = 0
    updates: int = 60
    num_envs: int = 64
    horizon: int = 100
    eval_envs: int = 64
    eval_interval: int = 10
    ppo_epochs: int = 3
    minibatch: int = 512
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95


class Policy(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.mean = nn.Linear(64, 2)
        self.value = nn.Linear(64, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.7))

    def forward(self, obs):
        hidden = self.body(obs)
        return self.mean(hidden), self.value(hidden).squeeze(-1)

    def sample(self, obs):
        mean, value = self(obs)
        dist = torch.distributions.Normal(mean, self.log_std.exp())
        raw = dist.rsample()
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        return action, raw, logp, value

    def logp_value(self, obs, raw):
        mean, value = self(obs)
        dist = torch.distributions.Normal(mean, self.log_std.exp())
        action = raw.tanh()
        logp = (dist.log_prob(raw) - torch.log(1 - action.square() + 1e-6)).sum(-1)
        entropy = dist.entropy().sum(-1)
        return logp, value, entropy


def make_env(config, seed, num_envs):
    kwargs = {"n_agents": 3} if config.scenario == "transport" else {
        "n_blue_agents": 3,
        "n_red_agents": 3,
        "ai_red_agents": True,
        "ai_blue_agents": False,
        "observe_teammates": True,
    }
    return vmas.make_env(
        scenario=config.scenario,
        num_envs=num_envs,
        device="cpu",
        continuous_actions=True,
        max_steps=config.horizon,
        seed=seed,
        **kwargs,
    )


def evaluate(policies, config, checkpoint):
    env = make_env(config, 70_000 + config.seed, config.eval_envs)
    obs = env.reset(seed=70_000 + config.seed)
    total = torch.zeros(config.eval_envs)
    active = torch.ones(config.eval_envs, dtype=torch.bool)
    blue_goals = torch.zeros(config.eval_envs, dtype=torch.bool)
    red_goals = torch.zeros(config.eval_envs, dtype=torch.bool)
    for _ in range(config.horizon):
        with torch.no_grad():
            actions = [policy(item)[0].tanh() for policy, item in zip(policies, obs)]
        obs, rewards, done, _ = env.step(actions)
        total += rewards[0] * active
        if config.scenario == "football":
            sparse = env.scenario._sparse_reward_blue
            blue_goals |= active & (sparse > 0)
            red_goals |= active & (sparse < 0)
        active &= ~done
    if config.scenario == "transport":
        object_pos = env.scenario.packages[0].state.pos
        goal_pos = env.world.landmarks[0].state.pos
    else:
        object_pos = env.scenario.ball.state.pos
        goal_pos = env.scenario.right_goal_pos
    distance = torch.linalg.norm(object_pos - goal_pos, dim=-1)
    return {
        "seed": config.seed,
        "checkpoint": checkpoint,
        "mean_return": float(total.mean()),
        "mean_final_distance": float(distance.mean()),
        "positive_worlds": int((total > 0).sum()),
        "blue_goals": int(blue_goals.sum()),
        "red_goals": int(red_goals.sum()),
        "worlds": config.eval_envs,
    }


def rollout(env, obs, policies, config):
    storage = [dict(obs=[], raw=[], logp=[], value=[]) for _ in policies]
    rewards = []
    dones = []
    for _ in range(config.horizon):
        actions = []
        with torch.no_grad():
            for index, policy in enumerate(policies):
                action, raw, logp, value = policy.sample(obs[index])
                actions.append(action)
                storage[index]["obs"].append(obs[index])
                storage[index]["raw"].append(raw)
                storage[index]["logp"].append(logp)
                storage[index]["value"].append(value)
        obs, reward, done, _ = env.step(actions)
        rewards.append(reward[0].detach())
        dones.append(done.detach().float())
        for world_index in done.nonzero(as_tuple=False).flatten().tolist():
            reset_obs = env.reset_at(world_index)
            for agent_index in range(len(obs)):
                obs[agent_index][world_index] = reset_obs[agent_index][world_index]
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)
    for index, policy in enumerate(policies):
        with torch.no_grad():
            next_value = policy(obs[index])[1]
        for name in storage[index]:
            storage[index][name] = torch.stack(storage[index][name])
        advantages = torch.zeros_like(rewards)
        running = torch.zeros(config.num_envs)
        for tick in reversed(range(config.horizon)):
            future = next_value if tick == config.horizon - 1 else storage[index]["value"][tick + 1]
            mask = 1 - dones[tick]
            delta = rewards[tick] + config.gamma * future * mask - storage[index]["value"][tick]
            running = delta + config.gamma * config.gae_lambda * mask * running
            advantages[tick] = running
        storage[index]["advantage"] = advantages
        storage[index]["return"] = advantages + storage[index]["value"]
    return storage, rewards.sum(0), obs


def update_policies(policies, optimizers, storage, config):
    losses = []
    for policy, optimizer, data in zip(policies, optimizers, storage):
        obs = data["obs"].reshape(-1, data["obs"].shape[-1])
        raw = data["raw"].reshape(-1, 2)
        old_logp = data["logp"].reshape(-1)
        advantages = data["advantage"].reshape(-1)
        returns = data["return"].reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        last_loss = 0.0
        for _ in range(config.ppo_epochs):
            for indices in torch.randperm(len(obs)).split(config.minibatch):
                new_logp, value, entropy = policy.logp_value(obs[indices], raw[indices])
                ratio = (new_logp - old_logp[indices]).exp()
                surrogate = torch.minimum(
                    ratio * advantages[indices],
                    ratio.clamp(0.8, 1.2) * advantages[indices],
                )
                loss = -surrogate.mean() + 0.5 * nn.functional.mse_loss(
                    value, returns[indices]
                ) - 0.005 * entropy.mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
                last_loss = float(loss.detach())
        losses.append(last_loss)
    return losses


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["transport", "football"], default=Config.scenario)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--updates", type=int, default=Config.updates)
    parser.add_argument("--num-envs", type=int, default=Config.num_envs)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--eval-interval", type=int, default=Config.eval_interval)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "learning_development")
    args = parser.parse_args()
    config = Config(
        scenario=args.scenario,
        seed=args.seed,
        updates=args.updates,
        num_envs=args.num_envs,
        horizon=args.horizon,
        eval_interval=args.eval_interval,
    )
    torch.manual_seed(config.seed + 9000)
    np.random.seed(config.seed + 10000)
    env = make_env(config, config.seed, config.num_envs)
    initial_obs = env.reset(seed=config.seed)
    policies = [Policy(item.shape[-1]) for item in initial_obs]
    optimizers = [torch.optim.Adam(policy.parameters(), lr=config.learning_rate) for policy in policies]
    evaluations = [evaluate(policies, config, 0)]
    training = []
    for update in range(config.updates):
        obs = env.reset(seed=config.seed * 100_000 + update)
        storage, total_reward, _ = rollout(env, obs, policies, config)
        losses = update_policies(policies, optimizers, storage, config)
        training.append({
            "seed": config.seed,
            "update": update + 1,
            "environment_steps": (update + 1) * config.num_envs * config.horizon,
            "sampled_mean_return": float(total_reward.mean()),
            **{f"agent_{index}_loss": loss for index, loss in enumerate(losses)},
        })
        if (update + 1) % config.eval_interval == 0:
            result = evaluate(policies, config, update + 1)
            evaluations.append(result)
            print(f"update {update + 1}: eval return {result['mean_return']:.3f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training.csv", training)
    write_csv(args.output_dir / "evaluations.csv", evaluations)
    summary = {
        "status": "development_continuous_learning_gate",
        "claim_boundary": "Three independent PPO policies train atomic actions continuously in VMAS; no ETM yet; learning must be judged by evaluation.",
        "config": asdict(config),
        "evaluations": evaluations,
        "initial_return": evaluations[0]["mean_return"],
        "final_return": evaluations[-1]["mean_return"],
        "gain": evaluations[-1]["mean_return"] - evaluations[0]["mean_return"],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save([policy.state_dict() for policy in policies], args.output_dir / "final_policies.pt")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
