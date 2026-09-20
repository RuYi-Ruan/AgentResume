"""Minimal PPO training utilities for the single trainable Alice policy."""
from __future__ import annotations

import copy
import os
import random

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch
from torch import nn

from ocres.agents import CookAgent
from ocres.data import TWO_POT
from ocres.grid import STAY, World
from ocres.trainable import TrainableMacroAgent


def randomize_start(world, seed):
    rng = random.Random(1000 + int(seed))
    first, second = rng.sample(sorted(world.grid.passable), 2)
    state = world.env.state
    state.players[0].update_pos_and_or(first, (1, 0))
    state.players[1].update_pos_and_or(second, (1, 0))
    state.timestep = 0


class MacroTrainingEnv:
    """Semi-MDP: one external action runs one macro intent to termination."""

    def __init__(self, actor, spec, device, horizon=700, max_goal_ticks=40):
        self.actor = actor
        self.spec = spec
        self.device = device
        self.horizon = horizon
        self.max_goal_ticks = max_goal_ticks
        self.world = self.controller = self.partner = None
        self.deliveries = 0

    def reset(self, seed):
        self.world = World.make(grid_rows=TWO_POT, horizon=self.horizon)
        randomize_start(self.world, seed)
        self.controller = TrainableMacroAgent(
            self.world.grid,
            0,
            self.actor,
            self.spec,
            device=self.device,
            horizon=self.horizon,
            max_goal_ticks=self.max_goal_ticks,
        )
        self.partner = CookAgent(self.world.grid, 1, parallel_after_delay=None)
        self.deliveries = 0
        return self.controller.observe(self.world.env.state, self.deliveries)

    def step(self, choice):
        state = self.world.env.state
        self.controller.begin_intent(state, self.deliveries, choice)
        reward = 0.0
        duration = 0
        while not self.world.env.is_done():
            state = self.world.env.state
            alice_action = self.controller.primitive_action(state, self.deliveries)
            partner_action = self.partner.action(state, self.deliveries)[0]
            _, tick_reward, _, _ = self.world.env.step(
                (alice_action if alice_action is not None else STAY,
                 partner_action if partner_action is not None else STAY)
            )
            duration += 1
            reward += float(tick_reward) / 20.0
            if tick_reward > 0:
                self.deliveries += 1
            if self.controller.macro_complete(self.world.env.state, self.deliveries):
                break
        done = self.world.env.is_done()
        observation, mask = self.controller.observe(self.world.env.state, self.deliveries)
        return observation, mask, reward, done, duration


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation):
        return self.network(observation).squeeze(-1)


def collect_rollouts(actor, critic, spec, device, seeds, horizon, temperature, gamma, gae_lambda, env_factory=None):
    records = []
    episode_metrics = []
    actor.eval()
    critic.eval()
    for seed in seeds:
        env = (
            env_factory(actor, spec, device, horizon)
            if env_factory is not None
            else MacroTrainingEnv(actor, spec, device, horizon=horizon)
        )
        observation, mask = env.reset(seed)
        episode = []
        done = False
        while not done:
            obs_t = torch.from_numpy(observation).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = actor(obs_t, mask_t) / temperature
                distribution = torch.distributions.Categorical(logits=logits)
                action = distribution.sample()
                log_prob = distribution.log_prob(action)
                value = critic(obs_t)
            next_observation, next_mask, reward, done, duration = env.step(int(action.item()))
            episode.append(
                {
                    "observation": observation,
                    "mask": mask,
                    "action": int(action.item()),
                    "old_log_prob": float(log_prob.item()),
                    "value": float(value.item()),
                    "reward": reward,
                    "duration": duration,
                }
            )
            observation, mask = next_observation, next_mask
        advantage = 0.0
        next_value = 0.0
        for transition in reversed(episode):
            discount = gamma ** transition["duration"]
            delta = transition["reward"] + discount * next_value - transition["value"]
            advantage = delta + discount * gae_lambda * advantage
            transition["advantage"] = advantage
            transition["return"] = advantage + transition["value"]
            next_value = transition["value"]
        records.extend(episode)
        episode_metrics.append(
            {"seed": int(seed), "deliveries": env.deliveries, "macro_steps": len(episode)}
        )
    return records, episode_metrics


def ppo_update(
    actor,
    critic,
    optimizer,
    records,
    device,
    temperature=2.5,
    update_epochs=4,
    batch_size=256,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.02,
):
    observations = torch.tensor(np.stack([r["observation"] for r in records]), device=device)
    masks = torch.tensor(np.stack([r["mask"] for r in records]), device=device)
    actions = torch.tensor([r["action"] for r in records], dtype=torch.long, device=device)
    old_log_probs = torch.tensor([r["old_log_prob"] for r in records], device=device)
    returns = torch.tensor([r["return"] for r in records], device=device)
    advantages = torch.tensor([r["advantage"] for r in records], device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    actor.train()
    critic.train()
    stats = []
    for _ in range(update_epochs):
        for indices in torch.randperm(len(records), device=device).split(batch_size):
            distribution = torch.distributions.Categorical(
                logits=actor(observations[indices], masks[indices]) / temperature
            )
            log_probs = distribution.log_prob(actions[indices])
            ratio = torch.exp(log_probs - old_log_probs[indices])
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.square(critic(observations[indices]) - returns[indices]).mean()
            entropy = distribution.entropy().mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            optimizer.step()
            stats.append((float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach())))
    return {
        "policy_loss": float(np.mean([s[0] for s in stats])),
        "value_loss": float(np.mean([s[1] for s in stats])),
        "entropy": float(np.mean([s[2] for s in stats])),
    }


@torch.no_grad()
def evaluate_policy(actor, spec, device, seeds, horizon=700, env_factory=None):
    actor.eval()
    rows = []
    for seed in seeds:
        env = (
            env_factory(actor, spec, device, horizon)
            if env_factory is not None
            else MacroTrainingEnv(actor, spec, device, horizon=horizon)
        )
        observation, mask = env.reset(seed)
        done = False
        macro_steps = 0
        while not done:
            obs_t = torch.from_numpy(observation).unsqueeze(0).to(device)
            mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
            action = int(actor(obs_t, mask_t).argmax(dim=1).item())
            observation, mask, _, done, _ = env.step(action)
            macro_steps += 1
        rows.append({"seed": int(seed), "deliveries": env.deliveries, "macro_steps": macro_steps})
    values = [row["deliveries"] for row in rows]
    return {
        "rows": rows,
        "delivery_mean": float(np.mean(values)),
        "delivery_std": float(np.std(values)),
        "delivery_min": int(min(values)),
        "delivery_max": int(max(values)),
    }


def clone_state_dict(model):
    return copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
