"""M18 approved training and local preflight; no LLM/network imports.

Keeps M17 unchanged. All new episodes use one seed implementation, task samples
are captured before the environment step, and rewards count simultaneous soups.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ocres.generic_agents import LayoutAwareTrainableAgent, ScriptedMacroAgent
from ocres.grid import STAY, World
from ocres.ppo import MacroTrainingEnv
from ocres.recipes import agent_held, pot_kinds
from ocres.trainable import (
    INTENTS, INTENT_ID, MacroIntentPolicy, ObservationSpec,
    encode_log_observation, legal_intent_mask, live_row,
)


def load_config(path="configs/m18_v1.json"):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if len(config["maps"]) != 3 or len(config["training_seeds"]) != 3:
        raise ValueError("M18 requires three maps and three training seeds")
    intervals = sorted(config["seed_offsets"].values())
    if any(a[1] >= b[0] for a, b in zip(intervals, intervals[1:])):
        raise ValueError("episode source intervals overlap")
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def group_spec(config, group):
    if not 0 <= group < 9:
        raise ValueError("group must be 0..8")
    map_index, seed_index = divmod(group, 3)
    name = sorted(config["maps"])[map_index]
    return {
        "group": group, "map": name, "rows": config["maps"][name],
        "training_seed": config["training_seeds"][seed_index],
        "episode_base": 100000 * (group + 1),
    }


def episode_seeds(config, group, partition):
    first, last = config["seed_offsets"][partition]
    base = group_spec(config, group)["episode_base"]
    return list(range(base + first, base + last + 1))


def randomized_world(rows, seed, horizon=700):
    # Overcooked prints a planner message for every environment construction.
    with redirect_stdout(io.StringIO()):
        world = World.make(grid_rows=rows, horizon=horizon)
    rng = random.Random(5000 + int(seed))
    positions = rng.sample(sorted(world.grid.passable), 2)
    for player, position in zip(world.env.state.players, positions):
        player.update_pos_and_or(position, (1, 0))
    world.env.state.timestep = 0
    return world


def observation_spec(rows):
    return ObservationSpec(width=len(rows[0]), height=len(rows),
                           max_pots=sum(row.count("P") for row in rows), version="m18-v1")


class StableWaitingMixin:
    """Do not restart empty-handed monitoring before reaching its waiting spot.

The inherited PRE completion condition is 'any soup is ready'. With an already
ready pot it resets the parking target every tick; a moving partner then makes
the two agents chase alternating parking locations indefinitely.
"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cycle_positions = deque(maxlen=8)
        self.cycle_yields = 0

    def primitive_action(self, state, deliveries):
        self._cycle_positions.append(tuple(state.players[self.me].position))
        span = 6 if self.me == 0 else 8
        positions = list(self._cycle_positions)[-span:]
        oscillating = (len(positions) == span and len(set(positions)) == 2
                       and all(positions[i] == positions[i-2] for i in range(2, span)))
        action = super().primitive_action(state, deliveries)
        if oscillating:
            action = self._yield_action(state)
            self._last_primitive_action = action
            self._yield_cooldown = 3
            self._cycle_positions.clear()
            self.cycle_yields += 1
        return action

    def _goal_satisfied(self, state, deliveries):
        if (self.intent == "PRE" and agent_held(state, self.me) is None
                and any(kind == "ready" for kind in pot_kinds(state, self.grid).values())):
            return tuple(state.players[self.me].position) == self._parking_target
        return super()._goal_satisfied(state, deliveries)


class M18ScriptedAgent(StableWaitingMixin, ScriptedMacroAgent):
    pass


class AlternatingAlice(StableWaitingMixin, LayoutAwareTrainableAgent):
    """Only the waiting-location hint follows the current public delivery count."""

    def begin_intent(self, state, deliveries, choice):
        self.role_hint = "cook" if self.me == deliveries % 2 else "serve"
        super().begin_intent(state, deliveries, choice)


def soup_count(reward):
    count = float(reward) / 20.0
    if count < 0 or not count.is_integer():
        raise ValueError(f"unexpected non-soup sparse reward: {reward}")
    return int(count)


def collect_expert(rows, seed, horizon=700, keep_trace=False):
    world = randomized_world(rows, seed, horizon)
    spec = observation_spec(rows)
    agents = [M18ScriptedAgent(world.grid, 0, "alternate", False, horizon),
              M18ScriptedAgent(world.grid, 1, "alternate", True, horizon)]
    samples, trace, counts = [], [], Counter()
    deliveries = 0
    visible_decisions = 0
    last_id = None
    delivery_ticks = []
    while not world.env.is_done():
        state = world.env.state
        row = live_row(state, world.grid)
        results = [agent.action(state, deliveries) for agent in agents]
        if results[0][3] is not None:
            decision_id = results[0][3]["decision_id"]
            if decision_id == last_id:
                raise AssertionError("duplicate task boundary")
            last_id = decision_id
            label = INTENT_ID[results[0][1]]
            mask = legal_intent_mask(row)
            if not mask[label]:
                raise AssertionError("expert emitted an illegal label")
            samples.append((encode_log_observation(row, deliveries, horizon, spec), label, mask))
            counts[results[0][1]] += 1
            a, b = state.players
            if max(abs(a.position[0]-b.position[0]), abs(a.position[1]-b.position[1])) <= 1:
                visible_decisions += 1
        actions = tuple(result[0] if result[0] is not None else STAY for result in results)
        _, reward, _, _ = world.env.step(actions)
        added = soup_count(reward)
        deliveries += added
        if added:
            delivery_ticks.extend([world.env.state.timestep] * added)
        if keep_trace:
            trace.append({"t": row["t"], "position0": row["p0"], "position1": row["p1"],
                          "held0": row["held0"], "held1": row["held1"], "pots": row["pot"],
                          "actions": actions, "intents": [r[1] for r in results],
                          "targets": [r[2] for r in results], "reward": reward,
                          "decision0": results[0][3]})
    return samples, {"seed": seed, "deliveries": deliveries, "reward": deliveries*20,
                     "steps": world.env.state.timestep, "decision_counts": dict(counts),
                     "visible_task_starts": visible_decisions,
                     "cycle_yields": [agent.cycle_yields for agent in agents],
                     "delivery_ticks": delivery_ticks}, trace


class M18TrainingEnv(MacroTrainingEnv):
    def __init__(self, actor, spec, device, rows, horizon=700, max_goal_ticks=50):
        super().__init__(actor, spec, device, horizon=horizon, max_goal_ticks=max_goal_ticks)
        self.rows = rows

    def reset(self, seed):
        self.world = randomized_world(self.rows, seed, self.horizon)
        self.controller = AlternatingAlice(self.world.grid, 0, self.actor, self.spec,
                                          device=self.device, horizon=self.horizon,
                                          max_goal_ticks=self.max_goal_ticks)
        self.partner = M18ScriptedAgent(self.world.grid, 1, "alternate", True, self.horizon)
        self.deliveries = 0
        return self.controller.observe(self.world.env.state, self.deliveries)

    def step(self, choice):
        self.controller.begin_intent(self.world.env.state, self.deliveries, choice)
        reward_sum, duration = 0.0, 0
        while not self.world.env.is_done():
            state = self.world.env.state
            alice = self.controller.primitive_action(state, self.deliveries)
            partner = self.partner.action(state, self.deliveries)[0]
            _, reward, _, _ = self.world.env.step((alice if alice is not None else STAY,
                                                  partner if partner is not None else STAY))
            duration += 1
            added = soup_count(reward)
            reward_sum += added
            self.deliveries += added
            if self.controller.macro_complete(self.world.env.state, self.deliveries):
                break
        observation, mask = self.controller.observe(self.world.env.state, self.deliveries)
        return observation, mask, reward_sum, self.world.env.is_done(), duration


def train_bc(config, model, samples, device, seed, progress=None):
    if not samples:
        raise ValueError("empty demonstration dataset")
    x = np.stack([row[0] for row in samples])
    y = np.asarray([row[1] for row in samples], dtype=np.int64)
    masks = np.stack([row[2] for row in samples])
    counts = np.bincount(y, minlength=len(INTENTS))
    if np.any(counts == 0):
        raise ValueError("missing expert intents: " + str([i for i, n in zip(INTENTS, counts) if not n]))
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(masks))
    options = config["bc"]
    loader = DataLoader(dataset, batch_size=options["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    weights = torch.tensor(len(y)/(len(INTENTS)*counts), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    losses = []
    for epoch in range(1, options["epochs"]+1):
        model.train()
        total = 0.0
        for obs, target, mask in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(obs.to(device), mask.to(device)), target.to(device))
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(target)
        losses.append(total/len(dataset))
        if progress and (epoch % 10 == 0 or epoch == 1):
            progress({"stage": "bc", "epoch": epoch, "loss": losses[-1]})
    return {"samples": len(y), "counts": dict(zip(INTENTS, counts.tolist())), "epoch_losses": losses}


@torch.no_grad()
def classification_report(model, samples, device):
    matrix = np.zeros((len(INTENTS), len(INTENTS)), dtype=int)
    model.eval()
    for first in range(0, len(samples), 512):
        batch = samples[first:first+512]
        obs = torch.from_numpy(np.stack([r[0] for r in batch])).to(device)
        masks = torch.from_numpy(np.stack([r[2] for r in batch])).to(device)
        pred = model(obs, masks).argmax(1).cpu().numpy()
        for row, guess in zip(batch, pred):
            matrix[row[1], guess] += 1
    return {"labels": list(INTENTS), "confusion": matrix.tolist(),
            "accuracy": float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None}


def geometry_report(rows):
    with redirect_stdout(io.StringIO()):
        world = World.make(grid_rows=rows, horizon=700)
    grid = world.grid
    root = min(grid.passable)
    connected = all(grid.bfs(root, cell) is not None for cell in grid.passable)
    features = grid.pot_locs + grid.onion_locs + grid.dish_locs + grid.serve_locs
    accessible = all(grid.interact_stand_cells(feature) for feature in features)
    return {"connected": connected, "accessible": accessible, "floor": len(grid.passable),
            "pots": len(grid.pot_locs), "onions": len(grid.onion_locs),
            "dishes": len(grid.dish_locs), "serves": len(grid.serve_locs),
            "spec": asdict(observation_spec(rows))}
