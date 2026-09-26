"""Development-only gate: can three-agent coordination change soup throughput?"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


GRID = (
    "XXXXXXXXXXXXX",
    "XO   P P   OX",
    "X           X",
    "X 1   2   3 X",
    "X           X",
    "XS   D D   SX",
    "XXXXXXXXXXXXX",
)

MOVE = ((0, -1), (0, 1), (1, 0), (-1, 0))
STAY = (0, 0)


@dataclass(frozen=True)
class Config:
    episodes: int = 60
    horizon: int = 500
    seed: int = 20260922


def held_kind(player):
    return None if not player.has_object() else player.get_object().name


class RoleAgent:
    def __init__(self, mdp, index: int, role: str, lane: int = 0):
        self.mdp = mdp
        self.index = index
        self.role = role
        self.lane = lane
        self.passable = set(mdp.get_valid_player_positions())
        self.onions = sorted(mdp.get_onion_dispenser_locations())
        self.pots = sorted(mdp.get_pot_locations())
        self.dishes = sorted(mdp.get_dish_dispenser_locations())
        self.serves = sorted(mdp.get_serving_locations())

    def pot_status(self, state, pot):
        if not state.has_object(pot):
            return "empty"
        soup = state.get_object(pot)
        if soup.is_ready:
            return "ready"
        if soup.is_cooking:
            return "cooking"
        return f"items{len(soup.ingredients)}"

    def target(self, state):
        held = held_kind(state.players[self.index])
        statuses = {pot: self.pot_status(state, pot) for pot in self.pots}
        if self.role == "cook":
            pot = self.pots[min(self.lane, len(self.pots) - 1)]
            status = statuses[pot]
            if held == "onion":
                return pot
            if held is not None:
                return None
            if status in ("empty", "items1", "items2"):
                return self.onions[min(self.lane, len(self.onions) - 1)]
            if status == "items3":
                return pot
            return None
        if held == "soup":
            return min(self.serves, key=lambda p: self.distance(state.players[self.index].position, p))
        ready = [pot for pot, status in statuses.items() if status == "ready"]
        if held == "dish":
            return min(ready, key=lambda p: self.distance(state.players[self.index].position, p)) if ready else None
        if held is None and any(status in ("cooking", "ready") for status in statuses.values()):
            return min(self.dishes, key=lambda p: self.distance(state.players[self.index].position, p))
        return None

    @staticmethod
    def distance(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def bfs(self, start, goals, occupied):
        queue = [start]
        previous = {start: None}
        while queue:
            current = queue.pop(0)
            if current in goals:
                path = []
                while current != start:
                    path.append(current)
                    current = previous[current]
                return path[::-1]
            for delta in MOVE:
                nxt = current[0] + delta[0], current[1] + delta[1]
                if nxt in self.passable and nxt not in occupied and nxt not in previous:
                    previous[nxt] = current
                    queue.append(nxt)
        return None

    def action(self, state):
        target = self.target(state)
        if target is None:
            return STAY
        player = state.players[self.index]
        pos = tuple(player.position)
        anchors = []
        for delta in MOVE:
            stand = target[0] + delta[0], target[1] + delta[1]
            if stand in self.passable:
                face = -delta[0], -delta[1]
                anchors.append((stand, face))
        for stand, face in anchors:
            if pos == stand:
                return Action.INTERACT if tuple(player.orientation) == face else face
        occupied = {tuple(p.position) for i, p in enumerate(state.players) if i != self.index}
        paths = [(self.bfs(pos, {stand}, occupied), stand) for stand, _ in anchors]
        paths = [(path, stand) for path, stand in paths if path]
        if not paths:
            return STAY
        path, _ = min(paths, key=lambda value: (len(value[0]), value[1]))
        nxt = path[0]
        return nxt[0] - pos[0], nxt[1] - pos[1]


def make_env(horizon):
    mdp = OvercookedGridworld.from_grid(
        [list(row) for row in GRID], params_to_overwrite={"layout_name": "three_agent_gap"}
    )
    return mdp, OvercookedEnv.from_mdp(mdp, horizon=horizon)


def run_episode(condition: str, episode_seed: int, horizon: int):
    mdp, env = make_env(horizon)
    env.reset()
    state = env.state
    rng = random.Random(episode_seed)
    # Paired nuisance variation: permute the three starting cells identically across conditions.
    starts = list(mdp.start_player_positions)
    rng.shuffle(starts)
    for player, position in zip(state.players, starts):
        player.update_pos_and_or(position, rng.choice(MOVE))
    roles = (
        (("cook", 1), ("cook", 0), ("serve", 0))
        if condition == "updated_model"
        else (("serve", 0), ("cook", 0), ("serve", 0))
    )
    agents = [RoleAgent(mdp, i, role, lane) for i, (role, lane) in enumerate(roles)]
    reward_total = 0
    for _ in range(horizon):
        actions = tuple(agent.action(env.state) for agent in agents)
        _, reward, done, _ = env.step(actions, joint_agent_action_info=[{} for _ in agents])
        reward_total += reward
        if done:
            break
    return reward_total / 20.0


def summarize(config: Config):
    rows = []
    for episode in range(config.episodes):
        episode_seed = config.seed + episode
        for condition in ("stale_model", "updated_model"):
            rows.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "condition": condition,
                    "soups": run_episode(condition, episode_seed, config.horizon),
                }
            )
    stale = [row["soups"] for row in rows if row["condition"] == "stale_model"]
    updated = [row["soups"] for row in rows if row["condition"] == "updated_model"]
    differences = [new - old for old, new in zip(stale, updated)]
    se = stdev(differences) / math.sqrt(len(differences)) if len(set(differences)) > 1 else 0.0
    summary = {
        "status": "development_score_separation_gate",
        "claim_boundary": "Scripted three-agent role-allocation test, not an ETM effectiveness result.",
        "config": vars(config),
        "conditions": {
            "stale_model": {"roles": ["serve", "cook", "serve"], "mean_soups": mean(stale)},
            "updated_model": {"roles": ["cook", "cook", "serve"], "mean_soups": mean(updated)},
        },
        "paired_difference_soups": {
            "mean": mean(differences),
            "normal_approx_95ci": [mean(differences) - 1.96 * se, mean(differences) + 1.96 * se],
            "updated_better_episodes": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "stale_better_episodes": sum(value < 0 for value in differences),
        },
        "episodes": rows,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--horizon", type=int, default=Config.horizon)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results.json")
    args = parser.parse_args()
    config = Config(episodes=args.episodes, horizon=args.horizon)
    result = summarize(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
