"""Deterministic rollout runner + per-tick logging + capability metrics."""
from __future__ import annotations

from ocres.grid import World
from ocres.recipes import agent_held, pot_kinds

STAY = (0, 0)


def run_episode(world, agents, horizon=400, verbose=False):
    """Run one deterministic episode; returns logs + metrics.

    agents: list of two objects with .action(state) -> (action, intent, target).
    """
    env, grid = world.env, world.grid
    logs = []
    prev_stay0 = prev_stay1 = 0
    stuck0 = stuck1 = 0
    deliveries = 0
    rewards = []
    soup_delivered_at = []
    last_delivery_t = None

    while env.state.timestep < horizon and not env.is_done():
        s = env.state
        acts, ints, targs, infos = [], [], [], []
        for i, ag in enumerate(agents):
            res = ag.action(s, deliveries)
            a, it, tg = res[0], res[1], res[2]
            info = res[3] if len(res) > 3 else None
            acts.append(a)
            ints.append(it)
            targs.append(tg)
            infos.append(info)
        s2, r, done, _ = env.step(tuple(a if a is not None else STAY for a in acts))
        rewards.append(r)
        if r > 0:
            deliveries += 1
            if last_delivery_t is not None:
                soup_delivered_at.append(env.state.timestep - last_delivery_t)
            last_delivery_t = env.state.timestep
        # stuck: consecutive stays
        prev_stay0 = prev_stay0 + 1 if acts[0] == STAY else 0
        prev_stay1 = prev_stay1 + 1 if acts[1] == STAY else 0
        stuck0 = max(stuck0, prev_stay0)
        stuck1 = max(stuck1, prev_stay1)
        logs.append(
            {
                "t": s.timestep,
                "a": [acts[0], acts[1]],
                "intent0": ints[0],
                "target0": targs[0],
                "intent1": ints[1],
                "target1": targs[1],
                "p0": s.players[0].position,
                "p1": s.players[1].position,
                "held0": agent_held(s, 0),
                "held1": agent_held(s, 1),
                "pot": str(sorted((p, k) for p, k in pot_kinds(s, grid).items())),
                "r": r,
                "info0": infos[0],
                "info1": infos[1],
            }
        )
        if verbose and r > 0:
            print(f"  delivery at t={s.timestep}")
    total = sum(rewards)
    gap = soup_delivered_at
    metrics = {
        "steps": env.state.timestep,
        "deliveries": deliveries,
        "reward": total,
        "per_delivery_gap": gap,
        "max_stay0": stuck0,
        "max_stay1": stuck1,
        "mean_gap": (sum(gap) / len(gap)) if gap else None,
    }
    return logs, metrics
