"""M0 smoke pass 5 (final): custom deterministic agents, double-run determinism,
manual stepping, feature vector, reward sanity."""
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld  # noqa: E402
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv  # noqa: E402
from overcooked_ai_py.mdp.actions import Action  # noqa: E402
from overcooked_ai_py.agents.agent import Agent, AgentPair  # noqa: E402

ACTIONS = Action.ALL_ACTIONS


class CycleAgent(Agent):
    """Deterministic agent: cycles through the action space from a fixed offset."""

    def __init__(self, offset=0):
        super().__init__()
        self.offset = offset
        self.i = 0

    def action(self, state):
        act = ACTIONS[(self.offset + self.i) % len(ACTIONS)]
        self.i += 1
        return act, {"cycle": True}

    def reset(self):
        self.i = 0
        super().reset()


def make_env(seed_state_fn=None):
    return OvercookedEnv.from_mdp(gw, start_state_fn=seed_state_fn, horizon=60)


gw = OvercookedGridworld.from_layout_name("cramped_room")
print("grid:", gw.shape, "| actions:", ACTIONS)

# --- determinism double run with custom agents ---
pair = AgentPair(CycleAgent(0), CycleAgent(3), allow_duplicate_agents=True)


def run():
    env = make_env()
    traj, n_steps, sparse, shaped = env.run_agents(pair)
    rows = [(str(s), a, r) for (s, a, r, done, info) in traj]
    return rows, n_steps, sparse, shaped


r1 = run()
r2 = run()
rows1, n1, sp1, sh1 = r1
rows2, n2, sp2, sh2 = r2
print(f"determinism: identical={rows1 == rows2}, steps={n1}, sparse_reward={sp1}, shaped={sh1}")
print("episode first 4 transitions:")
for s, a, r in rows1[:4]:
    print(f"  a={a} r={r} | {s}")

# --- manual stepping & feature vector ---
env = make_env()
s = env.state
f = env.featurize_state_mdp(s)
print("feature len:", len(f), "first8:", f[:8])
acts = [(0, 0), (0, -1), 'interact', (-1, 0), (0, 0), (0, 0)]  # fixed script
tot = 0.0
for k, a in enumerate(acts):
    s2, r, done, info = env.step((a, (0, 0)))
    tot += r
    print(f"  step{k}: a={a} r={r} done={done} | holding={[p.held_object for p in s2.players]}")
print("manual-step reward total:", tot)
# NOTE: env.proportion_stuck_time(trajectories, agent_idx) is post-hoc on trajectory data; use in M2. 
