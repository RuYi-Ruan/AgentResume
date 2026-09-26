"""Check ID channels reveal exactly the visible other-agent positions."""

import jax
import jax.numpy as jnp
import jaxmarl

from identity_obs import add_identity_channels
from jaxmarl.environments.overcooked_v2.layouts import Layout, grounded_coord_ring


def main():
    layout = Layout.from_string(
        grounded_coord_ring.replace("W       W", "W   A   W", 1),
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    env = jaxmarl.make("overcooked_v2", layout=layout, agent_view_size=2, random_agent_positions=True)
    other_position_channel = 7 + env.layout.num_ingredients
    checked = 0
    for seed in range(10):
        key = jax.random.PRNGKey(seed)
        obs, state = env.reset(key)
        for _ in range(20):
            encoded = add_identity_channels(obs, state)
            assert encoded.shape == (3, 5, 5, 43)
            # The first channel in the default "other agents" block marks their positions.
            # Appended channels 40:43 must reproduce that mask exactly.
            if not bool(jnp.array_equal(encoded[..., other_position_channel], encoded[..., 40:43].sum(-1))):
                print("positions:", state.agents.pos)
                print("default other-position sum:", float(encoded[..., other_position_channel].sum()))
                print("ID channel sums:", [float(encoded[..., k].sum()) for k in range(40, 43)])
                raise AssertionError(f"Identity mask mismatch, seed={seed}, step={checked}")
            actions = {agent: jnp.array(0) for agent in env.agents}
            key, step_key = jax.random.split(key)
            obs, state, _, _, _ = env.step(step_key, state, actions)
            checked += 1
    print(f"PASS: {checked} local observations checked across 10 seeds")


if __name__ == "__main__":
    main()
