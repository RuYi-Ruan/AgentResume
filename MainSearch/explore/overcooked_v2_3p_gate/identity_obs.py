"""Add visible teammate-ID channels to JaxMARL's local OvercookedV2 observation.

The default observation sums all other agents into one set of layers. These
extra channels distinguish their identities but never show an out-of-view agent.
"""

import jax.numpy as jnp


def add_identity_channels(obs, state, view_size=2):
    """Return [agent, y, x, channel] for one unbatched three-agent state."""
    agents = ("agent_0", "agent_1", "agent_2")
    if view_size != 2:
        raise ValueError("This exploration encoder is fixed to a 5x5 local view")
    yy = jnp.arange(5)[:, None]
    xx = jnp.arange(5)[None, :]
    positions = state.agents.pos
    encoded = []
    for i, agent in enumerate(agents):
        channels = []
        for j in range(3):
            if i == j:
                channels.append(jnp.zeros((5, 5), dtype=jnp.float32))
            else:
                dy = positions.y[j] - positions.y[i] + view_size
                dx = positions.x[j] - positions.x[i] + view_size
                channels.append(((yy == dy) & (xx == dx)).astype(jnp.float32))
        encoded.append(jnp.concatenate((obs[agent], jnp.stack(channels, axis=-1)), axis=-1))
    return jnp.stack(encoded, axis=0)
