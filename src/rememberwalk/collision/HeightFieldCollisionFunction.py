"""Foot and geom contact with the height field terrain.

Selected by ``imports.update_lasttouch_fn``, ``get_collision_lookup_fn`` and ``get_contact_fn``.
"""

import jax
import jax.numpy as jnp


def get_collision(state, geom_ids: jnp.ndarray, env) -> jnp.ndarray:
    """
    Determines which geometries are in collision with the terrain surface.

    For each geometry specified by geom_ids, compares its z-position to the interpolated terrain height
    at its (x, y) location. Returns a boolean array indicating whether each geometry is in collision
    (i.e., its z-position minus terrain height is less than its size).

    Args:
        state (MazeState): The current environment state.
        geom_ids (jnp.ndarray): Indices of geometries to check for collision.
        env (SimpleMazeEnv): The environment instance.

    Returns:
        jnp.ndarray: Boolean array indicating collision status for each geometry.
    """
    pipeline_state = state.pipeline_state
    geom_pos = pipeline_state.geom_xpos[geom_ids]
    geom_size = state.model.geom_size[geom_ids, 0]
    f_heights = env.terrain_observation.get_height_from_xpos(state, geom_pos[:, :2], env)
    return geom_pos[:, 2] - f_heights < geom_size


def get_contact(contact, geom1, geom2):
    """
    Checks if two specified geometries are in contact based on the contact information.

    Searches the contact object for a contact pair matching geom1 and geom2 (in either order).
    Returns True if the minimum contact distance for the pair is negative (indicating contact).

    Args:
        contact: Contact information object containing geometry pairs and distances.
        geom1: First geometry index.
        geom2: Second geometry index.

    Returns:
        bool: True if the geometries are in contact, False otherwise.
    """
    mask = (jnp.array([geom1, geom2]) == contact.geom).all(axis=1)
    mask |= (jnp.array([geom2, geom1]) == contact.geom).all(axis=1)
    idx = jnp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    return dist < 0


def get_collision_lookup(state, geom_ids: jnp.ndarray, env) -> jnp.ndarray:
    """
    Determine collision/contact for each geometry using the contact lookup.

    Uses the pipeline state's contact pairs and distances to check whether each geometry in geom_ids
    is in contact (negative contact distance) with any other geometry. Returns a boolean array with
    the same ordering as geom_ids: True if a contact with negative distance exists, False otherwise.

    Args:
        state (MazeState): The current environment state.
        geom_ids (jnp.ndarray): Indices of geometries to check for contact.
        env (SimpleMazeEnv): The environment instance (unused but kept for signature parity).

    Returns:
        jnp.ndarray: Boolean array indicating contact/collision status for each geometry.
    """
    contact = state.pipeline_state.contact

    def geom_in_contact(g):
        # True if geom g appears in any active MJX contact
        # mask contacts involving geometry g
        mask = (contact.geom[:, 0] == g) | (contact.geom[:, 1] == g)
        # substitute large positive distance for non-matching contacts and take minimum
        dists = jnp.where(mask, contact.dist, 1e4)
        min_dist = jnp.min(dists)
        return min_dist < 0

    return jax.vmap(geom_in_contact)(geom_ids)


def update_lasttouch(state, geom_ids: jnp.ndarray, env):
    """
    Updates the last touch time for each geometry based on ground contact status.

    For each geometry, checks if it is in collision with the terrain. Updates the last_touch field
    in the state: if the geometry is not in contact, increments its last_touch time by the environment
    time step; otherwise, resets last_touch to zero.

    Args:
        state (MazeState): The current environment state.
        geom_ids (jnp.ndarray): Indices of geometries to update.
        env (SimpleMazeEnv): The environment instance.

    Returns:
        MazeState: Updated state with new last_touch values.
    """
    ground_array = get_collision(state, geom_ids, env)
    last_touch = state.last_touch * ~ground_array + ~ground_array * env.dt
    state = state.replace(last_touch=last_touch)
    return state
