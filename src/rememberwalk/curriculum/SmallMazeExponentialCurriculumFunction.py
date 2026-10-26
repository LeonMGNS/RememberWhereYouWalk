"""Exponential curriculum of the maze task (``imports.update_curriculum_fn``)."""

import jax.numpy as jnp

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState

# Fraction of maze cells an episode must have visited to raise the curriculum, and below which it is lowered
GOAL_INCREASE_EXPLORED = 0.8
GOAL_DECREASE_EXPLORED = 0.4
NO_GOAL_INCREASE_EXPLORED = 0.6
NO_GOAL_DECREASE_EXPLORED = 0.3
# Episodes must last longer than this fraction of episode_length to raise it; episodes shorter than
# DECREASE_EPISODE_FRACTION lower it
INCREASE_EPISODE_FRACTION = 0.9
DECREASE_EPISODE_FRACTION = 0.6
# The terrain curriculum only rises if the sampled terrain level was at least this fraction of the current one
TERRAIN_SAMPLED_FRACTION = 0.9
# A failed episode lowers a level (only while it is below DECREASE_BELOW) by DECREASE_STEP * curr_coeff_delta
DECREASE_STEP = -0.25
DECREASE_BELOW = 0.95


def update_curriculum(state: MazeState, env: SimpleMazeEnv) -> MazeState:
    """Updates the task (``curr_coeff``) and terrain (``terrain_curr_coeff``) curriculum of one env.

    Only done envs change. A level rises after a long episode that visited enough maze cells, by
    ``exponential_*_curr_coeff * curr_coeff_delta`` (the step grows with consecutive increases). It falls by
    ``0.25 * curr_coeff_delta`` after a short or poorly explored episode. Both levels stay within
    ``max_delta_curriculum_value`` of the env's allowed level and within the configured bounds.

    Args:
        state (MazeState): Current state of one env.
        env (SimpleMazeEnv): The environment (curriculum bounds and step size).

    Returns:
        MazeState: Updated state with new curriculum coefficients.
    """
    # Compute metrics needed for update
    cells_reached_percentage = jnp.sum(jnp.clip(state.cell_visits, 0, 1)) / (
        state.cell_visits.shape[0] * state.cell_visits.shape[1]
    )

    # Without a goal position the exploration thresholds are lower
    increase_threshold = jnp.where(state.no_goal_position, NO_GOAL_INCREASE_EXPLORED, GOAL_INCREASE_EXPLORED)
    decrease_threshold = jnp.where(state.no_goal_position, NO_GOAL_DECREASE_EXPLORED, GOAL_DECREASE_EXPLORED)
    should_increase_curr_coeff = (cells_reached_percentage >= increase_threshold) & (
        state.n_steps > INCREASE_EPISODE_FRACTION * env.episode_length
    )
    should_decrease_curr_coeff = (cells_reached_percentage < decrease_threshold) | (
        state.n_steps < DECREASE_EPISODE_FRACTION * env.episode_length
    )
    # The terrain level only rises if the episode was actually sampled close to it
    should_increase_terrain_curr_coeff = should_increase_curr_coeff & (
        state.sampled_terrain_curr_coeff >= TERRAIN_SAMPLED_FRACTION * state.terrain_curr_coeff
    )
    should_decrease_terrain_curr_coeff = should_decrease_curr_coeff

    # Update Curr Coeff
    curr_coeff = jnp.clip(
        state.curr_coeff
        + jnp.where(should_increase_curr_coeff, state.exponential_curr_coeff * env.curr_coeff_delta, 0.0),
        env.curr_coeff,
        env.max_curr_coeff,
    )
    curr_coeff = jnp.clip(
        curr_coeff
        + jnp.where(
            should_decrease_curr_coeff & (state.curr_coeff < DECREASE_BELOW),
            DECREASE_STEP * env.curr_coeff_delta,
            0.0,
        ),
        env.curr_coeff,
        env.max_curr_coeff,
    )

    # Update terrain Curr Coeff
    terrain_curr_coeff = jnp.clip(
        state.terrain_curr_coeff
        + jnp.where(
            should_increase_terrain_curr_coeff,
            state.exponential_terrain_curr_coeff * env.curr_coeff_delta,
            0.0,
        ),
        env.terrain_curr_coeff,
        env.max_terrain_curr_coeff,
    )
    terrain_curr_coeff = jnp.clip(
        terrain_curr_coeff
        + jnp.where(
            should_decrease_terrain_curr_coeff & (state.terrain_curr_coeff < DECREASE_BELOW),
            DECREASE_STEP * env.curr_coeff_delta,
            0.0,
        ),
        env.terrain_curr_coeff,
        env.max_terrain_curr_coeff,
    )

    # Clip to allowed max range
    curr_coeff = jnp.clip(
        curr_coeff,
        state.max_allowed_curr_coeff - env.parameters["max_delta_curriculum_value"],
        state.max_allowed_curr_coeff + env.parameters["max_delta_curriculum_value"],
    )
    curr_coeff = jnp.where(curr_coeff < env.curr_coeff, env.curr_coeff, curr_coeff)
    terrain_curr_coeff = jnp.clip(
        terrain_curr_coeff,
        state.max_allowed_terrain_curr_coeff - env.parameters["max_delta_curriculum_value"],
        state.max_allowed_terrain_curr_coeff + env.parameters["max_delta_curriculum_value"],
    )
    terrain_curr_coeff = jnp.where(
        terrain_curr_coeff < env.terrain_curr_coeff, env.terrain_curr_coeff, terrain_curr_coeff
    )
    # Write curriculum coefficients
    curr_coeff = jnp.where(state.done, curr_coeff, state.curr_coeff)
    terrain_curr_coeff = jnp.where(state.done, terrain_curr_coeff, state.terrain_curr_coeff)

    # Update exponential curr coeffs
    # curr_coeff/terrain_curr_coeff above are already gated by state.done (unchanged when not done),
    # so comparing against the previous value implicitly restricts these updates to done episodes.
    curr_coeff_increased = curr_coeff > state.curr_coeff
    curr_coeff_decreased = curr_coeff < state.curr_coeff
    exponential_curr_coeff = jnp.where(
        curr_coeff_increased,
        state.exponential_curr_coeff + 1.0,
        jnp.where(curr_coeff_decreased, 1.0, state.exponential_curr_coeff),
    )

    terrain_curr_coeff_increased = terrain_curr_coeff > state.terrain_curr_coeff
    terrain_curr_coeff_decreased = terrain_curr_coeff < state.terrain_curr_coeff
    exponential_terrain_curr_coeff = jnp.where(
        terrain_curr_coeff_increased,
        state.exponential_terrain_curr_coeff + 1.0,
        jnp.where(terrain_curr_coeff_decreased, 1.0, state.exponential_terrain_curr_coeff),
    )

    state = state.replace(
        curr_coeff=curr_coeff,
        terrain_curr_coeff=terrain_curr_coeff,
        exponential_curr_coeff=exponential_curr_coeff,
        exponential_terrain_curr_coeff=exponential_terrain_curr_coeff,
    )

    return state
