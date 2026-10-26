"""Per-episode resets of the maze task (``imports.reset_fn``, ``imports.reset_variables_fn``)."""

import jax
import jax.numpy as jnp

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState


def reset_fn(state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:
    """Resets one env if it is done: domain randomisation, a new maze and start pose, a new command.

    Envs that are not done are returned unchanged (``lax.cond``).
    """
    start_state = state
    state, key = env.reset_randomize_state(state, env, key)
    state, key = env.terrain_generator.reset_start_position(state, env, key)
    state, key = env.command_creator.reset_command(state, env, key)
    state = state.replace(last_action=jnp.zeros_like(state.last_action))
    key, _ = jax.random.split(key)
    state = jax.lax.cond(start_state.done, lambda: state, lambda: start_state)
    return state, key


def reset_variables(state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:
    """Clears the per-episode counters and buffers of a done env at the start of the next step.

    B8: ``use_pred_loc`` is drawn but not used anywhere; the draw is kept so the RNG stream stays the same.
    """
    n_steps = jnp.where(state.done, 0, state.n_steps)
    summed_reward = jnp.where(state.done, 0.0, state.summed_reward)
    done = jnp.where(state.done, jnp.zeros_like(state.done), state.done)
    height_delay = jnp.where(state.done, jnp.zeros_like(state.height_delay), state.height_delay)
    action_delay = jnp.where(state.done, jnp.zeros_like(state.action_delay), state.action_delay)
    lidar_horizontal_pos = jnp.where(
        state.done, jnp.zeros_like(state.lidar_horizontal_pos), state.lidar_horizontal_pos
    )
    cell_visits = jnp.where(
        state.done * jnp.ones_like(state.cell_visits), jnp.zeros_like(state.cell_visits), state.cell_visits
    )
    travelled_distance = jnp.where(
        state.done, jnp.zeros_like(state.travelled_distance), state.travelled_distance
    )
    n_reached_goals = jnp.where(state.done, jnp.zeros_like(state.nr_goals_reached), state.nr_goals_reached)
    max_loc_error = jnp.where(state.done, jnp.zeros_like(state.max_loc_error), state.max_loc_error)
    avg_loc_error = jnp.where(state.done, jnp.zeros_like(state.avg_loc_error), state.avg_loc_error)
    use_pred_loc_new = jax.random.uniform(key) < state.curr_coeff
    key, _ = jax.random.split(key)
    use_pred_loc = jnp.where(state.done, use_pred_loc_new, state.use_pred_loc)
    state = state.replace(
        n_steps=n_steps,
        summed_reward=summed_reward,
        done=done,
        height_delay=height_delay,
        action_delay=action_delay,
        lidar_horizontal_pos=lidar_horizontal_pos,
        cell_visits=cell_visits,
        travelled_distance=travelled_distance,
        nr_goals_reached=n_reached_goals,
        max_loc_error=max_loc_error,
        use_pred_loc=use_pred_loc,
        avg_loc_error=avg_loc_error,
    )
    return state, key
