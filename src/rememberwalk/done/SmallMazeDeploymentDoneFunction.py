"""Termination conditions of the maze task (``imports.done_fn``)."""

import jax.numpy as jnp

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState


def get_done(state: MazeState, env: SimpleMazeEnv) -> MazeState:
    """Sets ``done`` and ``truncation`` of one env.

    An episode ends when:
    - a non-foot geom is below the terrain surface,
    - the trunk is lower than ``0.5 * (1 - curr_coeff) * nominal_trunk_z`` (stricter with the curriculum),
    - a base velocity component reaches 1000 (simulation blew up),
    - ``episode_length`` steps are reached (this one is a truncation, not a termination).

    ``done_joint_vel`` (a joint above 95 % of its velocity limit) is only logged as a metric.

    Args:
        state (MazeState): Current state of the robot and environment.
        env (SimpleMazeEnv): Box environment instance containing terrain and robot parameters.

    Returns:
        MazeState: Updated state with 'done' and 'truncation' flags set.
    """
    floor_heights = env.terrain_observation.get_height_from_xpos(
        state, state.pipeline_state.geom_xpos[env.robot.non_feet_geom_ids, :2], env
    )
    geom_heights = state.pipeline_state.geom_xpos[env.robot.non_feet_geom_ids, 2] - floor_heights
    geom_done = jnp.any(geom_heights < 0.0)  # Geom is below ground
    done_below_ground = (
        state.pipeline_state.qpos[2] < 0.5 * (1 - state.curr_coeff) * env.robot.nominal_trunk_z
    )
    end_of_episode = state.n_steps >= env.episode_length
    qvel_done = jnp.any(jnp.abs(state.pipeline_state.qvel[:3]) >= 1000.0)
    joint_vel_done = jnp.any(
        (jnp.abs(state.pipeline_state.qvel[env.robot.joint_qvel_adr]) / env.robot.max_joint_velocities) > 0.95
    )
    done = end_of_episode | done_below_ground | qvel_done | geom_done
    truncation = done & end_of_episode
    metrics = state.metrics
    metrics["done_end_of_episode"] = end_of_episode
    metrics["done_below_gorund"] = done_below_ground
    metrics["done_qvel"] = qvel_done
    metrics["done_geom"] = geom_done
    metrics["done_joint_vel"] = joint_vel_done
    metrics["done"] = done
    state = state.replace(done=done, truncation=truncation, metrics=metrics)
    return state
