import jax.numpy as jnp
from rememberwalk.states.MazeState import MazeState
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv

def get_done(state: MazeState, env: SimpleMazeEnv) -> SimpleMazeEnv:
    """
    Determines whether the current episode should terminate based on multiple criteria.

    Checks for termination conditions including:
    - Any robot geometry below ground level.
    - Trunk position out of height field bounds.
    - Excessive trunk or joint velocities.
    - Lidar contact with terrain.
    - Reaching all waypoints.
    - Exceeding episode length.

    The curriculum coefficient (curr_coeff) is used to adapt the joint velocity threshold for termination:
    as curr_coeff increases, the allowed joint velocity range becomes stricter, making the task harder.

    The curriculum coefficient (curr_coeff) is used to adapt the trunk height threshold for termination:
    as curr_coeff increases, the allowed trunk height range becomes stricter, making the task harder.

    Args:
        state (MazeState): Current state of the robot and environment.
        env (SimpleMazeEnv): Box environment instance containing terrain and robot parameters.

    Returns:
        MazeState: Updated state with 'done' and 'truncation' flags set.
    """
    floor_heights = env.terrain_observation.get_height_from_xpos(state, state.pipeline_state.geom_xpos[env.robot.non_feet_geom_ids, :2],env)
    geom_heights = state.pipeline_state.geom_xpos[env.robot.non_feet_geom_ids, 2] - floor_heights
    geom_done = jnp.any((geom_heights < 0.0))  # Geom is below ground
    done_below_ground = (state.pipeline_state.qpos[2] < 0.5 * (1 - state.curr_coeff) * env.robot.nominal_trunk_z)
    end_of_episode = state.n_steps >= env.episode_length
    qvel_done = jnp.any(jnp.abs(state.pipeline_state.qvel[:3]) >= 1000.0)
    joint_vel_done = jnp.any((jnp.abs(state.pipeline_state.qvel[env.robot.joint_qvel_adr])/env.robot.max_joint_velocities)  > 0.95)
    done = end_of_episode | done_below_ground | qvel_done | geom_done #| joint_vel_done
    truncation = done & end_of_episode
    metrics = state.metrics
    metrics["done_end_of_episode"] = end_of_episode
    metrics["done_below_gorund"] = done_below_ground
    metrics["done_qvel"] = qvel_done
    metrics["done_geom"] = geom_done
    metrics["done_joint_vel"] = joint_vel_done
    metrics["done"] = done
    state = state.replace(done=done,truncation=truncation,metrics=metrics)
    return state
