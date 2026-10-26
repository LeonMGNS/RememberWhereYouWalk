"""Command creator of the maze task (``imports.command_handler``).

The file name is historical; the class is ``StaticMazeCommandCreator``.
"""

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as R

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState


class StaticMazeCommandCreator:
    """Command of the maze task: the goal position in the robot's body frame.

    In the exploration task (``no_goal_position``) the command is zero.
    """

    def __init__(self, terrain_parameters):

        self.terrain_parameters = terrain_parameters

    def update_command(
        self, state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray
    ) -> tuple[MazeState, jnp.ndarray]:
        """Writes the goal position in the body frame (x, y) into ``current_target_heading``."""
        xyz = state.pipeline_state.qpos[:3]
        quat = state.pipeline_state.qpos[3:7]
        rot = R.from_quat(
            [
                quat[1],
                quat[2],
                quat[3],
                quat[0],
            ]
        )
        w_T_b = jnp.eye(4)
        w_T_b = w_T_b.at[:3, :3].set(rot.as_matrix().T)
        w_T_b = w_T_b.at[:3, 3].set(-rot.as_matrix().T @ xyz)

        goal_position = jnp.zeros(4)
        goal_position = goal_position.at[:2].set(state.goal_position)
        goal_position = goal_position.at[3].set(1)
        goal_position = w_T_b @ goal_position

        goal_position = jnp.where(state.no_goal_position, jnp.zeros_like(goal_position), goal_position)
        state = state.replace(current_target_heading=goal_position[:2])
        return state, key

    def reset_command(
        self, state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray
    ) -> tuple[MazeState, jnp.ndarray]:
        """Starts a new episode's command.

        The episode is an exploration episode (no goal) with probability
        ``initial_no_goal_pos_probability + no_goal_pos_multiplier * curr_coeff`` (clipped to [0, 1]).
        The goal position is reset to the origin.

        Args:
            state (MazeState): Current state of the robot and environment.
            env (SimpleMazeEnv): The environment.
            key (jnp.ndarray): JAX random key.

        Returns:
            tuple[MazeState, jnp.ndarray]: State and random key after reset.
        """
        use_no_goal_position = jax.random.uniform(key) < (
            jnp.clip(
                env.parameters["no_goal_pos_multiplier"] * state.curr_coeff
                + env.parameters["initial_no_goal_pos_probability"],
                0,
                1,
            )
        )
        key, _ = jax.random.split(key)
        goal_position = jnp.zeros_like(state.goal_position)
        state = state.replace(goal_position=goal_position, no_goal_position=use_no_goal_position)
        return state, key
