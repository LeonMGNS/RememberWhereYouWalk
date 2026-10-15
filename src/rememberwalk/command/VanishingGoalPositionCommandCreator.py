from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
import jax
import jax.numpy as jnp
from rememberwalk.states.MazeState import MazeState
from jax.scipy.spatial.transform import Rotation as R


class StaticMazeCommandCreator:

    def __init__(self, terrain_parameters):

        self.change_vel_prob = 0.004
        self.zero_vel_prob = self.change_vel_prob * 0.25
        self.terrain_parameters = terrain_parameters



    def update_command(self, state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:        
        goal_position = state.goal_position
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
        w_T_b = w_T_b.at[:3,:3].set(rot.as_matrix().T)
        w_T_b= w_T_b.at[:3,3].set(-rot.as_matrix().T @ xyz)

        goal_position = jnp.zeros(4)
        goal_position = goal_position.at[:2].set(state.goal_position)
        goal_position = goal_position.at[3].set(1)
        goal_position = w_T_b @ goal_position
        
        
        goal_position = jnp.where(state.no_goal_position,jnp.zeros_like(goal_position),goal_position)
        state = state.replace(current_target_heading=goal_position[:2])
        return state,key

        

    def reset_command(self, state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:
        """_Resets the command and waypoints for a new episode.

        Selects the appropriate reset function for box or walking terrain, computes the distance to travel,
        and resets the travelled distance.

        Args:
            state (MazeState): Current state of the robot and environment.
            env (SimpleMazeEnv): Box environment instance.
            key (jnp.ndarray): JAX random key.

        Returns:
            tuple[MazeState, jnp.ndarray]: State and random key after reset.
        """
        use_no_goal_position = jax.random.uniform(key) < (jnp.clip(env.parameters["no_goal_pos_multiplier"] * state.curr_coeff + env.parameters["initial_no_goal_pos_probability"],0,1))
        key,_ = jax.random.split(key)
        goal_position = jnp.zeros_like(state.goal_position)
        state = state.replace(goal_position=goal_position,no_goal_position=use_no_goal_position)
        return state, key

