import mujoco
from mujoco import mjx
from mujoco.mjx import Model, Data
import jax
import jax.numpy as jnp


class BaseMJXEnv:

    def __init__(self, parameters: dict) -> None:
        self.robot = parameters["robot"]
        self.model = self.robot.model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.physics_steps_per_env_step = parameters["physics_steps_per_env_step"]
        self.qpos_size = self.data.qpos.shape[0]
        self.qvel_size = self.data.qvel.shape[0]

    def reset(self) -> tuple[Data, Model]:
        self.robot.init_coll_ids(self.parameters["intial_state"].pipeline_state)
        return 

    def step(self, state, action: jax.Array):
        """
        Advances the simulation by one environment step using the provided action.

        Applies the action to the robot, performs the specified number of physics steps,
        and updates the simulation state. Joint velocities are clipped to their maximum allowed values.
        Returns the updated simulation state.

        Args:
            state (State): Current simulation state.
            action (jax.Array): Action to apply to the robot.

        Returns:
            State: Updated simulation state after applying the action.
        """
        def do_single_step(data_step: mjx.Data, x) -> mjx.Data:
            # Perform a single step in the environment
            data_step = data_step.replace(ctrl=action)
            data_step = mjx.step(state.model, data_step)
            return data_step, None

        data_new, _ = jax.lax.scan(
            do_single_step,
            state.pipeline_state,
            None,
            length=self.physics_steps_per_env_step,
        )
        data_new = data_new.replace(
            qvel=data_new.qvel.at[self.robot.joint_qvel_adr].set(
                jnp.clip(
                    data_new.qvel[self.robot.joint_qvel_adr],
                    -self.robot.max_joint_velocities,
                    self.robot.max_joint_velocities,
                )
            )
        )

        state = state.replace(pipeline_state=data_new)

        return state
