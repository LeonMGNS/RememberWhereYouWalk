"""The maze environment (``imports.env``)."""

import jax
import jax.numpy as jnp
from mujoco import mjx

from rememberwalk.states.MazeState import MazeState


class SimpleMazeEnv:
    """MJX maze environment of one env (vmapped by the trainer).

    The task-specific parts (observation, reward, done, command, curriculum, reset, domain randomisation,
    terrain) are the functions and handlers selected in the config's ``imports`` section.
    """

    def __init__(self, parameters: dict):
        self.robot = parameters["robot"]
        self.model = self.robot.model
        self.physics_steps_per_env_step = parameters["physics_steps_per_env_step"]
        # Defining environment variables from parameter dict
        self.parameters = parameters
        self.dr_parameters = parameters["dr_parameters"]
        self.terrain_parameters = parameters["terrain_parameters"]
        self.curr_coeff = parameters["initial_curr_coeff"]
        self.terrain_curr_coeff = parameters["initial_terrain_curr_coeff"]
        self.curr_coeff_delta = parameters["curr_coeff_delta"]
        self.max_curr_coeff = parameters["max_curr_coeff"]
        self.max_terrain_curr_coeff = parameters["max_terrain_curr_coeff"]
        self.episode_length = parameters["episode_length"]
        self.actor_observation_size = parameters["actor_observation_size"]
        self.dt = self.model.opt.timestep * parameters["physics_steps_per_env_step"]
        self.action_size = int(parameters["actor_output_size"] / 2)

        imports = parameters["imports"]
        # Defining needed environment functions
        self.get_obs = imports["get_obs"]
        self.reset_fn = imports["reset_fn"]
        self.reset_randomize_state = imports["reset_randomize_state"]
        self.get_done = imports["get_done"]
        self.update_lasttouch = imports["update_lasttouch"]
        self.reward_handler = imports["reward_handler"]
        self.get_reward = self.reward_handler.get_reward
        self.update_curriculum = imports["update_curriculum"]
        self.randomize_state_action = imports["randomize_state_action"]
        self.reset_variables = imports["reset_variables"]
        self.collision_fn = imports["get_collision"]
        self.contact_fn = imports["contact_fn"]

        # Defining needed environment objects
        self.pd_control = imports["PDControl"]
        self.terrain_observation = imports["TerrainObs"]
        self.terrain_generator = imports["TerrainGenerator"]
        self.proprioceptive_obs_dr_handler = imports["ObsDR"]
        self.InitStateRand = imports["InitStateRandomization"]
        self.DomainModel = imports["ModelDR"]
        self.DomainPerturbation = imports["DefaultDomainPerturbation"]
        self.control_randomization = imports["DefaultDomainControl"]
        self.DelayAction = imports["DelayAction"]
        self.command_creator = imports["CommandCreator"]

    def reset(
        self, initial_curr_coeff: jnp.ndarray, initial_terrain_curr_coeff, key: jnp.ndarray
    ) -> MazeState:
        """
        Initializes the MazeState at the beggining of training.

        Sets up the simulation state, including joint positions, velocities, observations, reward variables,
        curriculum coefficients, waypoints, and all domain randomization buffers. Applies environment-specific
        reset logic and generates the initial observation and reward.

        Args:
            key (jnp.ndarray): JAX random key for stochastic initialization.

        Returns:
            MazeState: The initialized state for the start of the episode.
        """

        state = self.parameters["intial_state"]
        # Height Map delay
        state = state.replace(curr_coeff=initial_curr_coeff, terrain_curr_coeff=initial_terrain_curr_coeff)
        state, key = self.reset_fn(state, self, key)
        state, key = self.get_obs(state, self, key)
        state = self.get_reward(state, state.last_action, self)
        state = self.get_done(state, self)
        state = state.replace(done=False)
        return state

    def step(self, state: MazeState, action: jnp.ndarray, key: jnp.ndarray):
        """One env step: (state, action (action_size,), key) -> (next state, key).

        Order: clear the counters of an env that finished last step, domain randomisation of state and
        action, PD control, physics, contacts, reward, done, command, curriculum, reset of done envs, and
        finally the observation of the (possibly reset) state.
        """
        state, key = self.reset_variables(state, self, key)

        # Randomize State and Action
        state, action, key = self.randomize_state_action(state, action, self, key)

        # PD control
        ctrl = self.pd_control.process_action(
            state.pipeline_state,
            action,
            state.extrinsic_p_gain_factor,
            state.extrinsic_d_gain_factor,
            state.extrinsic_motor_strength,
            state.extrinsic_position_offset,
        )
        # Apply the control to the environment
        state = self.physics_step(state, ctrl)

        # Update n_steps
        state = state.replace(n_steps=state.n_steps + 1)

        # Update the last touch for collision detection
        state = self.update_lasttouch(state, self.robot.geom_feet_ids, self)

        # Get the current reward
        state = self.get_reward(state, action, self)

        state = state.replace(last_action=action)

        # Update if episode is done
        state = self.get_done(state, self)

        # Update command
        state, key = self.command_creator.update_command(state, self, key)

        # Update current curriculum coefficient
        state = self.update_curriculum(state, self)

        # Reset the state if the episode is done
        state, key = self.reset_fn(state, self, key)

        # Get the current observation
        state, key = self.get_obs(state, self, key)

        return state, key

    def physics_step(self, state, action: jax.Array):
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
