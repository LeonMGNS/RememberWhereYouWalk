import jax.numpy as jnp
from rememberwalk.env.BaseMJXEnv import BaseMJXEnv
from rememberwalk.states.MazeState import MazeState
import jax

class SimpleMazeEnv(BaseMJXEnv):

    def __init__(self, parameters: dict):
        super().__init__(parameters)
        # Defining environment variables from parameter dict
        self.parameters = parameters
        self.dr_parameters = parameters["dr_parameters"]
        self.terrain_parameters = parameters["terrain_parameters"]
        self.curr_coeff = parameters["initial_curr_coeff"]
        self.terrain_curr_coeff = parameters["initial_terrain_curr_coeff"]
        self.mode = parameters["mode"]
        self.save_observation_history = parameters["save_observation_history"]
        if self.save_observation_history:
            assert "observation_history_length" in parameters.keys()
            self.observation_history_length = int(parameters["observation_history_length"])
        self.curr_coeff_delta = parameters["curr_coeff_delta"]
        self.max_curr_coeff = parameters["max_curr_coeff"]
        self.max_terrain_curr_coeff = parameters["max_terrain_curr_coeff"]
        self.episode_length = parameters["episode_length"]
        self.episode_length_in_seconds = self.episode_length * self.physics_steps_per_env_step * self.model.opt.timestep
        self.actor_observation_size = parameters["actor_observation_size"]
        self.critic_observation_size = parameters["critic_observation_size"]
        self.dt = self.model.opt.timestep * parameters["physics_steps_per_env_step"]
        self.height_map_max_delay = parameters["height_map_max_delay"]
        self.action_size = int(parameters["actor_output_size"]/2)
        self.nr_height_map_points = parameters["nr_height_map_points"]
        self.critic_nr_height_map_points = parameters["critic_nr_height_map_points"]
        self.max_allowed_loc_error = parameters["max_allowed_loc_error"]
        self.initial_mjx_model = parameters["initial_mjx_model"]

        # Defining reward parameters
        imports = parameters["imports"]
        # Defining needed environment functions
        self.get_obs = imports["get_obs"]
        self.reset_fn = imports["reset_fn"]
        self.reset_randomize_state = imports["reset_randomize_state"]
        self.get_done = imports["get_done"]
        self.update_lasttouch = imports["update_lasttouch"]
        self.reward_handler  = imports["reward_handler"]
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
        self.command_creater = imports["CommandCreator"]
        self.MazeCellBasedRand = imports["MazeCellBasedRand"]

    def reset(self,inital_curr_coeff:jnp.ndarray,initial_terrain_curr_coeff, key: jnp.ndarray) -> MazeState:
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
        state = state.replace(curr_coeff = inital_curr_coeff,terrain_curr_coeff=initial_terrain_curr_coeff)
        state, key = self.reset_fn(state, self, key)
        state, key = self.get_obs(state, self, key)
        state = self.get_reward(state, state.last_action, self)
        state = self.get_done(state, self)
        state = state.replace(done=False)
        return state

    def step(self, state: MazeState, action: jnp.ndarray, key: jnp.ndarray):

        #
        state,key = self.reset_variables(state, self,key)
        
        # Randomize State and Action
        state, action, key = self.randomize_state_action(state, action, self, key)
        
        # Reset the state variables if needed
        ctrl = self.pd_control.process_action(state.pipeline_state,
                                            action,
                                            state.extrinsic_p_gain_factor,
                                            state.extrinsic_d_gain_factor,
                                            state.extrinsic_motor_strength,
                                            state.extrinsic_position_offset,
                                            )
        # Apply the control to the environment
        state = super().step(state, ctrl)
        
        #Update n_steps
        state = state.replace(n_steps=state.n_steps+1)
        
        # Update the last touch for collision detection
        state = self.update_lasttouch(state, self.robot.geom_feet_ids, self)
        
        # Get the current reward
        state = self.get_reward(state, action, self)

        state = state.replace(last_action=action)

        # Update if episode is done
        state = self.get_done(state, self)
        
        # Update command
        state, key = self.command_creater.update_command(state, self, key)

        # Update current curriculum coefficient
        state = self.update_curriculum(state, self)
        
        # Reset the state if the episode is done
        state, key = self.reset_fn(state, self, key)

        # Get the current observation
        state, key = self.get_obs(state, self, key)
        
        return state, key
