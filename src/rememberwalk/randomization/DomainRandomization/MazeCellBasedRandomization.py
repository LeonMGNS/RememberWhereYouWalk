import jax
import jax.numpy as jnp
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState



class MazeCellDR:

    def __init__(self,proprio_parameters):
        self.proprio_parameters = proprio_parameters
        self.proprio_values = self.proprio_parameters["proprio_values"]
        self.randomization_prob = self.proprio_parameters["randomization_prob"]
        self.min_gravity = self.proprio_parameters["min_gravity"]
        self.max_gravity = self.proprio_parameters["max_gravity"]
        self.friction_min = self.proprio_parameters["friction_min"]
        self.friction_tagential_max = self.proprio_parameters["friction_tangential_max"]
        self.friction_rolling_max = self.proprio_parameters["friction_rolling_max"]
        self.friction_torsional_max = self.proprio_parameters["friction_torsional_max"]
        self.terrain_roughness_min = self.proprio_parameters["terrain_roughness_min"]
        self.terrain_roughness_max = self.proprio_parameters["terrain_roughness_max"]
        self.standard_terrain_roughness = self.proprio_parameters["terrain_roughness"]

        self.proprio_min_limits = jnp.array([
            self.min_gravity, # Gravity
            self.friction_min, # Tangential 1
            self.friction_min, # Torsional
            self.friction_min,   # Rolling
            self.terrain_roughness_min # Terrain Roughness
        ])

        self.proprio_max_limits = jnp.array([
            self.max_gravity, # Gravity
            self.friction_tagential_max, # Tangential
            self.friction_torsional_max, # Torsional
            self.friction_rolling_max,   # Rolling
            self.terrain_roughness_max, # Terrain Roughness
        ])
    
    def reset_randomize_propriocpetion_per_grid_cell(self,state:MazeState,env:SimpleMazeEnv,key):
        #Episodic values
        model= state.model
        gravity = model.opt.gravity[2]
        friction_tagential = model.pair_friction[env.robot.pair_ids[0],0]
        friction_torsional = model.pair_friction[env.robot.pair_ids[0],2]
        friciton_rolling = model.pair_friction[env.robot.pair_ids[0],3]
        proprio_episode_feature_vector = jnp.array([gravity,friction_tagential,friction_torsional,friciton_rolling,self.standard_terrain_roughness])
        proprio_episodic_values = jnp.zeros_like(state.proprio_cell_values)
        proprio_episodic_values = proprio_episodic_values.at[:].set(proprio_episode_feature_vector)
    
        #Randomization
        random_val = jax.random.uniform(key,shape=state.proprio_cell_values.shape[:-1])
        key,_ = jax.random.split(key)
        should_randomize = random_val < self.randomization_prob

        #Proprio values
        randomized_cells = self.interpolate_param(proprio_episodic_values,self.proprio_min_limits,self.proprio_max_limits,state.curr_coeff,key)
        key,_ = jax.random.split(key)
        proprio_cell_values = jnp.where(should_randomize[:,:,None],randomized_cells,proprio_episodic_values)
        state = state.replace(proprio_cell_values=proprio_cell_values)
        return state,key


    def randomize_propriocpetion_per_grid_cell(self,state:MazeState,env:SimpleMazeEnv):
        #Load cell based parameters
        cell_idx = env.terrain_generator.get_cell_idxs_from_maze(state)
        cell_idx_for_slice = jnp.array([cell_idx[0],cell_idx[1],0])
        proprio_values_01 = jax.lax.dynamic_slice(state.proprio_cell_values,cell_idx_for_slice,(1,1,self.proprio_values))
        proprio_values = self.proprio_min_limits + proprio_values_01 * (self.proprio_max_limits-self.proprio_min_limits)
        model= state.model

        #Set cell based parameters
        opt = model.opt.replace(gravity=model.opt.gravity.at[2].set(proprio_values[cell_idx[0],cell_idx[1],0]))
        pair_friction = model.pair_friction.at[env.robot.pair_ids,:2].set(proprio_values[cell_idx[0],cell_idx[1],1])
        pair_friction = pair_friction.at[env.robot.pair_ids,2].set(proprio_values[cell_idx[0],cell_idx[1],2])
        pair_friction = pair_friction.at[env.robot.pair_ids,3:].set(proprio_values[cell_idx[0],cell_idx[1],3])
        model = model.replace(opt=opt,pair_friction=pair_friction)

        state = state.replace(model=model)
        return state
    

    def interpolate_param(self,initial, min_val, max_val,curr_coeff, key):
            random_value = jax.random.uniform(key, shape=initial.shape)  # Sample in [0,1]
            sampled_value = (1 - random_value) * min_val + random_value * max_val  # Interpolates in [min_val, max_val]
            new_value = initial + curr_coeff * (sampled_value - initial)  # Scale deviation from initial
            return jnp.clip(new_value, min_val, max_val)




        

