import jax.numpy as jnp
import jax
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState
from mujoco import mjx
from rememberwalk.utils.script_utils import dir_path
from rememberwalk.terrain.maze_generation.wilson_algorithm import Maze
from jax.scipy.spatial.transform import Rotation
import pickle

class MazeStaticGenerator:

    def __init__(self, terrain_parameters: dict,from_file = False,eval_mode=False):
        self.terrain_parameters = terrain_parameters
        self.one_meter_in_height_field = int(1 / terrain_parameters["height_field_resolution"])
        self.height_field_resolution = terrain_parameters["height_field_resolution"]
        self.wall_thickness= terrain_parameters["wall_thickness"]
        self.zero_indices = jnp.array([self.terrain_parameters["local_height_field_x_length_in_meters"]*self.one_meter_in_height_field / 2, self.terrain_parameters["local_height_field_y_length_in_meters"]*self.one_meter_in_height_field / 2])
        self.local_hf_field_half_length=jnp.array([self.terrain_parameters["local_height_field_x_length_in_meters"]*self.one_meter_in_height_field,self.terrain_parameters["local_height_field_y_length_in_meters"]*self.one_meter_in_height_field])/2
        self.global_hfield_zero_indices = jnp.array([self.terrain_parameters["height_field_x_length_in_meters"]*self.one_meter_in_height_field/2,self.terrain_parameters["height_field_y_length_in_meters"]*self.one_meter_in_height_field/2])
        self.slice_sizes = (self.local_hf_field_half_length*2).astype(jnp.int32)
        self.slice_sizes = (self.slice_sizes[0],self.slice_sizes[1])
        self.space_per_maze_cell = self.terrain_parameters["space_maze_cell"]
        self.terrain_curr_coeff_window_size = self.terrain_parameters["terrain_curr_coeff_window_size"]
        self.start_pos_noise = self.terrain_parameters["start_pos_noise"]
        self.maze_creator = Maze(self.terrain_parameters,save_to_file=False)
        if not eval_mode:
            if from_file:
                maze_path = f"{dir_path}/src/visual_locomotion_mjx/terrain/generated_terrain/maze_{terrain_parameters["maze_file"]}.pkl"
                with open(maze_path, "rb") as f: 
                    maze_dict = pickle.load(f)
                
                if isinstance(terrain_parameters["maze_file"], str):
                    maze_descriptions = jnp.array(maze_dict["maze_descriptions"]).astype(jnp.float32)
                else:
                    self.hfield,self.start_fields,maze_descriptions = jnp.array(maze_dict["global_hfield"]), jnp.array(maze_dict["global_start_fields"]), jnp.array(maze_dict["maze_descriptions"])
                    self.valid_start_fields = jnp.argwhere(self.start_fields)
            else:
                self.hfield,self.start_fields,maze_descriptions = self.maze_creator.generate_terrain()
            self.nr_mazes = terrain_parameters["nr_mazes"]
        else:
            maze_path = f"{dir_path}/src/visual_locomotion_mjx/terrain/generated_terrain/maze_{terrain_parameters["eval_maze_file"]}.pkl"
            with open(maze_path, "rb") as f: 
                maze_dict = pickle.load(f)
                maze_descriptions = jnp.array(maze_dict["maze_descriptions"]).astype(jnp.float32)
            self.nr_mazes = terrain_parameters["nr_eval_mazes"]
        self.maze_descriptions = maze_descriptions
        self.height_field_points_per_maze_in_x = int(self.terrain_parameters["maze_rows"] * self.terrain_parameters["space_maze_cell"] / self.height_field_resolution)
        self.cell_h = (2*self.local_hf_field_half_length)[0]
        self.cell_w = (2*self.local_hf_field_half_length)[1]
        self.R,self.C = self.maze_creator.maze_rows,self.maze_creator.maze_cols
        self.maze_shape = (2*self.local_hf_field_half_length).astype(jnp.int32)
        self.maze_shape = (int(self.maze_shape[0]),int(self.maze_shape[1]))
        self.slice_vals = (int(self.maze_shape[0]/self.R),int(self.maze_shape[1]/self.C))
        self.y_h = (jnp.arange(self.R+1) * self.cell_h)[:, None] # (R+1, 1)
        self.x_h = (jnp.arange(self.C) * self.cell_w)[None, :] # (1, C)
        self.y_h_flat = self.y_h.repeat(self.C, axis=1).reshape(-1).astype(jnp.bool_)
        self.x_h_flat = self.x_h.repeat(self.R+1, axis=0).reshape(-1).astype(jnp.bool_)
        self.y_v = (jnp.arange(self.R) * self.cell_h)[:, None] # (R, 1) 
        self.x_v = (jnp.arange(self.C+1) * self.cell_w)[None, :] # (1, C+1) 
        self.y_v_flat = self.y_v.repeat(self.C+1, axis=1).reshape(-1).astype(jnp.bool_)
        self.x_v_flat = self.x_v.repeat(self.R, axis=0).reshape(-1).astype(jnp.bool_)
        self.maze_wall_height = self.terrain_parameters["wall_height"]/self.terrain_parameters["max_possible_height"]  
        

        xs = ((jnp.arange(self.terrain_parameters["cell_visit_matrix_x_size"]) + 0.5) * ((2*self.local_hf_field_half_length[0])/self.terrain_parameters["cell_visit_matrix_x_size"]) * terrain_parameters["height_field_resolution"]) - self.local_hf_field_half_length[0] * terrain_parameters["height_field_resolution"]
        ys = (jnp.arange(self.terrain_parameters["cell_visit_matrix_y_size"]) + 0.5) * ((2*self.local_hf_field_half_length[0])/self.terrain_parameters["cell_visit_matrix_y_size"]) * terrain_parameters["height_field_resolution"] - self.local_hf_field_half_length[0] * terrain_parameters["height_field_resolution"]
        grid_y, grid_x = jnp.meshgrid(xs,ys,indexing='xy')
        self.cell_center_positions = jnp.stack([grid_x,grid_y],axis=-1)

        self.x = jnp.arange(0,self.terrain_parameters["local_height_field_x_length_in_meters"],self.height_field_resolution)
        self.y = jnp.arange(0,self.terrain_parameters["local_height_field_y_length_in_meters"],self.height_field_resolution)
        self.X, self.Y = jnp.meshgrid(self.x,self.y)
        self.xy = jnp.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        self.height_field_xpos_positions = jnp.zeros((self.X.flatten().shape[0],2))
        self.height_field_xpos_positions = self.height_field_xpos_positions.at[:,0].set(self.X.flatten())
        self.height_field_xpos_positions = self.height_field_xpos_positions.at[:,1].set(self.Y.flatten())
        self.height_field_xpos_positions = self.height_field_xpos_positions - jnp.array([self.terrain_parameters["local_height_field_x_length_in_meters"]/2,self.terrain_parameters["local_height_field_y_length_in_meters"]/2])

    def get_global_position_from_hfield_coords(self,coords:jnp.ndarray):
        deviation_from_midpoint = coords - self.global_hfield_zero_indices
        global_position = deviation_from_midpoint * self.height_field_resolution
        return global_position
    
    def get_global_position_from_hfield_coords_vectorized(self,coords:jnp.ndarray):
        deivation_from_midpoint = coords.at[:].subtract(self.global_hfield_zero_indices)
        global_position = deivation_from_midpoint * self.height_field_resolution
        return global_position

    def shift_on_initial_terrain(self,state:MazeState,env:SimpleMazeEnv,key):
        closest_idxs = env.terrain_observation.get_hfield_coords_from_xpos(state.pipeline_state.qpos[:2].reshape(1,2)).reshape(-1)
        # Substract zero indicies so that we get deviation from [0,0] in the local height field and add this to the global index
        #closest_idxs = closest_idxs - self.zero_indices
        # global_idxs = state.global_hfield_coords
        maze_nr = state.maze_nr
        # global_x_val = jnp.array([self.local_hf_field_half_length[0] * maze_nr,0])
        # state = state.replace(global_hfield_coords=global_idxs)
        sampled_terrain_curr_coeff,key = self.sample_terrain_curriculum_level(state.terrain_curr_coeff,key)
        maze_edge_curriculum = jax.random.uniform(key,shape=(self.R+1,self.C+1,2)) <= sampled_terrain_curr_coeff
        maze_edge_curriculum = maze_edge_curriculum.at[0,:,1].set(True)
        maze_edge_curriculum = maze_edge_curriculum.at[-1,:,1].set(True)
        maze_edge_curriculum = maze_edge_curriculum.at[:,0,0].set(True)
        maze_edge_curriculum = maze_edge_curriculum.at[:,-1,0].set(True)
        
        maze_description = jax.lax.dynamic_slice(self.maze_descriptions,(maze_nr,0,0,0),(1,self.R+1,self.C+1,2)).squeeze()
        
        maze_description = jnp.where(maze_edge_curriculum,maze_description,jnp.zeros_like(maze_description))
        maze_description = jnp.clip(maze_description,0.005,0.995)
        new_hfield = self.generate_maze_walls(maze_description)

        new_hfield = new_hfield + (state.curr_coeff * jax.random.uniform(key,new_hfield.shape,minval=-self.terrain_parameters["terrain_roughness"],maxval=self.terrain_parameters["terrain_roughness"])/self.terrain_parameters["max_possible_height"])  
        
        
        state = state.replace(model=state.model.replace(hfield_data=new_hfield.T.reshape(-1)),maze_description=maze_description.flatten(),sampled_terrain_curr_coeff=sampled_terrain_curr_coeff)
        return state,closest_idxs + self.zero_indices,key
    
    
    def generate_maze_walls(self,maze_description):
        
        new_maze = jnp.zeros((self.maze_shape[0],self.maze_shape[1]),dtype=jnp.float32)
        horizontal_edges = maze_description[1:,1:,1].reshape((self.R,self.C))
        horizontal_edges_repeat = jnp.repeat(horizontal_edges[:-1,:],repeats=int(self.space_per_maze_cell/self.height_field_resolution),axis=1)
        vertical_edges = maze_description[1:,1:,0].reshape((self.R,self.C))
        vertical_edges_repeat = jnp.repeat(vertical_edges[:,:-1],repeats=int(self.space_per_maze_cell/self.height_field_resolution),axis=0)
        new_maze = new_maze.at[self.slice_vals[0]::self.slice_vals[0],:].set(self.maze_wall_height*horizontal_edges_repeat)
        new_maze = new_maze.at[:,self.slice_vals[1]::self.slice_vals[1]].set(self.maze_wall_height*vertical_edges_repeat)

        # Set boundaries
        new_maze = new_maze.at[0,:].set(self.maze_wall_height)
        new_maze = new_maze.at[:,0].set(self.maze_wall_height)
        new_maze = new_maze.at[-1,:].set(self.maze_wall_height)
        new_maze = new_maze.at[:,-1].set(self.maze_wall_height)

        n = self.wall_thickness

        new_maze = jax.lax.reduce_window(
            new_maze,
            0.0,                 # Initial value (base height of the floor)
            jax.lax.max,         # Rule: if a wall is in the window, this pixel becomes wall
            (n, n),              # Window size is the thickness
            (1, 1),              # Move 1 pixel at a time
            padding='SAME'       # Keep original dimensions
        )

        return new_maze

    def check_walls_between_cell_and_neighbors(self,maze_description,current_cell):
        maze = maze_description.reshape((self.maze_descriptions.shape[1],self.maze_descriptions.shape[2],self.maze_descriptions.shape[3]))
        current_cell = current_cell + 1
        wall_east_cell = maze[current_cell[0],current_cell[1],0]
        wall_south_cell = maze[current_cell[0],current_cell[1],1]
        wall_north_cell = maze[current_cell[0]-1,current_cell[1],1]
        wall_west_cell = maze[current_cell[0],current_cell[1]-1,0]
        return jnp.array([wall_east_cell,wall_south_cell,wall_west_cell,wall_north_cell])



    def sample_terrain_curriculum_level(self,max_terrain_curr_coeff,key, alpha=5.0, beta=0.5):
        min_level = jnp.clip(max_terrain_curr_coeff - self.terrain_curr_coeff_window_size, 0.0, 1.0)
        sample = jax.random.beta(key, alpha, beta)
        sampled_level = min_level + (max_terrain_curr_coeff - min_level) * sample
        return sampled_level,key

    def shift_robot_given_hfield_coords(self,state:MazeState,coords:jnp.ndarray,env):
        pos = env.terrain_observation.get_xpos_from_hfield_coords(coords)
        new_qpos = state.pipeline_state.qpos.at[:2].add(-pos)
        pipeline_state = state.pipeline_state.replace(qpos=new_qpos)
        return state.replace(pipeline_state=pipeline_state)
    
    def get_cell_idxs_from_maze(self,state:MazeState):
        pos = state.pipeline_state.qpos[:2] 
        cell_size = self.terrain_parameters["cell_visit_size_in_meters"] 
        num_cells = jnp.array([self.terrain_parameters["cell_visit_matrix_x_size"],self.terrain_parameters["cell_visit_matrix_y_size"]]) 
        total_size = cell_size * num_cells 
        shift = total_size / 2.0 
        pos_shifted = pos + shift
        return (pos_shifted // cell_size).astype(jnp.int32)
    
    def get_maze_nr_from_global_hfield_idx_from_coord(self,coords:jnp.ndarray):
        global_hfield_x_idx = coords[0]
        maze_nr = global_hfield_x_idx // self.height_field_points_per_maze_in_x
        return maze_nr
    

    def get_maze_nr_from_global_hfield_idx(self,state:MazeState):
        global_hfield_x_idx = state.global_hfield_coords[0]
        maze_nr = global_hfield_x_idx // self.height_field_points_per_maze_in_x
        return maze_nr

    def get_valid_global_goal_position(self,state:MazeState,key):
        maze_nr = state.maze_nr
        global_robot_pos = self.get_global_position_from_hfield_coords(state.global_hfield_coords) + state.pipeline_state.qpos[:2]
        global_valid_start_field_pos = self.get_global_position_from_hfield_coords_vectorized(self.valid_start_fields)
        potential_start_states_mask = (self.valid_start_fields[:,0] >= (maze_nr * 2*self.local_hf_field_half_length[0])) & (self.valid_start_fields[:,0] <=  ((maze_nr +1)* 2*self.local_hf_field_half_length[0])) & (jnp.linalg.norm(global_robot_pos-global_valid_start_field_pos) > 2.5)
        probs = potential_start_states_mask.astype(jnp.float32)
        probs = probs / jnp.sum(probs)
        valid_start_coords = jax.random.choice(key,self.valid_start_fields,p=probs,shape=())
        goal_position = self.get_global_position_from_hfield_coords(valid_start_coords)
        return goal_position


    def reset_start_position(self, state:MazeState , env: SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:
        """
        Resets the start position of the terrain.
        """
        maze_nr = jax.random.randint(key,shape=(),minval=0,maxval=self.nr_mazes)
        key,_ = jax.random.split(key)
        start_pos = jnp.array([0.01,0.0]) + jax.random.uniform(key,(2,),minval=-self.start_pos_noise, maxval=self.start_pos_noise)
        key,_ = jax.random.split(key)
        qpos = state.pipeline_state.qpos.at[:2].set(start_pos)
        _,euler_angles,_ = self.update_orientation(qpos)
        starting_pos = jnp.array([start_pos[0],start_pos[1],euler_angles[2]])
        state = state.replace(maze_nr=maze_nr,pipeline_state=state.pipeline_state.replace(qpos=qpos),starting_pos=starting_pos)
        state,_,key = self.shift_on_initial_terrain(state,env,key)
        key,_ = jax.random.split(key)
        data = mjx.step(state.model, state.pipeline_state)
        state = state.replace(pipeline_state=data)
        feet_ids = env.robot.geom_feet_ids
        feet_pos = data.geom_xpos[feet_ids]
        heights = env.terrain_observation.get_height_from_xpos(
            state, feet_pos[:, :2].reshape(-1, 2),env
        )
        min_feet_z_pos = jnp.min(
            data.geom_xpos[feet_ids, 2]
            - state.model.geom_size[feet_ids, 0]
        )
        offset = -min_feet_z_pos
        min_height = jnp.min(
            data.geom_xpos[feet_ids, 2]
            - heights
            - state.model.geom_size[feet_ids, 0]
        )
        offset = -min_height
        qpos = data.qpos.at[2].add(offset)
        pipeline_state = data.replace(
            qpos=qpos)
        state = state.replace(pipeline_state=pipeline_state)
        return state, key




    def update_orientation(self,qpos:jnp.ndarray):
        orientation_quat = Rotation.from_quat(
            [
                qpos[4],
                qpos[5],
                qpos[6],
                qpos[3],
            ]
        )
        orientation_euler = orientation_quat.as_euler("xyz")
        orietation_quat_inv = orientation_quat.inv()

        return orientation_quat, orientation_euler, orietation_quat_inv