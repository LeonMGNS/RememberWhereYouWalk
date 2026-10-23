from jax import numpy as jnp
import jax
from jax.scipy.spatial.transform import Rotation as R

class TerrainObs:

    def __init__(
        self,
        terrain_parameters,
        dr_height_map_parameters,
        lidar_parameters,
        robot,

    ):
        self.robot = robot
        self.resolution = terrain_parameters["height_map_resolution"]
        self.x_bounds = terrain_parameters["height_map_grid_x_length"]
        self.y_bounds = terrain_parameters["height_map_grid_y_length"]
        self.hf_field_length_x = terrain_parameters["cells_in_x"]
        self.hf_field_length_y = terrain_parameters["cells_in_y"]
        self.hf_field_length = jnp.array([self.hf_field_length_x, self.hf_field_length_y])
        self.terrain_max_possible_height = terrain_parameters["max_possible_height"]
        self.hf_field_x_half_length_in_meters = terrain_parameters["length_in_x"]
        self.hf_field_y_half_length_in_meters = terrain_parameters["length_in_y"]
        self.hf_field_half_length_in_meters = jnp.array([self.hf_field_x_half_length_in_meters, self.hf_field_y_half_length_in_meters])
        self.z_offset_noise = dr_height_map_parameters["height_map_z_offset_noise"]
        self.height_map_tilt_roll_noise = dr_height_map_parameters["height_map_tilt_roll_noise"]
        self.height_map_tilt_pitch_noise = dr_height_map_parameters["height_map_tilt_pitch_noise"]
        
        self.xy_grid_noise = dr_height_map_parameters["xy_grid_noise"]
        self.avg_kernel_deviation = dr_height_map_parameters["avg_kernel_deviation"]
        self.critic_x_bounds = terrain_parameters["critic_height_map_grid_x_length"]
        self.critic_y_bounds = terrain_parameters["critic_height_map_grid_y_length"]
        self.critic_max_pool_window_size = terrain_parameters["critic_max_pool_window_size"]
        

        # Lidar parameters
        self.lidar_point_noise = lidar_parameters["point_noise"]

        # Create Actor and critic height map
        self.x = jnp.arange(-self.x_bounds, self.x_bounds, self.resolution)
        self.y = jnp.arange(-self.x_bounds, self.y_bounds, self.resolution)
        self.critic_x = jnp.arange(-self.critic_x_bounds,self.critic_x_bounds,self.resolution)
        self.critic_y = jnp.arange(-self.critic_y_bounds,self.critic_y_bounds,self.resolution)
        self.critic_X, self.critic_Y = jnp.meshgrid(self.critic_x,self.critic_y)
        self.critic_xy = jnp.stack([self.critic_X.ravel(),self.critic_Y.ravel()],axis=-1)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        self.xy = jnp.stack([self.X.ravel(), self.Y.ravel()], axis=-1)


        #Create radial height map
        self.height_map_radius = lidar_parameters["height_map_radius"]
        self.height_map_nr_rays = lidar_parameters["height_map_nr_rays"]
        self.height_map_nr_points_per_ray = lidar_parameters["height_map_nr_points_per_ray"]
        self.max_clipping_height = lidar_parameters["max_clipping_height"]
        angles = jnp.linspace(0, 2 * jnp.pi, self.height_map_nr_rays, endpoint=False)
        radii = jnp.linspace(
            self.height_map_radius / self.height_map_nr_points_per_ray, 
            self.height_map_radius, 
            self.height_map_nr_points_per_ray
        )
        Angles, Radii = jnp.meshgrid(angles, radii, indexing='ij')

        # Convert from Polar (r, theta) to Cartesian (X, Y)
        X = Radii * jnp.cos(Angles)
        Y = Radii * jnp.sin(Angles)
        self.polar_xy = jnp.stack([X, Y], axis=-1)

        flat_xy = self.polar_xy.reshape(-1, 2)

        # Concatenate a column of zeros for the Z-axis
        self.radial_height_map = jnp.concatenate(
            [flat_xy, jnp.zeros((flat_xy.shape[0], 1))],
            axis=-1
        )

    def get_critic_height_map(self,state,env):
        _,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        height_map_points =  jnp.concatenate([self.critic_xy, jnp.zeros_like(self.critic_xy[:, 0]).reshape((-1, 1))], axis=-1)
        yaw_rotated_height_map_points = (R_world_yaw @ height_map_points.T).T
        lidar_centered_height_map_points = yaw_rotated_height_map_points + lidar_pos_in_world
        critic_heights = self.get_height_from_xpos(state,lidar_centered_height_map_points[:,:2],env)
        critic_heights = jnp.where(jnp.any(jnp.abs(lidar_centered_height_map_points[:,:2])>self.hf_field_half_length_in_meters,axis=1),jnp.zeros_like(critic_heights),critic_heights)
        critic_heights_max_pooled=self.max_pool(critic_heights)
        return critic_heights_max_pooled.reshape(-1)

    def max_pool(self,height_map):
        H = self.critic_x.shape[0]
        W = self.critic_y.shape[0]
        reshaped_height_map = height_map.reshape((H,W))
        reshaped_height_map = reshaped_height_map.reshape(H//self.critic_max_pool_window_size, self.critic_max_pool_window_size, W//self.critic_max_pool_window_size, self.critic_max_pool_window_size)
        return jnp.max(reshaped_height_map,(1,3))




    def get_gr_height_map_with_noise(self,state,env,key):
        T_world_to_lidar,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        yaw_rotated_height_map_points = (R_world_yaw @ self.radial_height_map.T).T
        robot_centered_height_map_points = yaw_rotated_height_map_points.at[:,:2].add(state.pipeline_state.qpos[:2]+ jax.random.uniform(key,(2,),minval=-self.xy_grid_noise,maxval=self.xy_grid_noise)) 
        raw_heights = self.get_height_from_xpos(state,robot_centered_height_map_points[:,:2],env)
        raw_heights = raw_heights
        heights = jnp.clip(raw_heights,min=None,max=self.max_clipping_height)

        heights = heights + jax.random.uniform(key,heights.shape,minval=-self.lidar_point_noise,maxval=self.lidar_point_noise)
        radial_height_map = self.radial_height_map.at[:,2].set(heights)
        key,_ = jax.random.split(key) 
        return radial_height_map[:,2],key
    

    def get_hfield_idx_from_xpos(self, coords: jnp.ndarray):
        coords = self.get_hfield_coords_from_xpos(coords)
        idx = (coords[:, 1] * (self.hf_field_length[0]) + coords[:, 0]).astype(jnp.int32)
        return idx

    def get_xpos_from_hfield_coords(self, coords: jnp.ndarray):
        coords = coords - self.hf_field_length / 2
        coords = coords / (self.hf_field_length / 2)
        coords = coords * self.hf_field_half_length_in_meters
        return coords

    def get_hfield_coords_from_xpos(self, coords: jnp.ndarray):
        coords /= self.hf_field_half_length_in_meters
        coords = coords * self.hf_field_length / 2 + self.hf_field_length / 2
        coords = jnp.floor(coords).astype(jnp.int32)
        return coords

    def get_height_from_xpos(self, state, xpos: jnp.ndarray,env):
        grid_ids = self.get_hfield_idx_from_xpos(xpos)
        z = state.model.hfield_data[grid_ids] * self.terrain_max_possible_height
        return z

    def randomize_height_map_parameters(self, state, key):
        avg_kernel = jnp.ones((3, 3, 1, 1)) / 9.0
        random_kernel = avg_kernel + jax.random.uniform(key, shape=(3, 3), minval=-self.avg_kernel_deviation, maxval=self.avg_kernel_deviation).reshape(3, 3, 1, 1)
        key,_ = jax.random.split(key)
        random_kernel = jnp.where(random_kernel > 0,random_kernel,jnp.zeros_like(random_kernel))
        random_kernel = (random_kernel/jnp.sum(random_kernel))
        roll_angle = jax.random.uniform(
            key, shape=(1,), minval=-self.height_map_tilt_roll_noise, maxval=self.height_map_tilt_roll_noise
        )
        key,_ = jax.random.split(key)
        pitch_angle = jax.random.uniform(
            key, shape=(1,), minval=-self.height_map_tilt_pitch_noise, maxval=self.height_map_tilt_pitch_noise
        )
        key,_ = jax.random.split(key)
        roll_matrix = R.from_euler("x", roll_angle).as_matrix()
        pitch_matrix = R.from_euler("y", pitch_angle).as_matrix()
        rotation_matrix = roll_matrix @ pitch_matrix
        z_offset = jax.random.uniform(
            key, shape=(), minval=-self.z_offset_noise, maxval=self.z_offset_noise
        )
        key,_ = jax.random.split(key)
        xy_noise = jax.random.choice(key, jnp.array([0.0, 0.05, 0.1]), shape=(2,), p=jnp.array([0.7, 0.25, 0.05]))
        _, key = jax.random.split(key)
        xy_noise_sign = jax.random.choice(key, jnp.array([-1, 1]), shape=(2,), p=jnp.array([0.5, 0.5]))
        xy_noise = xy_noise * xy_noise_sign
        _, key = jax.random.split(key)

        xy_noise = jnp.zeros((2,))
        state = state.replace(
            height_map_kernel=random_kernel,
            height_map_tilt=rotation_matrix,
            height_map_offset=z_offset,
            height_map_xy_noise=xy_noise
        )
        return state

    def update_orientation(sefl, pipeline_state):
        orientation_quat = R.from_quat(
            [
                pipeline_state.qpos[4],
                pipeline_state.qpos[5],
                pipeline_state.qpos[6],
                pipeline_state.qpos[3],
            ]
        )
        orientation_euler = orientation_quat.as_euler("xyz")
        yaw = orientation_euler[2]  # Yaw is the third angle in 'xyz' convention

        # Create a rotation matrix for yaw only
        yaw_matrix = R.from_euler("z", yaw).as_matrix()
        orietation_quat_inv = orientation_quat.inv()

        return orientation_quat, orientation_euler, orietation_quat_inv, yaw_matrix
    


    def get_world_to_lidar_transformation(self,state):
        xyz_rviz = jnp.array([0.29,0.0,-0.08])
        quat_rviz = jnp.array([0,0,0,1])
        
        R_mat = R.from_quat(quat_rviz).as_matrix()
        b_T_l = jnp.eye(4)
        b_T_l = b_T_l.at[:3,:3].set(R_mat)
        b_T_l= b_T_l.at[:3,3].set(xyz_rviz)


        # World To Base
        xyz = state.pipeline_state.qpos[:3]
        xyz = state.pipeline_state.xpos[self.robot.base_body_id]
        quat = state.pipeline_state.xquat[self.robot.base_body_id]
        rot = R.from_quat(
        [
            quat[1],
            quat[2],
            quat[3],
            quat[0],
        ]
        )
        w_T_b = jnp.eye(4)
        w_T_b = w_T_b.at[:3,:3].set(rot.as_matrix())
        w_T_b= w_T_b.at[:3,3].set(xyz)

        #World To Lidar
        w_T_l = w_T_b @ b_T_l
        lidar_pos_in_world = w_T_l[:3,3]

        
        l_T_w = jnp.eye(4)
        
        l_T_w = l_T_w.at[:3,:3].set(w_T_l[:3,:3].T)
        l_T_w = l_T_w.at[:3,3].set(-w_T_l[:3,:3].T @ w_T_l[:3,3]) 
        
        return l_T_w,lidar_pos_in_world,w_T_l
    
