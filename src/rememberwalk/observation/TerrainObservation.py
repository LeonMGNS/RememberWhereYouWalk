from jax import numpy as jnp
import jax
from jax.scipy.spatial.transform import Rotation as R
from flax import linen as nn
import jaxkd as jk

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
        self.gaussian_std = dr_height_map_parameters["height_map_gaussian_std"]
        self.z_offset_noise = dr_height_map_parameters["height_map_z_offset_noise"]
        self.height_map_tilt_roll_noise = dr_height_map_parameters["height_map_tilt_roll_noise"]
        self.height_map_tilt_pitch_noise = dr_height_map_parameters["height_map_tilt_pitch_noise"]
        
        self.xy_grid_noise = dr_height_map_parameters["xy_grid_noise"]
        self.avg_kernel_deviation = dr_height_map_parameters["avg_kernel_deviation"]
        self.critic_x_bounds = terrain_parameters["critic_height_map_grid_x_length"]
        self.critic_y_bounds = terrain_parameters["critic_height_map_grid_y_length"]
        self.critic_max_pool_window_size = terrain_parameters["critic_max_pool_window_size"]
        self.knn_lidar_height_map = dr_height_map_parameters["knn_lidar_height_map"]
        self.cam_offset = 0.046825
        

        # Lidar parameters
        self.lidar_horizontal_fov = lidar_parameters["horizontal_fov"]
        self.lidar_vertical_fov = lidar_parameters["vertical_fov"]
        self.lidar_horizontal_step = lidar_parameters["horizontal_step"]
        self.lidar_vertical_step = lidar_parameters["vertical_step"]
        self.min_depth = lidar_parameters["min_depth"]
        self.max_depth = lidar_parameters["max_depth"]
        self.sampling_resolution = lidar_parameters["sampling_resolution"]
        self.lidar_horizontal_noise = lidar_parameters["horizontal_noise"]
        self.lidar_vertical_noise = lidar_parameters["vertical_noise"]
        self.lidar_point_noise = lidar_parameters["point_noise"]
        self.lidar_pos_noise = lidar_parameters["lidar_pos_noise"]
        self.lidar_roll_noise = lidar_parameters["lidar_roll_noise"]
        self.lidar_pitch_noise = lidar_parameters["lidar_pitch_noise"]
        self.lidar_yaw_noise = lidar_parameters["lidar_yaw_noise"]
        self.spinning_step_size = lidar_parameters["spinning_step_size"]
        self.height_shadow_noise = lidar_parameters["height_shadow_noise"]
        self.num_samples = int((self.max_depth-self.min_depth)/self.sampling_resolution)
        t_vals = jnp.linspace(self.min_depth,self.max_depth,self.num_samples)
        self.t_vals = t_vals[None,:,None]
        self.x_min = -self.hf_field_x_half_length_in_meters
        self.x_max = self.hf_field_x_half_length_in_meters
        self.y_min = -self.hf_field_y_half_length_in_meters
        self.y_max = self.hf_field_y_half_length_in_meters
        self.x_range = self.x_max - self.x_min
        self.y_range = self.x_max - self.x_min

        # Create Actor and critic height map
        self.x = jnp.arange(-self.x_bounds, self.x_bounds, self.resolution)
        self.y = jnp.arange(-self.x_bounds, self.y_bounds, self.resolution)
        self.critic_x = jnp.arange(-self.critic_x_bounds,self.critic_x_bounds,self.resolution)
        self.critic_y = jnp.arange(-self.critic_y_bounds,self.critic_y_bounds,self.resolution)
        self.critic_X, self.critic_Y = jnp.meshgrid(self.critic_x,self.critic_y)
        self.critic_xy = jnp.stack([self.critic_X.ravel(),self.critic_Y.ravel()],axis=-1)
        self.X, self.Y = jnp.meshgrid(self.x, self.y)
        self.height_map = jnp.zeros((self.x.shape[0], self.y.shape[0]))
        self.cell_size = self.resolution
        self.xy = jnp.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        self.height_map_points = jnp.zeros((self.X.flatten().shape[0],3))
        self.height_map_points = self.height_map_points.at[:,0].set(self.X.flatten())
        self.height_map_points = self.height_map_points.at[:,1].set(self.Y.flatten())


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

        self.distances = Radii 

        flat_xy = self.polar_xy.reshape(-1, 2)

        # Concatenate a column of zeros for the Z-axis
        self.radial_height_map = jnp.concatenate(
            [flat_xy, jnp.zeros((flat_xy.shape[0], 1))], 
            axis=-1
        )


        xyz_rviz = jnp.array([0.29,0.0,-0.08])
        quat_rviz = jnp.array([0,0,0,1])
        
        R_mat = R.from_quat(quat_rviz).as_matrix()
        b_T_l = jnp.eye(4)
        b_T_l = b_T_l.at[:3,:3].set(R_mat)
        self.b_T_l= b_T_l.at[:3,3].set(xyz_rviz)

        R_mat = R.from_quat(quat_rviz).as_matrix() 
        R_mat_T = R_mat.T

        t_inv = -R_mat_T @ xyz_rviz
        l_T_b = jnp.eye(4)
        l_T_b = l_T_b.at[:3, :3].set(R_mat_T)
        self.l_T_b = l_T_b.at[:3, 3].set(t_inv)

        self.static_kd_tree = jk.build_tree(self.height_map_points[:, :2])




    def generate_lidar_rays_with_noise(self,lidar_horizontal_position:jnp.ndarray,key):
        lidar_horizontal_position = lidar_horizontal_position % 360.0
        # Horizontal angles: 0 to horiz_fov
        horiz_angles = jnp.deg2rad(jnp.arange(0,self.lidar_horizontal_fov, self.lidar_horizontal_step)+lidar_horizontal_position)
        horiz_angles = horiz_angles + jax.random.uniform(key,shape=horiz_angles.shape,minval=-self.lidar_horizontal_noise, maxval=self.lidar_horizontal_noise)
        
        key,_ = jax.random.split(key)
        # Vertical angles: -vert_fov/2 to +vert_fov/2
        vert_angles = jnp.deg2rad(jnp.arange(-self.lidar_vertical_fov, 0 + 1e-6, self.lidar_vertical_step))
        vert_angles = vert_angles + jax.random.uniform(key,shape=vert_angles.shape,minval=-self.lidar_vertical_noise, maxval=self.lidar_vertical_noise)
        key,_ = jax.random.split(key)

        # Meshgrid for all combinations
        ha, va = jnp.meshgrid(horiz_angles, vert_angles, indexing="xy")

        # Convert spherical to Cartesian (unit vectors)
        x = jnp.cos(va) * jnp.cos(ha)
        y = jnp.cos(va) * jnp.sin(ha)
        z = jnp.sin(va)

        rays_local = jnp.stack([x, y, z], axis=-1).reshape(-1, 3)
        rays_local = rays_local / jnp.linalg.norm(rays_local, axis=-1, keepdims=True)
        # Origins: all rays start at (0,0,0) + additional uniform noise
        origins_local = jnp.zeros_like(rays_local) + jax.random.uniform(key,shape=rays_local.shape,minval=-self.lidar_pos_noise, maxval=self.lidar_pos_noise)
        return rays_local,origins_local,key
    
    def generate_lidar_rays(self,lidar_horizontal_position:jnp.ndarray):
        lidar_horizontal_position = lidar_horizontal_position % 360.0
        # Horizontal angles: 0 to horiz_fov
        horiz_angles = jnp.deg2rad(jnp.arange(0,self.lidar_horizontal_fov, self.lidar_horizontal_step)+lidar_horizontal_position)
        # Vertical angles: -vert_fov/2 to +vert_fov/2
        vert_angles = jnp.deg2rad(jnp.arange(-self.lidar_vertical_fov, 0 + 1e-6, self.lidar_vertical_step))

        # Meshgrid for all combinations
        ha, va = jnp.meshgrid(horiz_angles, vert_angles, indexing="xy")

        # Convert spherical to Cartesian (unit vectors)
        x = jnp.cos(va) * jnp.cos(ha)
        y = jnp.cos(va) * jnp.sin(ha)
        z = jnp.sin(va)

        rays_local = jnp.stack([x, y, z], axis=-1).reshape(-1, 3)
        rays_local = rays_local / jnp.linalg.norm(rays_local, axis=-1, keepdims=True)
        # Origins: all rays start at (0,0,0)
        origins_local = jnp.zeros_like(rays_local)
        return rays_local,origins_local 
    
    def transform_lidar_rays(self,state):
        lidar_rays,lidar_origins = self.generate_lidar_rays(state.lidar_horizontal_pos)
        _, _,T_lidar_to_world= self.lidar_transformations(state)
        origins_h = jnp.concatenate([lidar_origins, jnp.ones((lidar_origins.shape[0], 1))], axis=-1)
        origins_world_h = origins_h @ T_lidar_to_world.T
        origins_world = origins_world_h[:, :3]
        # For directions, only rotation matters (ignore translation)
        R = T_lidar_to_world[:3, :3]
        rays_world = (lidar_rays @ R.T)
        rays_world = rays_world / jnp.linalg.norm(rays_world, axis=-1, keepdims=True)

        return rays_world, origins_world

    def transform_lidar_rays_with_noise(self,state,key):
        lidar_rays,lidar_origins,key = self.generate_lidar_rays_with_noise(state.lidar_horizontal_pos,key)
        world_to_lidar, _,lidar_to_world,key= self.lidar_transformations_with_noise(state,key)
        origins_h = jnp.concatenate([lidar_origins, jnp.ones((lidar_origins.shape[0], 1))], axis=-1)
        origins_world_h = origins_h @ lidar_to_world.T
        origins_world = origins_world_h[:, :3]
        # For directions, only rotation matters (ignore translation)
        R = lidar_to_world[:3, :3]
        rays_world = (lidar_rays @ R.T)
        rays_world = rays_world / jnp.linalg.norm(rays_world, axis=-1, keepdims=True)

        return rays_world, origins_world,world_to_lidar,lidar_to_world,key
    
    def get_lidar_points_with_noise(self,state,key):
        rays, origins,world_to_lidar,lidar_to_world,key = self.transform_lidar_rays_with_noise(state,key)
        dir = rays[:,None,:]
        ray_origins = origins[:,None,:]
        positions = ray_origins + self.t_vals * dir
        x = positions[:, :, 0]
        y = positions[:, :, 1]
        z_ray = positions[:, :, 2]
        u = (jnp.clip(x, self.x_min, self.x_max) - self.x_min) / self.x_range
        v = (jnp.clip(y, self.y_min, self.y_max) - self.y_min) / self.y_range
        i = u * ( self.robot.model.hfield_ncol[0] - 1)
        j = v * (self.robot.model.hfield_nrow[0] - 1)
        i0 = jnp.floor(i).astype(int)
        j0 = jnp.floor(j).astype(int)
        h_vals = state.model.hfield_data.reshape((self.robot.model.hfield_nrow[0],self.robot.model.hfield_ncol[0]))[j0, i0] * self.robot.model.hfield_size[0][2]
        below = z_ray <= h_vals
        first_idx = jnp.argmax(below, axis=1)
        has_hit = jnp.any(below, axis=1)
        first_idx = jnp.where(has_hit, first_idx, self.num_samples)
        points = positions[jnp.arange(first_idx.shape[0]), first_idx]
        points = jnp.where((first_idx < self.num_samples).reshape(-1,1),points,jnp.ones_like(points)*1000)
        points = jnp.where(((jnp.abs(points[:,0]) >5.0) | (jnp.abs(points[:,1]) >5.0)).reshape(-1,1),points.at[:,:2].set(jnp.array([1000,1000])),points)
        points = points + jax.random.uniform(key,shape=points.shape,minval=-self.lidar_point_noise,maxval=self.lidar_point_noise)
        key,_ = jax.random.split(key)
        return points,world_to_lidar,lidar_to_world,key

    def get_lidar_points(self,state):
        rays, origins = self.transform_lidar_rays(state)
        dir = rays[:,None,:]
        ray_origins = origins[:,None,:]
        positions = ray_origins + self.t_vals * dir
        x = positions[:, :, 0]
        y = positions[:, :, 1]
        z_ray = positions[:, :, 2]
        x_clipped = jnp.clip(x, self.x_min, self.x_max)
        y_clipped = jnp.clip(y, self.y_min, self.y_max)
        u = (x_clipped - self.x_min) / self.x_range
        v = (y_clipped - self.y_min) / self.y_range
        i = u * ( self.robot.model.hfield_ncol[0] - 1)
        j = v * (self.robot.model.hfield_nrow[0] - 1)
        i0 = jnp.floor(i).astype(int)
        j0 = jnp.floor(j).astype(int)
        h_vals = state.model.hfield_data.reshape((self.robot.model.hfield_nrow[0],self.robot.model.hfield_ncol[0]))[j0, i0] * self.robot.model.hfield_size[0][2]
        below = z_ray <= h_vals
        def find_first_true(arr):
            idx = jnp.argmax(arr)
            is_found = arr[idx]
            idx = jnp.where(is_found, idx, arr.shape[0])
            return idx
        first_idx = jax.vmap(find_first_true)(below)
        ray_ids = jnp.arange(positions.shape[0])
        points = positions[ray_ids, first_idx, :]  # (N_rays, 3)
        points = jnp.where((first_idx < self.num_samples).reshape(-1,1),points,jnp.zeros_like(points))
        points = jnp.where(((jnp.abs(points[:,0]) >5.0) | (jnp.abs(points[:,1]) >5.0)).reshape(-1,1),points.at[:,:2].set(jnp.array([1000,1000])),points)
        return points 



    
    def height_map_from_lidar(self,state,key):
        lidar_points = self.get_lidar_points(state)
        world_to_lidar,_,lidar_to_world = self.lidar_transformations(state)
        height_map_h = jnp.concatenate([self.height_map_points, jnp.ones((self.height_map_points.shape[0], 1))], axis=-1)
        height_map = self.create_height_map(lidar_points,height_map_h)
        height_map_in_lidar_coords = height_map @ world_to_lidar.T
        state = state.replace(lidar_horizontal_pos=state.lidar_horizontal_pos+self.spinning_step_size)
        return height_map_in_lidar_coords[:,2],state,key

    def create_height_map(self,points):
        kd_tree = jk.build_tree(points[:,:2])
        idcs,_ = jk.query_neighbors(kd_tree, self.height_map_points[:,:2], k=3)
        queried_heights = points[idcs,2]
        height_map =  self.height_map_points.at[:,2].set(jnp.mean(queried_heights,axis=1))
        return height_map
    
    def apply_polar_occlusion(self, heights, sensor_z,key):
        # Assume heights is reshaped to (num_rays, num_points_per_ray)
        # Assume self.distances is a precomputed array of shape (num_points_per_ray,) 
        # representing the radial distance of each concentric circle.
        
        # 1. Calculate the slope from the sensor to every point
        # Add a small epsilon to avoid division by zero if a point is exactly at origin
        slopes = (heights - sensor_z) / (self.distances + 1e-5)
        
        # 2. A point is occluded if a closer point has a higher slope.
        # cummax scans along the radial axis (axis=1) and propagates the maximum slope seen so far.
        max_slopes = jax.lax.cummax(slopes, axis=1)
        
        # 3. Calculate what the "perceived" height would be if a ray hit the occluder.
        # This extrudes the wall outward, creating a realistic depth-shadow.
        perceived_heights = 0.0 + jax.random.uniform(key,shape=heights.shape,minval=-self.height_shadow_noise,maxval=self.height_shadow_noise)
        key,_ = jax.random.split(key)
        # 4. Mask the true heights. If the max slope is strictly greater than the point's slope,
        # it is occluded, so we replace it with the perceived shadow height.
        is_occluded = max_slopes > slopes
        realistic_heights = jnp.where(is_occluded, perceived_heights, heights)
        return realistic_heights.flatten(),key
 
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




    def get_height_map(self,state,env,key):
        T_world_to_lidar,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        height_map_points =  jnp.concatenate([self.xy, jnp.zeros_like(self.xy[:, 0]).reshape((-1, 1))], axis=-1)
        noisy_xy_position = (R_world_yaw @ jnp.concatenate([state.height_map_xy_noise.reshape(-1, 1), jnp.zeros((1, 1))], axis=0).reshape(3,))
        
        yaw_rotated_height_map_points = (R_world_yaw @ height_map_points.T).T
        lidar_centered_height_map_points = yaw_rotated_height_map_points + (lidar_pos_in_world + jax.random.uniform(key,shape=(3,), minval=-self.lidar_pos_noise, maxval=self.lidar_pos_noise))
        key,_ = jax.random.split(key)
        noisy_height_map_points = lidar_centered_height_map_points + (noisy_xy_position + jax.random.uniform(key,shape=(3,), minval=-self.xy_grid_noise, maxval=self.xy_grid_noise))
        key,_ = jax.random.split(key)
        heights = self.get_height_from_xpos(state,noisy_height_map_points[:,:2],env)
        height_map = noisy_height_map_points.at[:,2].set(heights)
        noisy_height_map,key = self.apply_height_map_noise(state,height_map,key)
        padded_input = jnp.pad(
            noisy_height_map[:, 2].reshape((1,self.height_map.shape[0], self.height_map.shape[1],1)),
            pad_width=((0,0),(1, 1), (1, 1),(0,0)),  # Add padding to height and width
            mode="constant",
            constant_values=jnp.min(noisy_height_map[:,2])  # Set padding value to 0
        )
        pooled_noisy_heights = jax.lax.conv_general_dilated(padded_input,state.height_map_kernel,window_strides=(1,1),padding="VALID",dimension_numbers=("NHWC", "HWIO", "NHWC"))
        noisy_height_map = noisy_height_map.at[:,2].set(pooled_noisy_heights.reshape(-1))
        noisy_height_map_in_lidar_coords = ((T_world_to_lidar[:3,:3] @ noisy_height_map.T)).T
        noisy_height_map_in_lidar_coords = noisy_height_map_in_lidar_coords.at[:].add(T_world_to_lidar[:3,3])
        actor_heights = noisy_height_map_in_lidar_coords[:,2]
        return actor_heights,key
    
    def get_gr_height_map(self,state,env):
        T_world_to_lidar,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        yaw_rotated_height_map_points = (R_world_yaw @ self.radial_height_map.T).T
        robot_centered_height_map_points = yaw_rotated_height_map_points.at[:,:2].add(state.pipeline_state.qpos[:2])
        raw_heights = self.get_height_from_xpos(state,robot_centered_height_map_points[:,:2],env)
        raw_heights = raw_heights - state.pipeline_state.qpos[2] 
        heights = jnp.clip(raw_heights,a_min=None,a_max=self.max_clipping_height)
        radial_height_map = self.radial_height_map.at[:,2].set(heights) 
        return radial_height_map[:,2]
    
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
    

    def get_noisy_height_map(self,state,env,key):
        T_world_to_lidar,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        sensor_z = lidar_pos_in_world[2]
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        yaw_rotated_height_map_points = (R_world_yaw @ self.radial_height_map.T).T
        robot_centered_height_map_points = yaw_rotated_height_map_points.at[:,:2].add(state.pipeline_state.qpos[:2])
        raw_heights = self.get_height_from_xpos(state, robot_centered_height_map_points[:, :2], env)
        radial_heights_2d = raw_heights.reshape(self.height_map_nr_rays, self.height_map_nr_points_per_ray)
        realistic_heights,key = self.apply_polar_occlusion(radial_heights_2d,sensor_z,key) 
        realistic_heights = realistic_heights - sensor_z
        realistic_heights = jnp.clip(realistic_heights,a_min=None,a_max=self.max_clipping_height)
        radial_height_map = self.radial_height_map.at[:,2].set(realistic_heights)
        noisy_radial_height_map,key = self.apply_height_map_noise(state,radial_height_map,key)
        key,_ = jax.random.split(key)
        return noisy_radial_height_map[:,2],key
        
        

    
    def get_gr_height_map_new(self, state, env):
        T_world_to_lidar,lidar_pos_in_world,_ = self.get_world_to_lidar_transformation(state)
        R_world_yaw = self.update_orientation(state.pipeline_state)[3]
        yaw_rotated_height_map_points = (R_world_yaw @ self.radial_height_map.T).T
        robot_centered_height_map_points = yaw_rotated_height_map_points.at[:,:2].add(state.pipeline_state.qpos[:2])
        raw_heights = self.get_height_from_xpos(state,robot_centered_height_map_points[:,:2],env) 
        raw_heights = jnp.clip(raw_heights,a_min=None,a_max=self.max_clipping_height)
        heights = raw_heights - state.pipeline_state.qpos[2]
        radial_height_map = self.radial_height_map.at[:,2].set(heights) 
        return radial_height_map


        


    def get_hfield_idx_from_xpos(self, coords: jnp.ndarray):
        coords = self.get_hfield_coords_from_xpos(coords)
        idx = (coords[:, 1] * (self.hf_field_length[0]) + coords[:, 0]).astype(jnp.int32)
        return idx

    def get_idx_from_hfield_coords(self, coords: jnp.ndarray):
        idx = (coords[:, 1] * (self.hf_field_length[1]) + coords[:, 0]).astype(jnp.int32)
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

    def get_height_from_xpos_interpolated(self, state, xpos: jnp.ndarray):
        grid_coords = self.get_hfield_coords_from_xpos(xpos)
        grid_ids = self.get_hfield_idx_from_xpos(xpos)
        u = (
            xpos[:, 0]
            - (grid_coords[:, 0] - self.hf_field_length[0] / 2) * (self.cell_size)
        ) / (self.cell_size)
        v = (
            xpos[:, 1]
            - (grid_coords[:, 1] - self.hf_field_length[1] / 2) * (self.cell_size)
        ) / (self.cell_size)

        hfield = state.model.hfield_data

        ij = grid_ids
        i_1j_coords = grid_coords.at[:, 0].add(1)
        i_1j = (i_1j_coords[:, 1] * (self.hf_field_length[1]) + i_1j_coords[:, 0]).astype(
            jnp.int32
        )

        ij_1_coords = grid_coords.at[:, 1].add(1)
        ij_1 = (ij_1_coords[:, 1] * (self.hf_field_length[1]) + ij_1_coords[:, 0]).astype(
            jnp.int32
        )

        i_1j_1_coords = grid_coords.at[:, :].add(1)
        i_1j_1 = (
            i_1j_1_coords[:, 1] * (self.hf_field_length[1]) + i_1j_1_coords[:, 0]
        ).astype(jnp.int32)
        i_1j_coords = jnp.clip(i_1j_coords, 0, self.hf_field_length - 1)
        ij_1_coords = jnp.clip(ij_1_coords, 0, self.hf_field_length - 1)
        i_1j_1_coords = jnp.clip(i_1j_1_coords, 0, self.hf_field_length - 1)

        z = (
            (1 - u) * (1 - v) * hfield[ij]
            + u * (1 - v) * hfield[i_1j]
            + (1 - u) * v * hfield[ij_1]
            + u * v * hfield[i_1j_1]
        ) * self.terrain_max_possible_height

        return z

    def get_height_from_xpos(self, state, xpos: jnp.ndarray,env):
        grid_ids = self.get_hfield_idx_from_xpos(xpos)
        z = state.model.hfield_data[grid_ids] * self.terrain_max_possible_height
        return z

    def apply_height_map_noise(self, state, height_map, key):
        # Apply gausian noise per point

        hfield_noise = jax.random.normal(
            key, shape=height_map[:, 2].shape
        ) * self.gaussian_std
        height_map_with_noise = height_map.at[:, 2].add(state.height_map_offset+hfield_noise)
        tilt_noise_hfield = (state.height_map_tilt @ (height_map_with_noise.T)).T
        outlier_mask = jax.random.uniform(key, shape=(tilt_noise_hfield.shape[0],), minval=0.0, maxval=1.0) < 0.005
        _, key = jax.random.split(key)
        sign_mask = jax.random.choice(key, jnp.array([-1, 1]), shape=(tilt_noise_hfield.shape[0],), p=jnp.array([0.5, 0.5]))
        _, key = jax.random.split(key)
        outlier_heights = jnp.where(outlier_mask, tilt_noise_hfield[:,2] + (sign_mask*0.4),tilt_noise_hfield[:,2])
        noise_height_map = tilt_noise_hfield.at[:,2].set(outlier_heights)
        return noise_height_map, key

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

    def get_feet_in_edge_box(self, state, geom_ids: jnp.ndarray):
        pos = state.pipeline_state.geom_xpos[geom_ids].reshape(-1, 3)
        # Get all edge coordinates
        edges_coords_x = jnp.arange(state.edge_map.shape[0])
        edges_coords_y = jnp.arange(state.edge_map.shape[1])
        edges_coords = jnp.stack(jnp.meshgrid(edges_coords_x, edges_coords_y, indexing='ij'), axis=-1).reshape(-1, 2)
        edges_position = self.get_xpos_from_hfield_coords(edges_coords)
        edges_heights = self.get_height_from_xpos(state, edges_position)
        feet_size = state.model.geom_size[geom_ids, 0]

        def check_in_box(geom_pos, geom_size):
            in_xy = jnp.logical_and(jnp.abs(edges_position[:, 0] - geom_pos[0]) < 0.05, jnp.abs(edges_position[:, 1] - geom_pos[1]) < 0.05)
            distance_to_edge = jnp.min(jnp.linalg.norm(edges_position - geom_pos[:2], axis=1) + (state.edge_map[edges_coords[:, 0], edges_coords[:, 1]] == 0)*1000)
            check_low_edge = jnp.logical_and(state.edge_map[edges_coords[:, 0], edges_coords[:, 1]] == -1, in_xy)
            check_high_edge = jnp.logical_and(state.edge_map[edges_coords[:, 0], edges_coords[:, 1]] == 1, jnp.logical_and(in_xy, edges_heights > geom_pos[2]-geom_size))
            in_box = check_low_edge | check_high_edge
            return jnp.any(in_box), distance_to_edge
        in_box, min_distances = jax.vmap(check_in_box)(pos, feet_size)
        min_distances = jnp.min(min_distances)
        return in_box, min_distances

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
    

    def lidar_transformations(self,state):
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

            T_noise = R.from_euler("xyz",[self.lidar_roll_noise,self.lidar_pitch_noise,self.lidar_yaw_noise]).as_matrix()
            return l_T_w,lidar_pos_in_world,w_T_l
    
    def lidar_transformations_with_noise(self,state,key):
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
            roll_noise = jax.random.uniform(key,minval=-self.lidar_roll_noise,maxval=self.lidar_roll_noise)
            key,_ = jax.random.split(key)
            pitch_noise = jax.random.uniform(key,minval=-self.lidar_pitch_noise,maxval=self.lidar_pitch_noise)
            key,_ = jax.random.split(key)
            yaw_noise = jax.random.uniform(key,minval=-self.lidar_yaw_noise,maxval=self.lidar_yaw_noise)
            key,_ = jax.random.split(key)
            T_noise = R.from_euler("xyz",[roll_noise,pitch_noise,yaw_noise]).as_matrix()
            w_T_l_rot = w_T_l[:3,:3] @ T_noise
            w_T_l = w_T_l.at[:3,:3].set(w_T_l_rot)
            lidar_pos_in_world = w_T_l[:3,3]
            l_T_w = jnp.eye(4)
            
            l_T_w = l_T_w.at[:3,:3].set(w_T_l[:3,:3].T)
            l_T_w = l_T_w.at[:3,3].set(-w_T_l[:3,:3].T @ w_T_l[:3,3])

            

            return l_T_w,lidar_pos_in_world,w_T_l,key


