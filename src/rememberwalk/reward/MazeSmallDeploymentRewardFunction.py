import jax.numpy as jnp
from rememberwalk.states.MazeState import MazeState
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from jax.scipy.spatial.transform import Rotation

class MazeReward:
       
    def __init__(self,reward_parameters,robot,dt=0.02):
            self.dt = dt
            self.robot = robot
            self.heading_reward_coeff = self.dt * reward_parameters["heading_reward_coeff"]
            self.heading_front_coeff = self.dt * reward_parameters["heading_front_coeff"]
            self.yaw_velocity_coeff = self.dt * reward_parameters["yaw_velocity_coeff"]
            self.pitch_roll_vel_coeff = self.dt * reward_parameters["pitch_roll_vel_coeff"]
            self.pitch_roll_pos_coeff = self.dt * reward_parameters["pitch_roll_pos_coeff"]
            self.joint_position_limit_coeff = self.dt * reward_parameters["joint_position_limit_coeff"]
            self.joint_velocity_coeff = self.dt * reward_parameters["joint_velocity_coeff"]
            self.joint_acceleration_coeff = self.dt * reward_parameters["joint_acceleration_coeff"]
            self.action_rate_coeff = self.dt * reward_parameters["action_rate_coeff"]
            self.actuator_joint_nominal_diff_coeff = self.dt * reward_parameters["actuator_joint_nominal_diff_coeff"]
            self.base_height_coeff = self.dt * reward_parameters["base_height_coeff"]
            self.symmetry_air_coeff = self.dt * reward_parameters["symmetry_air_coeff"]
            self.alive_clipped_coeff =   self.dt * reward_parameters["alive_clipped_coeff"]
            self.alive_unclipped_coeff =  self.dt * reward_parameters["alive_unclipped_coeff"] 
            self.all_feet_in_air_coeff = self.dt * reward_parameters["all_feet_in_air_coeff"]
            self.feet_clearance_coeff = self.dt * reward_parameters["feet_clearance_coeff"]
            self.geom_below_ground_coeff = self.dt * reward_parameters["geom_below_ground_coeff"]
            self.standing_penalty_coeff = self.dt * reward_parameters["standing_penalty_coeff"]
            self.front_feet_in_air_coeff = self.dt * reward_parameters["front_feet_in_air_coeff"]
            self.joint_torque_coeff = self.dt * reward_parameters["joint_torque_coeff"]
            self.reached_goal_coeff =  self.dt * reward_parameters["reached_goal_coeff"]
            self.lidar_contact_coeff = self.dt * reward_parameters["lidar_contact_coeff"]
            self.z_velocity_coeff = self.dt * reward_parameters["z_velocity_coeff"]
            self.air_time_coeff = self.dt * reward_parameters["air_time_coeff"]
            self.bias_reward_coeff = self.dt * reward_parameters["bias_reward_coeff"]
            self.air_time_max = reward_parameters["air_time_max"]
            self.joint_torque_frac_coeff = self.dt * reward_parameters["joint_torque_frac_coeff"] 
            self.reward_episode_length_factor = reward_parameters["reward_episode_length_factor"]
            self.still_alive_reward_coeff =  self.dt *reward_parameters["still_alive_reward_coeff"]
            self.stall_reward_coeff = self.dt * reward_parameters["stall_reward_coeff"]
            self.cell_visit_reward = self.dt * reward_parameters["cell_visit_reward"]
            self.bias_cell_reward_coeff = self.dt * reward_parameters["bias_cell_reward_coeff"]
            self.maze_coverage_reward_coeff = self.dt * reward_parameters["maze_coverage_reward_coeff"]
            self.exploration_timesteps = reward_parameters["exploration_part"]
            self.first_goal_timestep = reward_parameters["first_goal_part"]
            self.second_goal_timestep = reward_parameters["second_goal_part"]
            self.update_step_for_cell_visit = reward_parameters["update_step_for_cell_visit"]
            self.close_to_wall_reward = self.dt * reward_parameters["close_to_wall_reward"]
            self.centering_coeff = self.dt * reward_parameters["centering_coeff"]
            self.joint_velocity_limit_coeff = self.dt * reward_parameters["joint_velocity_limit_coeff"]
            self.power_draw_frac_penalty_coeff = self.dt * reward_parameters["power_draw_frac_penalty_coeff"] 
            self.joint_actuator_force_limited = robot.model.actuator_forcelimited[robot.actuators_id]
            self.joint_actuator_joint_force_limited = robot.model.jnt_actfrclimited[robot.actuator_joint_ids]
            

    def get_reward(self, state:MazeState, action:jnp.ndarray,env:SimpleMazeEnv) -> MazeState:
        curriculum_coeff = state.curr_coeff
        data = state.pipeline_state
        _, orientation_euler, quat_inv = self.update_orientation(data)
        current_global_linear_velocity = data.qvel[:3]
        current_local_linear_velocity = quat_inv.apply(current_global_linear_velocity)
        current_local_angular_velocity = data.qvel[3:6]


        # Angular velocity reward
        current_local_angular_velocity = data.qvel[3:6]
        angular_velocity_norm = jnp.sum(jnp.square(current_local_angular_velocity[:2]))
        angular_velocity_reward = (
            curriculum_coeff * self.pitch_roll_vel_coeff * -angular_velocity_norm
        )
        
        # Alive clipped reward
        alive_clipped_reward = self.alive_clipped_coeff * 1.0

        # Alive unclipped reward
        alive_unclipped_reward =  self.alive_unclipped_coeff * 1.0

        # Joint position limit reward
        joint_positions = data.qpos[self.robot.joint_qpos_adr]
        lower_limit_penalty = -jnp.minimum(
            joint_positions - self.robot.soft_joint_limits[:, 0], 0.0
        ).mean()
        upper_limit_penalty = jnp.maximum(
            joint_positions - self.robot.soft_joint_limits[:, 1], 0.0
        ).mean()
        joint_position_limit_reward = (
            curriculum_coeff
            * self.joint_position_limit_coeff
            * -(lower_limit_penalty + upper_limit_penalty)
        )
        # Joint velocity reward
        joint_velocities = jnp.abs(data.qvel[self.robot.joint_qvel_adr])
        soft_joint_limits = self.robot.max_joint_velocities * 0.9
        joint_velocitiy_penalty = jnp.mean(jnp.maximum(
            joint_velocities - soft_joint_limits, 0.0))
        joint_velocity_limit_reward = curriculum_coeff * self.joint_velocity_limit_coeff * -joint_velocitiy_penalty

        joint_velocity_norm = jnp.mean(jnp.square(joint_velocities))
        
        joint_velocity_reward = curriculum_coeff * self.joint_velocity_coeff * -joint_velocity_norm

        # Joint acceleration reward
        acceleration_norm = jnp.mean(
            jnp.square(
                (state.prev_joint_vel - data.qvel[self.robot.joint_qvel_adr]) / self.dt
            )
        )
        acceleration_reward = (
            curriculum_coeff * self.joint_acceleration_coeff * -acceleration_norm
        )
        # Joint torque reward
        joint_torques = data.qfrc_actuator[self.robot.joint_qvel_adr] 
        actuator_force_capacity = jnp.abs(state.model.actuator_forcerange[self.robot.actuators_id]).max(axis=1)
        joint_force_capacity =jnp.abs(state.model.jnt_actfrcrange[self.robot.actuator_joint_ids]).max(axis=1)
        joint_actuator_force_capacity = jnp.minimum(
            jnp.where(self.joint_actuator_force_limited, actuator_force_capacity, jnp.inf),
            jnp.where(self.joint_actuator_joint_force_limited, joint_force_capacity, jnp.inf),
        )
        joint_actuator_force_capacity_available = jnp.isfinite(joint_actuator_force_capacity) & (joint_actuator_force_capacity > 0.0)
        safe_joint_actuator_force_capacity = jnp.where(joint_actuator_force_capacity_available, joint_actuator_force_capacity, 1.0)
        joint_actuator_max_velocities = env.robot.max_joint_velocities
        force_fraction = jnp.where(joint_actuator_force_capacity_available, joint_torques / safe_joint_actuator_force_capacity, 0.0)
        power_fraction = jnp.where(joint_actuator_force_capacity_available, jnp.maximum(joint_torques * data.qvel[self.robot.joint_qvel_adr], 0.0) / (safe_joint_actuator_force_capacity * joint_actuator_max_velocities), 0.0)
        nr_force_capacity_available = jnp.maximum(jnp.sum(joint_actuator_force_capacity_available), 1)
        torque_norm = jnp.sum(jnp.square(force_fraction)) / nr_force_capacity_available
        power_draw = jnp.sum(power_fraction) / nr_force_capacity_available
        torque_reward = curriculum_coeff * self.joint_torque_frac_coeff * -torque_norm
        power_draw_penalty_reward = curriculum_coeff * self.power_draw_frac_penalty_coeff * -power_draw

        # Action rate reward
        action_rate_norm = jnp.mean(jnp.square(action - state.last_action))
        action_rate_reward = (
            curriculum_coeff * self.action_rate_coeff * -action_rate_norm
        )
        # Joint nominal position difference reward
        actuator_joint_nominal_diff_norm = jnp.nan_to_num(jnp.mean(jnp.square(data.qpos[self.robot.joint_qpos_adr]- self.robot.nominal_joint_positions)), nan=0.0)
        actuator_joint_nominal_diff_reward = curriculum_coeff * self.actuator_joint_nominal_diff_coeff * -actuator_joint_nominal_diff_norm

        # Walking height 
        trunk_pos = data.qpos[:3]
        floor_height = env.terrain_observation.get_height_from_xpos(
            state, trunk_pos[:2].reshape(1, 2),env
        )
        height_difference_squared = jnp.clip(((trunk_pos[2] - floor_height) - self.robot.nominal_trunk_z) ** 2,0,1.0)
        base_height_reward = (
            curriculum_coeff * self.base_height_coeff * -height_difference_squared
        )[0]
        feet_floor_contacts = env.collision_fn(state,self.robot.geom_feet_ids,env)
        
        
        # Angular position reward
        roll_pitch_position_norm = jnp.sum(jnp.square(orientation_euler[:2]))
        angular_position_reward = (
            curriculum_coeff * self.pitch_roll_pos_coeff * -roll_pitch_position_norm
        )
        
        # Air time reward
        air_time_reward = jnp.mean(
                jnp.minimum(
                    jnp.where(
                        feet_floor_contacts, state.last_touch - self.air_time_max, 0.0
                    ),
                0.0
                )
            )
        
        air_time_reward = curriculum_coeff * self.air_time_coeff * air_time_reward

        # Symmetry air reward
        first_statement = jnp.logical_and(
            ~feet_floor_contacts[0], ~feet_floor_contacts[1]
        )
        second_statement = jnp.logical_and(
            ~feet_floor_contacts[2], ~feet_floor_contacts[3]
        )
        symmetry_air_reward = (  
            curriculum_coeff
            * self.symmetry_air_coeff
            * -1
            * (first_statement + second_statement)
        )
        # Geom below ground reward
        floor_heights = env.terrain_observation.get_height_from_xpos(state,state.pipeline_state.geom_xpos[self.robot.non_feet_geom_ids,:2],env)
        geom_heights = state.pipeline_state.geom_xpos[self.robot.non_feet_geom_ids,2] - floor_heights
        geom_done = jnp.any(geom_heights < 0.0)
        geom_reward = -1 * curriculum_coeff * self.geom_below_ground_coeff * jnp.sum(geom_done)
        # Linear velocity reward
        z_velocity_squared = current_local_linear_velocity[2] ** 2
        linear_velocity_reward = (
            curriculum_coeff * self.z_velocity_coeff * -z_velocity_squared
        )
        # Lidar Collison reward
        lidar_contact =env.contact_fn(data._impl.contact,self.robot.lidar_id,self.robot.geom_ids[-1]) # Check if front lidar is colliding with front geom

        lidar_contact_reward =  -1 * lidar_contact * self.lidar_contact_coeff
        # Goal Reward (from Advanced Skills Paper)

        
        # Stall Reward (from Advanced skills paper)
        veloctiy = jnp.linalg.norm(data.qvel[:2])
        stall_reward = - curriculum_coeff * self.stall_reward_coeff * ((0.1-veloctiy) *(veloctiy<0.1))


        # Cell Visit Reward (Groundtrugh "intrinsic" motivation for new cells)
        state,nr_cell_visited,current_cell_ids = self.update_cell_visists(state,env)
        
        cell_visit_reward =   (self.cell_visit_reward * jnp.exp(-(nr_cell_visited/200)))

        cell_visit_reward = jnp.where(state.no_goal_position,cell_visit_reward,jnp.zeros_like(cell_visit_reward))

        # Maze coverage reward
        coverage = (state.cell_visits > 0).mean()
        maze_coverage_reward = self.maze_coverage_reward_coeff * coverage
        maze_coverage_reward = jnp.where(state.no_goal_position,maze_coverage_reward,jnp.zeros_like(maze_coverage_reward))
        

        #Close to wall Reward
        height_field_xpos_positions = env.terrain_generator.height_field_xpos_positions
        height_field_xpos_heights = env.terrain_observation.get_height_from_xpos(state,height_field_xpos_positions,env)
        height_field_xpos_positions = jnp.where((height_field_xpos_heights<0.05)[:,None],1000*jnp.ones_like(height_field_xpos_positions),height_field_xpos_positions)
        min_height_field_xpos_distance = jnp.min(jnp.linalg.norm(height_field_xpos_positions-state.pipeline_state.qpos[:2],axis=-1))
        close_to_wall_reward = -1 * self.close_to_wall_reward * (min_height_field_xpos_distance<0.2) * jnp.exp(-5*(min_height_field_xpos_distance-0.1))    

        # Bias Cell reward
        neighbor_positions = env.terrain_generator.cell_center_positions
        walls_tensor = env.terrain_generator.check_walls_between_cell_and_neighbors(state.maze_description,current_cell_ids)
        old_neighbor_positions_distances = jnp.linalg.norm(neighbor_positions-state.pipeline_state.qpos[:2],axis=-1) + state.cell_visits
        neighbor_positions_distances = jnp.ones_like(old_neighbor_positions_distances)* 1000
        neighbor_positions_distances = neighbor_positions_distances.at[current_cell_ids[0],current_cell_ids[1]+1].set(old_neighbor_positions_distances[current_cell_ids[0],current_cell_ids[1]+1]+walls_tensor[0]*1000)
        neighbor_positions_distances = neighbor_positions_distances.at[current_cell_ids[0]+1,current_cell_ids[1]].set(old_neighbor_positions_distances[current_cell_ids[0]+1,current_cell_ids[1]]+walls_tensor[1]*1000)
        neighbor_positions_distances = neighbor_positions_distances.at[current_cell_ids[0],current_cell_ids[1]-1].set(old_neighbor_positions_distances[current_cell_ids[0],current_cell_ids[1]-1]+walls_tensor[2]*1000)
        neighbor_positions_distances = neighbor_positions_distances.at[current_cell_ids[0]-1,current_cell_ids[1]].set(old_neighbor_positions_distances[current_cell_ids[0]-1,current_cell_ids[1]]+walls_tensor[3]*1000)
        nearest_neighbor_idx = jnp.argmin(neighbor_positions_distances.reshape((-1,1)),axis=0)
        distance_to_goal_position_old = jnp.linalg.norm(state.goal_position-state.pipeline_state.qpos[:2])
        goal_position = jnp.where(((distance_to_goal_position_old < 0.25)*jnp.ones_like(state.goal_position)).astype(jnp.int32) |(state.n_steps==1),neighbor_positions.reshape((-1,2))[nearest_neighbor_idx],state.goal_position)[0]
        distance_to_goal_position = jnp.linalg.norm(goal_position-state.pipeline_state.qpos[:2])
        cosine_similarity = (jnp.dot(data.qvel[:2], (goal_position - state.pipeline_state.qpos[:2]))) /((jnp.linalg.norm(data.qvel[:2]) * distance_to_goal_position)+1e-6)     
        bias_cell_reward =  self.bias_cell_reward_coeff * (cosine_similarity+1)/2 * jnp.clip(jnp.linalg.norm(data.qvel[:2]),0,1)
        bias_cell_reward = bias_cell_reward +self.reached_goal_coeff * (state.n_steps > 1) * (distance_to_goal_position_old<0.25)
        
        bias_cell_reward = jnp.where(state.no_goal_position,jnp.zeros_like(bias_cell_reward),bias_cell_reward)
        
        row, col = current_cell_ids[0], current_cell_ids[1]
        centers = env.terrain_generator.cell_center_positions          # (R, C, 2)
        cell_center = centers[row, col]
        delta = state.pipeline_state.qpos[:2] - cell_center            # (dx, dy)

        # Recover the (x,y) direction of a +1 step in row / col straight from the grid,
        # so we don't have to assume how (row,col) maps to world axes.
        row_axis = centers[1, 0] - centers[0, 0]
        col_axis = centers[0, 1] - centers[0, 0]
        row_axis = row_axis / (jnp.linalg.norm(row_axis) + 1e-9)
        col_axis = col_axis / (jnp.linalg.norm(col_axis) + 1e-9)

        # Same neighbor order used elsewhere in your code: [+col, +row, -col, -row]; 1 == wall
        walls = env.terrain_generator.check_walls_between_cell_and_neighbors(
            state.maze_description, current_cell_ids
        )
        dir_vecs = jnp.stack([col_axis, row_axis, -col_axis, -row_axis], axis=0)  # (4, 2)

        proj = dir_vecs @ delta                                        # signed deviation toward each neighbor
        half = 0.5                                                   
        toward_wall = walls * jnp.square(jnp.maximum(proj, 0.0) / half)  # only penalize toward blocked sides
        centering_penalty = -1 * self.centering_coeff * jnp.sum(toward_wall)

        
        # Bias Moving forward
        v_world = data.qvel[:2]
        theta = orientation_euler[2]
        # Rotation matrix world → body
        R = jnp.array([
            [ jnp.cos(theta),  jnp.sin(theta)],
            [-jnp.sin(theta),  jnp.cos(theta)]
        ])
        v_body = R @ v_world
        bias_heading_front_reward =  curriculum_coeff *self.heading_front_coeff * (jnp.sign(v_body[0]))
        # Cross Entropy Reward


        base_reward =  cell_visit_reward + maze_coverage_reward + bias_heading_front_reward + bias_cell_reward + alive_clipped_reward + centering_penalty + close_to_wall_reward + stall_reward

        always_active_penalty = (geom_reward 
                                    + joint_position_limit_reward
                                    + joint_velocity_limit_reward 
                                    + joint_velocity_reward  
                                    + actuator_joint_nominal_diff_reward  
                                    + torque_reward
                                    + action_rate_reward
                                    + angular_velocity_reward
                                    + acceleration_reward
                                    + base_height_reward
                                    + symmetry_air_reward
                                    + linear_velocity_reward
                                    + air_time_reward
                                    + angular_position_reward
                                    + power_draw_penalty_reward
                                    )
        
        reward = jnp.clip(base_reward + always_active_penalty,0) + alive_unclipped_reward

        info = {
            "reward": reward,
            "angular_vel_reward": angular_velocity_reward,
            "joint_position_limit_reward": joint_position_limit_reward,
            "acceleration_reward": acceleration_reward,
            "torque_reward": torque_reward,
            "action_rate_reward": action_rate_reward,
            "base_height_reard": base_height_reward,
            "symmetry_air_reward": symmetry_air_reward,
            "joint_velocity_reward": joint_velocity_reward,
            "joint_nominal_diff_reward": actuator_joint_nominal_diff_reward,
            "geom_below_ground_reward": geom_reward,
            "curr_coeff": state.curr_coeff,
            "terrain_curr_coeff": state.terrain_curr_coeff,
            "lidar_contact_reward": lidar_contact_reward,
            "linear_velocity_reward": linear_velocity_reward,
            "air_time_reward": air_time_reward,
            "angular_position_reward": angular_position_reward,
            "cell_visit_reward": cell_visit_reward,
            "bias_cell_reward":bias_cell_reward,
            "stall_reward": stall_reward,
            "bias_heading_front_reward": bias_heading_front_reward,
            "active_penalty": always_active_penalty,
            "reward_scaling_factor":state.reward_scaling_factor,
            "close_to_wall_reward":close_to_wall_reward,
            "centering_penalty": centering_penalty,
            "power_draw_penalty_reward": power_draw_penalty_reward,
            "maze_coverage_reward": maze_coverage_reward,
            "clipped_fraction": (base_reward + always_active_penalty < 0).astype(jnp.float32),
            "is_stalled": (veloctiy < 0.1).astype(jnp.float32),
        }

        summed_reward = state.summed_reward + reward
        summed_reward = jnp.nan_to_num(summed_reward,nan=0.0, posinf=0.0, neginf=0.0)
        state = state.replace(
            reward=reward,
            metrics=info,
            summed_reward=summed_reward,
            goal_position=goal_position,
        )
        return state


    def update_orientation(self,pipeline_state):
            orientation_quat = Rotation.from_quat(
                [
                    pipeline_state.qpos[4],
                    pipeline_state.qpos[5],
                    pipeline_state.qpos[6],
                    pipeline_state.qpos[3],
                ]
            )
            orientation_euler = orientation_quat.as_euler("xyz")
            orietation_quat_inv = orientation_quat.inv()

            return orientation_quat, orientation_euler, orietation_quat_inv


    def angle_diff(self,a, b):
            return (a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi
    

    def update_cell_visists(self,state:MazeState,env:SimpleMazeEnv):
        current_cell_idcs = env.terrain_generator.get_cell_idxs_from_maze(state)
        nr_visits= state.cell_visits[current_cell_idcs[0],current_cell_idcs[1]]
        new_cell_visits = state.cell_visits.at[current_cell_idcs[0],current_cell_idcs[1]].add(1)
        updated_cell_visits = jnp.where((((state.n_steps % self.update_step_for_cell_visit)==0)) * jnp.ones_like(new_cell_visits),new_cell_visits,state.cell_visits)
        state = state.replace(cell_visits=updated_cell_visits)
        return state, nr_visits,current_cell_idcs
