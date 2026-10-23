import jax.numpy as jnp
import jax
from jax.scipy.spatial.transform import Rotation
from mujoco import mjx


class InitStateRandomization:

    def __init__(self, parameters, robot,terrain_mode=False):
        self.joint_pos_noise = parameters["joint_pos_noise"]
        self.joint_vel_scale = parameters["joint_vel_scale"]
        self.linear_vel_noise = parameters["linear_vel_noise"]
        self.angular_vel_noise = parameters["angular_vel_noise"]
        self.roll_angle_noise = parameters["roll_angle_noise"]
        self.pitch_angle_noise = parameters["pitch_angle_noise"]
        self.yaw_angle_noise = parameters["yaw_angle_noise"]
        self.joint_pos_offset = parameters["joint_pos_offset"]
        self.terrain_mode = terrain_mode
        self.robot = robot

    def init_state_rand(self, state,env, key):
        pipeline_state = state.pipeline_state
        model = state.model
        curr_coeff = state.curr_coeff
        keys = jax.random.split(key, 8)
        def interpolate_param(initial, min_val, max_val, key):
            random_value = jax.random.uniform(key, shape=initial.shape)  # Sample in [0,1]
            sampled_value = (1 - random_value) * min_val + random_value * max_val  # Interpolates in [min_val, max_val]
            new_value = initial + curr_coeff * (sampled_value - initial)  # Scale deviation from initial
            return jnp.clip(new_value, min_val, max_val)

        roll_angle = interpolate_param(
            jnp.zeros(1), -self.roll_angle_noise, self.roll_angle_noise, keys[0])[0]
        pitch_angle = interpolate_param(
            jnp.zeros(1), -self.pitch_angle_noise, self.pitch_angle_noise, keys[1])[0]
        yaw_angle = interpolate_param(
            jnp.zeros(1), -self.yaw_angle_noise, self.yaw_angle_noise, keys[2])[0]
        quaternion = Rotation.from_euler(
            "xyz", jnp.array([roll_angle, pitch_angle, yaw_angle])
        ).as_quat()
        quaternion = jnp.array(
            [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]
        )
        joint_positions_noise = interpolate_param(jnp.ones(self.robot.joint_qpos_adr.shape), jnp.ones(self.robot.joint_qpos_adr.shape)-self.joint_pos_noise,jnp.ones(self.robot.joint_qpos_adr.shape)+ self.joint_pos_noise, keys[3])
        joint_positions = self.robot.nominal_joint_positions * joint_positions_noise
        joint_positions += interpolate_param(jnp.zeros(self.robot.joint_qpos_adr.shape), -self.joint_pos_offset, self.joint_pos_offset, keys[4])
        joint_positions = jnp.clip(joint_positions,self.robot.soft_joint_limits[:,0], self.robot.soft_joint_limits[:,1])

        joint_velocities = interpolate_param(
            jnp.zeros(self.robot.joint_qvel_adr.shape),
            -self.joint_vel_scale * self.robot.max_joint_velocities,
            self.joint_vel_scale * self.robot.max_joint_velocities,
            keys[5],
        )
        linear_velocities= interpolate_param(
            jnp.zeros(3),
            -self.linear_vel_noise,
            self.linear_vel_noise,
            keys[6],
        )
        angular_velocities = interpolate_param(
            jnp.zeros(3),
            -self.angular_vel_noise,
            self.angular_vel_noise,
            keys[7],
        )
        qpos = self.robot.initial_qpos
        qpos = qpos.at[3:7].set(quaternion)
        qpos = qpos.at[self.robot.joint_qpos_adr].set(joint_positions)
        qvel = jnp.zeros_like(pipeline_state.qvel)
        qvel = qvel.at[:3].set(linear_velocities)
        qvel = qvel.at[3:6].set(angular_velocities)
        qvel = qvel.at[self.robot.joint_qvel_adr].set(joint_velocities)
        feet_ids = env.robot.geom_feet_ids
        
        tmp_data = pipeline_state.replace(
            qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.robot.action_size)
        )
        data = mjx.step(model, tmp_data)
        feet_pos = data.geom_xpos[feet_ids]
        heights = env.terrain_observation.get_height_from_xpos(
            state, feet_pos[:, :2].reshape(-1, 2),env
        )
        min_feet_z_pos = jnp.min(
            data.geom_xpos[feet_ids, 2]
            - heights
            - model.geom_size[feet_ids, 0]
        )
        
        offset = -min_feet_z_pos
        qpos = data.qpos.at[2].add(offset)
        new_pipeline_state = pipeline_state.replace(qpos=qpos, qvel=qvel)
        
        default_qpos = self.robot.initial_qpos
        default_qpos = new_pipeline_state.qpos.at[3:].set(default_qpos[3:])
        default_qvel = jnp.zeros_like(qvel)
        tmp_data_2 = pipeline_state.replace(
            qpos=default_qpos, qvel=default_qvel, ctrl=jnp.zeros(self.robot.action_size)
        )
        data_2 = mjx.step(model, tmp_data_2)
        feet_pos_2 = data_2.geom_xpos[feet_ids]
        heights_2 = env.terrain_observation.get_height_from_xpos(
            state, feet_pos_2[:, :2].reshape(-1, 2),env
        )
        min_feet_z_pos_2 = jnp.min(
            data_2.geom_xpos[feet_ids, 2]
            - heights_2
            - model.geom_size[feet_ids, 0]
        )
        default_offset = -min_feet_z_pos_2

        default_pipeline_state = tmp_data_2.replace(qpos=data_2.qpos.at[2].add(default_offset))

        qpos_height = env.terrain_observation.get_height_from_xpos(
            state, qpos[:2].reshape(-1,2),env
        )
        qpos_above_nominal = new_pipeline_state.qpos[2] - qpos_height > 0.5 * (1 - curr_coeff) * self.robot.nominal_trunk_z
        

        qpos_not_too_much_higher_than_start = new_pipeline_state.qpos[2] < (1.5 * self.robot.nominal_trunk_z)     
        
        use_new_pipeline_state = qpos_above_nominal[0] & qpos_not_too_much_higher_than_start
        
        pipeline_state = data.replace(
            qpos=qpos)

        pipeline_state = jax.lax.cond(use_new_pipeline_state,lambda: new_pipeline_state,lambda: default_pipeline_state)
        return pipeline_state, keys[-1]
