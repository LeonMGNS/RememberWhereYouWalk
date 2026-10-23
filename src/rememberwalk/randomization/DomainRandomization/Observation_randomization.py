import jax.numpy as jnp
import jax


class ObsDR:

    def __init__(
        self,
        parameters,
    ):
        dr_parameters = parameters["dr_parameters"]
        self.joint_position_noise = dr_parameters["joint_position_noise"]
        self.joint_velocity_noise = dr_parameters["joint_velocity_noise"]
        self.trunk_angular_velocity_noise = dr_parameters["trunk_angular_velocity_noise"]
        self.trunk_linear_velocity_noise = dr_parameters["trunk_linear_velocity_noise"]
        self.gravity_vector_noise = dr_parameters["gravity_vector_noise"]
        self.height_noise = dr_parameters["height_noise"]
        self.joint_position_ids = parameters["joint_position_ids"]
        self.joint_vel_ids = parameters["joint_vel_ids"]
        self.last_action_ids = parameters["last_action_ids"]
        self.linear_velocity_ids = parameters["linear_velocity_ids"]
        self.angular_velocity_ids = parameters["angular_velocity_ids"]
        self.projected_gravity_vector_ids = parameters["projected_gravity_vector_ids"] 

    def randomize_obs(self, obs: jnp.ndarray, key):

        keys = jax.random.split(key, 6)
        joint_position_noise = jax.random.uniform(
            keys[0],
            (12,),
            minval=-self.joint_position_noise,
            maxval=self.joint_position_noise,
        )
        joint_velocity_noise = jax.random.uniform(
            keys[1],
            (12,),
            minval=-self.joint_velocity_noise,
            maxval=self.joint_velocity_noise,
        )
        trunk_angular_velocity_noise = jax.random.uniform(
            keys[2],
            (3,),
            minval=-self.trunk_angular_velocity_noise,
            maxval=self.trunk_angular_velocity_noise,
        )
        gravity_vector_noise = jax.random.uniform(
            keys[3],
            (3,),
            minval=-self.gravity_vector_noise,
            maxval=self.gravity_vector_noise,
        )

        obs = obs.at[self.joint_position_ids].add(joint_position_noise)
        obs = obs.at[self.joint_vel_ids].add(joint_velocity_noise)
        obs = obs.at[self.angular_velocity_ids].add(trunk_angular_velocity_noise)
        obs = obs.at[self.projected_gravity_vector_ids].add(gravity_vector_noise)
        return obs, keys[-1]
