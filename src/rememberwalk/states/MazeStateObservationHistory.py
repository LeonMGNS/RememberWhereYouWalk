from mujoco import mjx
import jax.numpy as jnp
import jax
from typing import Dict
from flax import struct



@struct.dataclass
class MazeStateObservationHistory:
    model: mjx.Model
    pipeline_state: mjx.Data
    last_action: jax.Array
    prev_joint_vel: jax.Array
    obs: jax.Array
    critic_obs: jax.Array
    reward: jax.Array
    done: jax.Array
    last_touch: jax.Array
    goal_position: jax.Array
    wall_position: jax.Array
    curr_coeff: jax.Array
    current_target_heading: jax.Array
    summed_reward: jnp.float32
    action_delay: jax.Array
    height_obs: jax.Array
    height_delay: jax.Array
    critic_height_obs: jax.Array
    truncation: jnp.int32
    n_steps: jnp.int32
    max_allowed_curr_coeff: jnp.float32
    max_allowed_terrain_curr_coeff: jnp.float32
    extrinsic_motor_strength: jax.Array
    extrinsic_p_gain_factor: jax.Array
    extrinsic_d_gain_factor: jax.Array
    extrinsic_position_offset: jax.Array
    maze_nr:jnp.int32
    terrain_curr_coeff: jnp.float32
    height_map_kernel: jax.Array
    height_map_tilt: jax.Array
    height_map_offset: jnp.float32
    lidar_horizontal_pos:jax.Array
    height_map_xy_noise: jax.Array
    bias_reward_active: jax.Array
    time_to_reach_goal: jnp.float32
    cell_visits: jax.Array
    first_goal_reward_given: jax.Array
    reached_goals:jax.Array
    second_goal_reward_given: jax.Array
    starting_pos: jax.Array
    delta_pos:jax.Array
    noisy_true_pos:jnp.ndarray
    travelled_distance: jnp.float32
    nr_goals_reached: jnp.int32
    exploration_phase_length: jnp.int32
    maze_description: jax.Array
    no_goal_position: jnp.bool_
    sampled_terrain_curr_coeff: jnp.float32
    reward_scaling_factor: jnp.float32
    exponential_curr_coeff: jnp.float32
    exponential_terrain_curr_coeff: jnp.float32
    proprio_cell_values: jax.Array
    max_loc_error: jax.Array
    use_pred_loc: jax.Array
    avg_loc_error: jax.Array
    observation_history: jax.Array
    metrics: Dict[str, jax.Array]
    