"""Per-env state of the maze task (``imports.state``)."""

import jax
import jax.numpy as jnp
from flax import struct
from mujoco import mjx


@struct.dataclass
class MazeState:
    """State of one maze env. Field order is the pytree order; fields marked P4 are obsolete."""

    # MJX model of this env (domain-randomised, carries the maze height field)
    model: mjx.Model
    # MJX simulation data
    pipeline_state: mjx.Data
    # (action_size,) action of the previous step
    last_action: jax.Array
    # B13: read by the acceleration reward but never updated after init
    prev_joint_vel: jax.Array
    # (actor_observation_size,) actor observation
    obs: jax.Array
    # privileged critic observation
    critic_obs: jax.Array
    # reward of the last step
    reward: jax.Array
    # episode ended this step
    done: jax.Array
    # per foot: last contact time, for the air-time reward
    last_touch: jax.Array
    # (2,) goal x, y in the maze frame (origin in the exploration task)
    goal_position: jax.Array
    # unused (P4)
    wall_position: jax.Array
    # task curriculum level in [curr_coeff, max_curr_coeff]
    curr_coeff: jax.Array
    # (2,) goal position in the body frame (zero without a goal)
    current_target_heading: jax.Array
    # return of the running episode
    summed_reward: jnp.float32
    # buffer of delayed actions (domain randomisation)
    action_delay: jax.Array
    # (height_sample_size,) noisy, delayed actor height map
    height_obs: jax.Array
    # buffer of past height maps (delay randomisation)
    height_delay: jax.Array
    # (critic_height_sample_size,) noise-free critic height map
    critic_height_obs: jax.Array
    # episode ended by the time limit
    truncation: jnp.int32
    # steps in the running episode
    n_steps: jnp.int32
    # curriculum ceiling of this env
    max_allowed_curr_coeff: jnp.float32
    # terrain curriculum ceiling of this env
    max_allowed_terrain_curr_coeff: jnp.float32
    # per-joint torque scale (domain randomisation)
    extrinsic_motor_strength: jax.Array
    # per-joint P gain factor (domain randomisation)
    extrinsic_p_gain_factor: jax.Array
    # per-joint D gain factor (domain randomisation)
    extrinsic_d_gain_factor: jax.Array
    # per-joint target offset (domain randomisation)
    extrinsic_position_offset: jax.Array
    # index of the maze in the terrain generator's maze set
    maze_nr: jnp.int32
    # terrain curriculum level (wall density)
    terrain_curr_coeff: jnp.float32
    # height map noise parameters (domain randomisation)
    height_map_kernel: jax.Array
    # height map tilt (domain randomisation)
    height_map_tilt: jax.Array
    # height map offset (domain randomisation)
    height_map_offset: jnp.float32
    # unused, only reset (P4)
    lidar_horizontal_pos: jax.Array
    # height map x, y shift (domain randomisation)
    height_map_xy_noise: jax.Array
    # unused (P4)
    bias_reward_active: jax.Array
    # unused (P4)
    time_to_reach_goal: jnp.float32
    # (rows, cols) steps spent in each cell of the cell visit grid
    cell_visits: jax.Array
    # unused (P4)
    first_goal_reward_given: jax.Array
    # unused (P4)
    reached_goals: jax.Array
    # unused (P4)
    second_goal_reward_given: jax.Array
    # (3,) start x, y, yaw of the episode
    starting_pos: jax.Array
    # (3,) x, y, yaw relative to the start
    delta_pos: jax.Array
    # unused, only reset (P4)
    travelled_distance: jnp.float32
    # unused, only reset (P4)
    nr_goals_reached: jnp.int32
    # unused (P4)
    exploration_phase_length: jnp.int32
    # (distribution_size,) wall map of the maze: target of the cross entropy head
    maze_description: jax.Array
    # exploration episode (no goal)
    no_goal_position: jnp.bool_
    # terrain level sampled for this episode's maze
    sampled_terrain_curr_coeff: jnp.float32
    # only logged as a metric
    reward_scaling_factor: jnp.float32
    # step multiplier of the next curriculum increase
    exponential_curr_coeff: jnp.float32
    # step multiplier of the next terrain curriculum increase
    exponential_terrain_curr_coeff: jnp.float32
    # (3,) noisy x, y (m) and yaw (rad), input of the memories
    noisy_true_pos: jax.Array
    # unused since MazeCellDR was removed (P4)
    proprio_cell_values: jax.Array
    # unused, only reset (P4)
    max_loc_error: jax.Array
    # unused, drawn at reset (B8)
    use_pred_loc: jax.Array
    # unused, only reset (P4)
    avg_loc_error: jax.Array
    # per-step values logged to wandb
    metrics: dict[str, jax.Array]
