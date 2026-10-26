import jax
import jax.numpy as jnp
from rememberwalk.states.MazeState import MazeState
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from jax.scipy.spatial.transform import Rotation


def get_obs(state: MazeState, env:SimpleMazeEnv, key: jnp.ndarray) -> tuple[MazeState, jnp.ndarray]:
    return compute_obs(state, env, key, hold_every_2nd_step=False)


def compute_obs(state: MazeState, env:SimpleMazeEnv, key: jnp.ndarray, hold_every_2nd_step: bool) -> tuple[MazeState, jnp.ndarray]:
    """Actor/critic observations, height maps and the noisy position of one env.

    ``hold_every_2nd_step`` (small deployment maze) refreshes the height map and the position only on even
    steps and keeps the previous ones otherwise.
    """
    # Get proprioceptive observations + command
    pipeline_state = state.pipeline_state
    _, euler_angles, quat_inv = update_orientation(pipeline_state)
    projected_gravity_vector = quat_inv.apply(jnp.array([0.0, 0.0, -1.0]))
    qpos = (
        pipeline_state.qpos[env.robot.joint_qpos_adr]
        - env.robot.nominal_joint_positions
    ) / 3.14
    qvel = (
        pipeline_state.qvel[env.robot.joint_qvel_adr]
        / env.robot.max_joint_velocities
    )
    trunk_linear_velocity = quat_inv.apply(
        pipeline_state.qvel[env.robot.local_linear_qvel_adr]
    )
    trunk_angular_velocity = pipeline_state.qvel[env.robot.local_angular_qvel_adr]

    trunk_angular_velocity = jnp.clip(trunk_angular_velocity / 50.0, -1.0, 1.0)
    trunk_linear_velocity = jnp.clip(trunk_linear_velocity / 10.0, -1.0, 1.0)
    last_action = state.last_action
    height = pipeline_state.qpos[2].reshape((1,))
    feet_coll = env.collision_fn(state, env.robot.geom_feet_ids, env)

    delta_yaw = (angle_diff(euler_angles[2],state.starting_pos[2])).reshape((1,))
    delta_xy = (state.pipeline_state.qpos[:2]-state.starting_pos[:2]).reshape((2,))

    delta_pos = jnp.concatenate([delta_xy,delta_yaw],axis=-1)
    
    robot_pos = pipeline_state.qpos[:2]
    cell_visits = jnp.clip(state.cell_visits.reshape((-1,))/env.episode_length,0,1)
    critic_obs = jnp.concatenate(
        (
            state.current_target_heading.reshape((2,)),
            state.no_goal_position.reshape((1,)),
            qpos,
            qvel,
            last_action,
            trunk_angular_velocity,
            projected_gravity_vector,
            trunk_linear_velocity,
            height,
            feet_coll,
            robot_pos,
            cell_visits,
        ),
        axis=0,
    )
    critic_obs = jnp.where(jnp.isnan(critic_obs), 1000, critic_obs)
    critic_obs = jnp.clip(critic_obs,-1000,1000)
    obs = critic_obs[:env.actor_observation_size]
    obs, key = env.proprioceptive_obs_dr_handler.randomize_obs(critic_obs[:env.actor_observation_size], key)
    _, key = jax.random.split(key)
    # Get HeightField observations
    hf_obs,key = env.terrain_observation.get_gr_height_map_with_noise(state,env,key)
    critic_hf_obs = env.terrain_observation.get_critic_height_map(state,env)
    _, key = jax.random.split(key)
    height_buffer = jnp.roll(state.height_delay, -1, axis=0)
    height_buffer = height_buffer.at[-1].set(hf_obs)
    chosen_height = jax.random.randint(key, (), 0, env.dr_parameters["height_map_parameters"]["height_map_max_delay"] + 1)
    _, key = jax.random.split(key)
    chosen_height = height_buffer[chosen_height]
    hf_obs = jnp.where(
        jnp.ones(hf_obs.shape[-1]) * env.dr_parameters["height_map_parameters"]["height_map_delay_probability"] < jax.random.uniform(key),
        chosen_height,
        hf_obs,
    )
    if hold_every_2nd_step:
        # Extra split of the deployment variant, kept so its RNG stream stays the same
        _,key = jax.random.split(key)
    yaw = (euler_angles[2]).reshape((1,))
    xy = (state.pipeline_state.qpos[:2]).reshape((2,))
    noisy_true_pos = jnp.concat([xy,yaw]) + jax.random.uniform(key,(3,),minval=-env.dr_parameters["observation_parameters"]["true_pos_noise"],maxval=env.dr_parameters["observation_parameters"]["true_pos_noise"])
    _, key = jax.random.split(key)
    if hold_every_2nd_step:
        hf_obs = jnp.where(((state.n_steps % 2) == 0) *jnp.ones_like(hf_obs),hf_obs,state.height_obs)
        noisy_true_pos = jnp.where(((state.n_steps % 2) == 0)*jnp.ones_like(noisy_true_pos),noisy_true_pos,state.noisy_true_pos)

    state = state.replace(
        obs=obs,
        critic_obs=critic_obs,
        height_obs=hf_obs,
        critic_height_obs=critic_hf_obs,
        height_delay=height_buffer,
        delta_pos=delta_pos,
        noisy_true_pos=noisy_true_pos,
    )
    return state, key


def update_orientation(pipeline_state):
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

def angle_diff(a, b):
    return (a - b + jnp.pi) % (2 * jnp.pi) - jnp.pi
