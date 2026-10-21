import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def linear_velocity_target(qpos: jnp.ndarray, qvel: jnp.ndarray) -> jnp.ndarray:
    """Target of the linear velocity head for a single env: [v_x, v_y, yaw_rate] in the body frame.

    For the MuJoCo free joint qvel[0:3] is the linear velocity in the world frame and
    qvel[3:6] the angular velocity in the body frame. The head only sees body-frame
    proprioception, so the linear part is rotated into the body frame, the same way the
    observation function does it.
    """
    world_to_body = Rotation.from_quat(jnp.array([qpos[4], qpos[5], qpos[6], qpos[3]])).inv()
    body_linear_velocity = world_to_body.apply(qvel[0:3])
    return jnp.concatenate([body_linear_velocity[:2], qvel[5:6]], axis=-1)
