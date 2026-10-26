"""Binary cross-entropy between the predicted and the true maze map, and the rewards derived from it.

Shared by the eval/experiment callbacks, the trajectory collectors and the loss functions.
"""
import jax.numpy as jnp
import optax


def compute_bce_loss(predicted_distribution: jnp.ndarray, true_distribution: jnp.ndarray) -> jnp.ndarray:
    """Mean BCE over the map cells: logits and 0/1 targets of shape (..., cells) -> (...)."""
    loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution, true_distribution)
    loss = jnp.mean(loss, axis=-1)
    return loss


def compute_bce_reward(predicted_distribution: jnp.ndarray, true_distribution: jnp.ndarray, env) -> jnp.ndarray:
    """Map reconstruction reward used in training: 0 at a BCE of 0.30 or more, rising linearly to 1 at 0,
    scaled by ``env.dt`` and ``reward_parameters.bce_reward``."""
    loss = compute_bce_loss(predicted_distribution, true_distribution)
    bce_reward = (
        env.dt
        * env.parameters["reward_parameters"]["bce_reward"]
        * jnp.clip(0.30 - loss, 0.0, 0.30) / 0.30  # normalize to [0, 1]
    )
    return bce_reward


def compute_exponential_bce_reward(bce_error: jnp.ndarray, env) -> jnp.ndarray:
    """``env.dt * bce_reward * exp(-5 * bce_error)``: the variant logged by the training-time deterministic eval."""
    return env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5 * bce_error)
