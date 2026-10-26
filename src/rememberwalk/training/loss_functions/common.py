"""Pieces shared by every PPO loss function."""
import jax
import jax.numpy as jnp


def interpolate_entropy_cost(min_entropy_cost, max_entropy_cost, training_steps, max_training_steps=6_000_000_000.0):
    """Entropy cost falling linearly from max to min over max_training_steps."""
    interpolation_parameter = jnp.clip(training_steps/max_training_steps,0,1)
    entropy_cost = max_entropy_cost - interpolation_parameter*(max_entropy_cost-min_entropy_cost)
    return entropy_cost


def compute_gae(rewards, baseline, next_value, dones, truncations, discount_factor, gae_lambda):
    """Generalised advantage estimates (not normalised) over the time axis 0, as (T, ...).

    Terminations cut the bootstrap. After a truncation the env has already reset, so next_value belongs
    to the next episode: those steps get no TD error and stop the recursion.
    """
    termination = dones * (1 - truncations)
    truncation_mask = 1 - truncations
    deltas = rewards + discount_factor * next_value * (1-termination) - baseline
    deltas *= truncation_mask
    acc = jnp.zeros_like(deltas[0])
    def gae_step(carry, target):
        lambda_, acc = carry
        truncation_mask, delta, termination= target
        acc = delta + discount_factor * (1-termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)
    _, accs = jax.lax.scan(gae_step, (gae_lambda, acc), (truncation_mask, deltas, termination), length=int(termination.shape[0]), reverse=True)
    return accs
