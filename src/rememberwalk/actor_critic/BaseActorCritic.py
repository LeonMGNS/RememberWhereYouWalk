import jax.numpy as jnp


class BaseActorCritic:

    def log_prob(self, actions, loc, scale):
        log_unnormalized = -0.5 * jnp.square((actions - loc) / scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        return jnp.sum(log_unnormalized - log_normalization, axis=-1)

    def entropy(self, loc, scale):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        entropy = 0.5 + log_normalization
        entropy = jnp.sum(entropy, axis=-1)
        return entropy
