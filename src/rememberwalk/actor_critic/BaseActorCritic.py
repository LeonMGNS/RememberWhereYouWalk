import jax.numpy as jnp




class BaseActorCritic:

    def __init__(self):
        pass

    def sample_action(self):
        pass

    def get_deterministic_action():
        pass

    def evaluate_actions():
        pass

    def log_prob_from_observation():
        pass

    def log_prob(self, actions, loc, scale):
        log_unnormalized = -0.5 * jnp.square((actions - loc) / scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        return jnp.sum(log_unnormalized - log_normalization, axis=-1)

    def entropy(self, loc, scale):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        entropy = 0.5 + log_normalization
        entropy = jnp.sum(entropy, axis=-1)
        return entropy