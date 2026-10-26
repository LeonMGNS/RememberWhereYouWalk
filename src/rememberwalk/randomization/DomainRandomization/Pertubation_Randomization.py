"""Push perturbations (``imports.pertubation_randomization_handler``)."""

import jax
import jax.numpy as jnp


class DefaultDomainPerturbation:
    """Random pushes of the base (velocity kicks), scaled with the curriculum."""

    def __init__(self, parameters):
        self.push_velocity_x_min = parameters["push_velocity_x_min"]
        self.push_velocity_x_max = parameters["push_velocity_x_max"]
        self.push_velocity_y_min = parameters["push_velocity_y_min"]
        self.push_velocity_y_max = parameters["push_velocity_y_max"]
        self.push_velocity_z_min = parameters["push_velocity_z_min"]
        self.push_velocity_z_max = parameters["push_velocity_z_max"]
        self.add_chance = parameters["add_chance"]
        self.additive_multiplier = parameters["additive_multiplier"]

    def sample(self, data, key, curr_coef):
        """Pushes the base: adds (``add_chance``) or sets a random linear velocity. Returns (data, key)."""
        _, key = jax.random.split(key)
        sampled_push_velocity_x = jax.random.uniform(
            key, minval=self.push_velocity_x_min, maxval=self.push_velocity_x_max
        )
        _, key = jax.random.split(key)
        sampled_push_velocity_y = jax.random.uniform(
            key, minval=self.push_velocity_y_min, maxval=self.push_velocity_y_max
        )
        _, key = jax.random.split(key)
        sampled_push_velocity_z = jax.random.uniform(
            key, minval=self.push_velocity_z_min, maxval=self.push_velocity_z_max
        )

        def true_func():
            return (
                jnp.array(
                    [
                        sampled_push_velocity_x,
                        sampled_push_velocity_y,
                        sampled_push_velocity_z,
                    ]
                )
                * self.additive_multiplier
                * curr_coef
                + data.qvel[0:3]
            )

        def false_func():
            return curr_coef * jnp.array(
                [
                    sampled_push_velocity_x,
                    sampled_push_velocity_y,
                    sampled_push_velocity_z,
                ]
            )

        _, key = jax.random.split(key)
        qvel = jax.lax.cond(jax.random.uniform(key) < self.add_chance, true_func, false_func)
        data = data.replace(qvel=data.qvel.at[0:3].set(qvel))
        return data, key
