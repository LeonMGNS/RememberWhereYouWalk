"""Auxiliary heads trained next to PPO: linear velocity estimation and maze map reconstruction."""

import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


class LinearVelocityHead(nn.Module):
    """(batch, actor_observation_size) -> (batch, 3) estimated base linear velocity."""

    @nn.compact
    def __call__(self, data):

        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)

        x = dense_layer(128)(data)
        x = nn.elu(x)

        x = dense_layer(64)(x)
        x = nn.elu(x)

        x = dense_layer(3)(x)
        return x


class CrossEntropyHead(nn.Module):
    """(batch, input) -> (batch, distribution_size) wall logits of the maze map, clipped to [-10, 10]."""

    distribution_size: int

    @nn.compact
    def __call__(self, data):

        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)

        x = dense_layer(256)(data)
        x = nn.elu(x)
        x = nn.LayerNorm()(x)

        x = dense_layer(128)(x)
        x = nn.elu(x)

        x = dense_layer(self.distribution_size)(x)
        x = jnp.clip(x, -10, 10)
        return x
