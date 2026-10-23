import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


class Actor(nn.Module):
    actor_output_size: int
    layer_sizes:list[int]

    @nn.compact
    def __call__(self, data):
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.WeightNorm(
                nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init),
            )

        action_latent = dense_layer(self.layer_sizes[0])(data)
        action_latent = nn.LayerNorm()(action_latent)
        action_latent = nn.elu(action_latent)

        action_latent = dense_layer(self.layer_sizes[1])(action_latent)
        action_latent = nn.elu(action_latent)

        action_latent = dense_layer(self.layer_sizes[2])(action_latent)
        action_latent = nn.elu(action_latent)

        action_mean = nn.Dense(self.actor_output_size,kernel_init=orthogonal(0.01),bias_init=constant(0.0))(action_latent)
        #action_std = nn.Dense(self.actor_output_size, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(action_latent)
        action_std = self.param('log_std', constant(0.0), (self.actor_output_size,))
        # Broadcast the standalone parameter to match the batch dimension of action_mean
        action_std = jnp.broadcast_to(action_std, action_mean.shape)
        return action_mean,action_std


class Critic(nn.Module):
    layer_sizes: list[int]
    @nn.compact
    def __call__(self, data):
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.WeightNorm(
                nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init),
                variable_filter={"kernel"}
            )
        value_latent = dense_layer(self.layer_sizes[0])(data)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(self.layer_sizes[1])(value_latent)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(self.layer_sizes[2])(value_latent)
        value_latent = nn.elu(value_latent)
        value_latent = dense_layer(1,kernel_init=orthogonal(1), bias_init=constant(0.0))(value_latent)
        return value_latent



class HeightEncoder(nn.Module):
    height_encoder_size: int
    layer_sizes: list[int]

    @nn.compact
    def __call__(self, data):
        def dense_layer(units):
            return nn.Dense(units)
        height_latent = dense_layer(self.layer_sizes[0])(data)
        height_latent = nn.elu(height_latent)
        height_latent = dense_layer(self.layer_sizes[1])(data)
        height_latent = nn.elu(height_latent)
        height_latent = dense_layer(self.height_encoder_size)(height_latent)
        return height_latent
