"""FixedWindow baseline (``actor_critic.FixedWindow.ActorCritic``).

An MLP over the flattened history window.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

# TrainingState and Transition are shared with the Transformer
from rememberwalk.actor_critic.history import HistoryActorCritic, TrainingState, Transition  # noqa: F401


@flax.struct.dataclass
class Params:
    """Network parameters. Field names are part of the checkpoint format: do not rename."""

    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    fixed_window_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray


class FixedWindowEncoder(nn.Module):
    """Flatten the observation window and run a plain MLP to a feature vector.

    Input:  (batch, sequence_length, window_input_size)
    Output: (batch, memory_output_features)

    The whole window is flattened into a single vector, so for a window of
    ``sequence_length`` steps each of size ``window_input_size`` the first dense
    layer sees ``sequence_length * window_input_size`` inputs (e.g. 10 * 600 =
    6000). The LayerNorm after the flatten keeps that wide input well-scaled.
    """

    memory_output_features: int
    layer_sizes: tuple = (512, 256)

    @nn.compact
    def __call__(self, data):
        x = data.reshape((data.shape[0], -1))  # (batch, sequence_length * window_input_size)
        x = nn.LayerNorm()(x)
        for units in self.layer_sizes:
            x = nn.Dense(units)(x)
            x = nn.elu(x)
        x = nn.Dense(self.memory_output_features)(x)
        x = nn.elu(x)
        return x


class ActorCritic(HistoryActorCritic):
    """FixedWindow actor-critic; ``config`` is the actor-critic dict of ``generate_setup``."""

    def __init__(self, config: dict):
        super().__init__()
        actor_observation_size = config["actor_observation_size"]
        actor_output_size = config["actor_output_size"]
        value_observation_size = config["value_observation_size"]
        height_sample_size = config["height_sample_size"]
        critic_height_sample_size = config["critic_height_sample_size"]
        height_encoder_size = config["height_encoder_size"]
        distribution_size = config["distribution_size"]
        # Window-encoder dims. Falls back to the transformer config keys so the
        # exact same config used for the transformer works here unchanged.
        window_sequence_length = config.get(
            "fixed_window_sequence_length", config["transformer_sequence_length"]
        )
        window_memory_output_features = config.get(
            "fixed_window_memory_output_features", config["transformer_memory_output_features"]
        )
        window_layer_sizes = tuple(config.get("fixed_window_layer_sizes", [512, 256]))
        min_scale = config.get("min_scale", 0.001)
        max_scale = config.get("max_scale", 2.0)
        seed = config.get("seed", 0)
        actor_layer_sizes = config.get("actor_layer_sizes", [512, 256, 128])
        critic_layer_sizes = config.get("critic_layer_sizes", [512, 256, 128])
        actor_height_encoder_layer_sizes = config.get("actor_height_encoder_layer_sizes", [256, 128])
        critic_height_encoder_layer_sizes = config.get("critic_height_encoder_layer_sizes", [512, 256])

        self.min_scale = min_scale
        self.max_scale = max_scale
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key, 9)

        self.actor_observation_size = actor_observation_size
        self.value_observation_size = value_observation_size
        self.actor_output_size = actor_output_size
        self.action_size = int(actor_output_size / 2)
        self.height_sample_size = height_sample_size
        self.critic_height_sample_size = critic_height_sample_size
        self.height_encoder_size = height_encoder_size
        self.distribution_size = distribution_size
        self.window_sequence_length = window_sequence_length
        self.window_input_size = self.actor_observation_size + height_encoder_size + 3 + 3
        self.window_memory_output_features = window_memory_output_features
        self.sequence_length = self.window_sequence_length
        self.history_input_size = self.window_input_size
        self.memory_output_features = self.window_memory_output_features

        # Init actor, critic, height encoders and heads
        self.init_shared_networks(
            keys={
                "actor": init_keys[0],
                "critic": init_keys[1],
                "cross_entropy": init_keys[3],
                "linear_velocity": init_keys[4],
                "height_encoder": init_keys[6],
                "critic_height_encoder": init_keys[7],
            },
            actor_memory_size=window_memory_output_features,
            head_memory_size=self.window_memory_output_features,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_height_encoder_layer_sizes=actor_height_encoder_layer_sizes,
            critic_height_encoder_layer_sizes=critic_height_encoder_layer_sizes,
        )

        # Init Fixed Window Encoder (flatten + MLP)
        self.fixed_window_encoder = FixedWindowEncoder(self.window_memory_output_features, window_layer_sizes)
        self.init_fixed_window_nn_params = self.fixed_window_encoder.init(
            init_keys[2], jnp.zeros((1, self.window_sequence_length, self.window_input_size))
        )

        self.params = Params(
            policy_parameters=self.init_actor_nn_params,
            value_parameters=self.init_critic_nn_params,
            fixed_window_parameters=self.init_fixed_window_nn_params,
            cross_entropy_params=self.init_cross_entropy_head_nn_params,
            height_encoder_parameters=self.init_height_encoder_nn_params,
            critic_height_encoder_parameters=self.init_critic_height_encoder_nn_params,
            linear_velocity_parameters=self.init_linear_velocity_head_nn_params,
        )
        self.labels = Params(
            policy_parameters="actor",
            value_parameters="critic",
            height_encoder_parameters="actor",
            critic_height_encoder_parameters="critic",
            fixed_window_parameters="actor",
            cross_entropy_params="head",
            linear_velocity_parameters="head",
        )

    def encode_history(self, parameters: Params, observation_history: jnp.ndarray) -> jnp.ndarray:
        """(batch, sequence_length, history_input_size) -> (batch, memory_output_features)."""
        return self.fixed_window_encoder.apply(parameters.fixed_window_parameters, observation_history)
