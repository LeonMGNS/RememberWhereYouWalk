import jax
import jax.numpy as jnp
import flax.linen as nn
import flax
from rememberwalk.actor_critic.history import HistoryActorCritic
# TrainingState and Transition are shared with FixedWindow
from rememberwalk.actor_critic.history import TrainingState, Transition  # noqa: F401

@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    transformer_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray


class EncoderLayer(nn.Module):
    n_heads: int
    feed_forward_hidden: int
    feed_forward_output: int
    @nn.compact
    def __call__(self,data):
        #MultiHead Attention Part
        data = nn.LayerNorm()(data)
        attention_module = nn.MultiHeadDotProductAttention(num_heads=self.n_heads)
        attention_output = attention_module(data,data,data)
        # Skip connection 1
        x = data + attention_output

        # Feedforward Part
        y = nn.LayerNorm()(x)
        y = nn.Dense(self.feed_forward_hidden)(y)
        y = nn.relu(y)
        y = nn.Dense(self.feed_forward_output)(y)

        # Skip connection 2
        y = x+y

        return y

class Encoder(nn.Module):
    num_layers: int
    n_heads: int
    feed_forward_hidden: int
    feed_forward_output: int
    memory_output_features: int

    @nn.compact
    def __call__(self,data):
        def dense_layer(units):
            return nn.Dense(units)
        encoding = dense_layer(256)(data)
        encoding = nn.elu(encoding)
        encoding = dense_layer(self.feed_forward_output)(encoding)
        encoding = nn.elu(encoding)
        x = encoding
        pos_embedding = self.param('pos_embedding', nn.initializers.normal(stddev=0.02),(1, x.shape[1], x.shape[2]))

        x = x + pos_embedding

        for _ in range(self.num_layers):
            x = EncoderLayer(n_heads=self.n_heads,feed_forward_hidden=self.feed_forward_hidden,feed_forward_output = self.feed_forward_output)(x)
        
        x = x.swapaxes(1,2)
        memory_encoding = nn.elu(dense_layer(1)(x)).squeeze(-1)
        return memory_encoding


class ActorCritic(HistoryActorCritic):

    def __init__(self, config: dict):
        super().__init__()
        actor_observation_size         = config["actor_observation_size"]
        actor_output_size              = config["actor_output_size"]
        value_observation_size         = config["value_observation_size"]
        height_sample_size             = config["height_sample_size"]
        critic_height_sample_size      = config["critic_height_sample_size"]
        height_encoder_size            = config["height_encoder_size"]
        transformer_encoded_size       = config["transformer_encoded_size"]
        transformer_hidden_size        = config.get("transformer_hidden_size", transformer_encoded_size)
        transformer_n_heads            = config["transformer_n_heads"]
        transformer_n_layers           = config["transformer_n_layers"]
        transformer_sequence_length    = config["transformer_sequence_length"]
        distribution_size              = config['distribution_size']
        transformer_memory_output_features = config["transformer_memory_output_features"]
        min_scale                      = config.get("min_scale", 0.001)
        max_scale                      = config.get("max_scale", 2.0)
        seed                           = config.get("seed", 0)
        actor_layer_sizes              = config.get("actor_layer_sizes", [512, 256, 128])
        critic_layer_sizes             = config.get("critic_layer_sizes", [512, 256, 128])
        actor_height_encoder_layer_sizes  = config.get("actor_height_encoder_layer_sizes", [256, 128])
        critic_height_encoder_layer_sizes = config.get("critic_height_encoder_layer_sizes", [512, 256])
        
        self.min_scale = min_scale
        self.max_scale = max_scale
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key, 9)
        self.actor_observation_size = actor_observation_size
        self.value_observation_size = value_observation_size
        self.action_size = int(actor_output_size / 2)
        self.height_sample_size = height_sample_size
        self.critic_height_sample_size = critic_height_sample_size
        self.transformer_sequence_length = transformer_sequence_length
        self.transformer_input_size = self.actor_observation_size + height_encoder_size + 3 + 3
        self.transformer_encoded_size = transformer_encoded_size
        self.transformer_memory_output_features = transformer_memory_output_features
        self.height_encoder_size = height_encoder_size
        self.distribution_size = distribution_size
        self.sequence_length = self.transformer_sequence_length
        self.history_input_size = self.transformer_input_size
        self.memory_output_features = self.transformer_memory_output_features

        # Init actor, critic, height encoders and heads.
        # The cross entropy head is sized with transformer_encoded_size (B2, equal to the memory output size in all configs).
        self.init_shared_networks(
            keys={"actor": init_keys[0], "critic": init_keys[1], "height_encoder": init_keys[7], "critic_height_encoder": init_keys[8],
                  "cross_entropy": init_keys[9], "linear_velocity": init_keys[10]},
            actor_memory_size=transformer_memory_output_features,
            head_memory_size=self.transformer_encoded_size,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_height_encoder_layer_sizes=actor_height_encoder_layer_sizes,
            critic_height_encoder_layer_sizes=critic_height_encoder_layer_sizes,
        )

        # Init Transformer
        self.transformer_encoder = Encoder(transformer_n_layers, transformer_n_heads, transformer_hidden_size, transformer_encoded_size, self.transformer_memory_output_features)
        self.init_transformer_nn_params = self.transformer_encoder.init(init_keys[2], jnp.zeros((1,self.transformer_sequence_length, self.transformer_input_size)))

        self.params = Params(
            policy_parameters=self.init_actor_nn_params,
            value_parameters = self.init_critic_nn_params,
            transformer_parameters = self.init_transformer_nn_params,
            cross_entropy_params = self.init_cross_entropy_head_nn_params,
            height_encoder_parameters = self.init_height_encoder_nn_params,
            critic_height_encoder_parameters = self.init_critic_height_encoder_nn_params,
            linear_velocity_parameters = self.init_linear_velocity_head_nn_params,
        )
        self.labels = Params(
            policy_parameters="actor",
            value_parameters="critic",
            height_encoder_parameters="actor",
            critic_height_encoder_parameters="critic",
            transformer_parameters="actor",
            cross_entropy_params = "head",
            linear_velocity_parameters = "head",
        )

    def encode_history(self, parameters: Params, observation_history: jnp.ndarray) -> jnp.ndarray:
        return self.transformer_encoder.apply(parameters.transformer_parameters, observation_history)
