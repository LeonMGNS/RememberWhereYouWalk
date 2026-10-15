import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic
from rememberwalk.actor_critic.MLP import Actor, Critic, HeightEncoder
import optax
import flax

@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    transformer_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray

@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    training_steps: jnp.ndarray

@flax.struct.dataclass
class Transition:
    done: jnp.ndarray
    truncation: jnp.ndarray
    actor_observation: jnp.ndarray
    actor_height_observation:jnp.ndarray
    critic_observation: jnp.ndarray
    critic_height_observation: jnp.ndarray
    delta_position:jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    critic_next_observation: jnp.ndarray
    critic_next_height_observation: jnp.ndarray
    log_prob: jnp.ndarray
    true_distribution: jnp.ndarray
    linear_velocity:jnp.ndarray
    noisy_true_pos:jnp.ndarray


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
    
class LinearVelocityHead(nn.Module):

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
    distribution_size: int

    @nn.compact
    def __call__(self,data):
        
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)
        
        x = dense_layer(256)(data)
        x = nn.elu(x)
        x = nn.LayerNorm()(x)

        x = dense_layer(128)(x)
        x = nn.elu(x)

        x = dense_layer(self.distribution_size)(x)
        x = jnp.clip(x,-10,10)
        return x
    
class ActorCritic(BaseActorCritic):

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
        self.actor_output_ize = actor_output_size
        self.action_size = int(actor_output_size / 2)
        self.height_sample_size = height_sample_size
        self.critic_height_sample_size = critic_height_sample_size
        self.memory_buffer_size = int(self.actor_observation_size + self.height_sample_size)
        self.transformer_sequence_length = transformer_sequence_length
        self.transformer_input_size = self.actor_observation_size + height_encoder_size + 3 + 3
        self.transformer_encoded_size = transformer_encoded_size
        self.transformer_memory_output_features = transformer_memory_output_features
        self.height_encoder_size = height_encoder_size
        self.distribution_size = distribution_size

        # Init Actor
        self.actor = Actor(actor_output_size=int(actor_output_size / 2), layer_sizes=actor_layer_sizes)
        self.init_actor_nn_params = self.actor.init(init_keys[0], jnp.zeros((1, actor_observation_size + height_encoder_size + transformer_memory_output_features)))

        # Init Critic
        self.critic = Critic(layer_sizes=critic_layer_sizes)
        self.init_critic_nn_params = self.critic.init(init_keys[1], jnp.zeros((1, value_observation_size + 2 * height_encoder_size)))

        # Init Cross Entropy Head
        self.cross_entropy_head = CrossEntropyHead(int(self.distribution_size))
        self.init_cross_entropy_head_nn_params = self.cross_entropy_head.init(init_keys[9],jnp.zeros((1,self.actor_observation_size+self.height_encoder_size+self.transformer_encoded_size)))

        #Init Linear Velocity head
        self.linear_velocity_head = LinearVelocityHead()
        self.init_linear_velocity_head_nn_params = self.linear_velocity_head.init(init_keys[10],jnp.zeros((1,self.actor_observation_size)))

        # Init Height Encoder
        self.height_encoder = HeightEncoder(height_encoder_size=height_encoder_size, layer_sizes=actor_height_encoder_layer_sizes)
        self.critic_height_encoder = HeightEncoder(height_encoder_size=2 * height_encoder_size, layer_sizes=critic_height_encoder_layer_sizes)
        self.init_height_encoder_nn_params = self.height_encoder.init(init_keys[7], jnp.zeros((1, height_sample_size)))
        self.init_critic_height_encoder_nn_params = self.critic_height_encoder.init(init_keys[8], jnp.zeros((1, critic_height_sample_size)))
        self.sqrt_height_sample_size = jnp.sqrt(critic_height_sample_size).astype(jnp.int32)

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

    def create_optimizer_and_training_state(self, train_cfg: dict):
        """Create a single Adam optimizer and the initial TrainingState."""
        # Create optimizer
        learning_rate = train_cfg["learning_rate"]
        critic_learning_rate = train_cfg["critic_learning_rate"]
        clip_norm = train_cfg["optimizer_clip_norm"]
        eps = train_cfg["optimizer_eps"]
        weight_decay = train_cfg["optimizer_weight_decay"]
        actor_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate, eps=eps))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=critic_learning_rate, eps=eps, weight_decay=weight_decay))
        head_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=train_cfg.get("head_learning_rate"), eps=eps, weight_decay=weight_decay))

        optimizer = optax.multi_transform(
            transforms={'actor': actor_optimizer, 'critic': critic_optimizer, "head": head_optimizer},
            param_labels=self.labels,
        ) 
        training_state = TrainingState(
            optimizer_state=optimizer.init(self.params),
            params=self.params,
            training_steps=jnp.zeros((), dtype=jnp.float32),
        )
        return optimizer, training_state

    def create_dictionaries(self, eval_cfg: dict):
        """Return initialised visualizer and eval dicts.

        Both dicts include ``observation_history`` so callbacks can maintain
        the transformer's context window across steps without depending on a
        particular state type.
        """
        nr_eval_envs = eval_cfg.get("nr_eval_envs", 1)
        empty_history_single = jnp.zeros((self.transformer_sequence_length, self.transformer_input_size))
        empty_history_batch  = jnp.zeros((nr_eval_envs, self.transformer_sequence_length, self.transformer_input_size))

        visualizer_dict = {
            "action":              jnp.zeros((self.action_size,)),
            "observation_history": empty_history_single,
        }
        eval_dict = {
            "observation_history": empty_history_batch,
        }
        return visualizer_dict, eval_dict


    def sample_action(self, parameters:Params ,observation:jnp.ndarray, height_observation:jnp.ndarray,observation_history: jnp.ndarray,key:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        transformer_parameters = parameters.transformer_parameters
        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        linear_velocity_head_nn_params = parameters.linear_velocity_parameters
        
        # Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(linear_velocity_head_nn_params,observation.reshape((-1,self.actor_observation_size)))

        transformer_encoding=self.transformer_encoder.apply(transformer_parameters,observation_history.reshape(-1, self.transformer_sequence_length, self.transformer_input_size))
        transformer_encoding = transformer_encoding.reshape((-1,self.transformer_memory_output_features))
        
        
        loc,log_scale= self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,transformer_encoding], axis=-1))
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        action = jax.random.normal(key, shape=loc.shape) * scale + loc
        log_prob = self.log_prob(action, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)
        
        # Cross Entropy Head setp
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), transformer_encoding,jax.lax.stop_gradient(height_encoded)],axis=-1))
        return action,scale, log_prob,height_encoded,predicted_distribution,pred_linear_vel

    def get_deterministic_action(self, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray,observation_history:jnp.ndarray):

        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        actor_transformer_parameters = parameters.transformer_parameters
        linear_velocity_head_nn_params = parameters.linear_velocity_parameters
        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))
        
        #Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(linear_velocity_head_nn_params,observation.reshape((-1,self.actor_observation_size)))
        
        transformer_encoding= self.transformer_encoder.apply(actor_transformer_parameters,observation_history.reshape(-1,self.transformer_sequence_length,self.transformer_input_size))
        transformer_encoding = transformer_encoding.reshape((-1,self.transformer_memory_output_features))

        loc,scale= self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,transformer_encoding], axis=-1))
        
        # Cross Entropy Head setp
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), transformer_encoding,jax.lax.stop_gradient(height_encoded)],axis=-1))
        
        return loc,predicted_distribution,height_encoded,pred_linear_vel

    def evaluate_actions(self, parameters:Params ,observation:jnp.ndarray, height_observation:jnp.ndarray):
        value_nn_params = parameters.value_parameters
        height_encoder_params = parameters.critic_height_encoder_parameters
        height_encoded = self.critic_height_encoder.apply(height_encoder_params, height_observation.reshape(-1, self.critic_height_sample_size))
        height_encoded = height_encoded.reshape((-1, int(self.critic_height_encoder.height_encoder_size)))
        value = self.critic.apply(value_nn_params,jnp.concatenate([observation.reshape((-1,self.value_observation_size)),height_encoded],axis=-1))
        return value
    

    def log_prob_from_observation(self, actions, parameters:Params, observation:jnp.ndarray, height_observation:jnp.ndarray,observation_history:jnp.ndarray):
        actor_nn_params = parameters.policy_parameters
        height_encoder_nn_params = parameters.height_encoder_parameters
        transformer_parameters = parameters.transformer_parameters
        actions = actions.reshape((-1,self.action_size))

        height_encoded = self.height_encoder.apply(height_encoder_nn_params, height_observation.reshape((-1, self.height_sample_size)))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        transformer_encoding= self.transformer_encoder.apply(transformer_parameters,observation_history.reshape(-1,self.transformer_sequence_length,self.transformer_input_size))
        transformer_encoding = transformer_encoding.reshape((-1,self.transformer_memory_output_features))

        loc,log_scale= self.actor.apply(actor_nn_params, jnp.concatenate([observation.reshape((-1,self.actor_observation_size)),height_encoded,transformer_encoding], axis=-1))
        log_scale = jnp.clip(log_scale,jnp.log(self.min_scale),jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        log_prob= self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob,axis=-1)

        
        # Cross Entropy Head setp
        predicted_distribution = self.cross_entropy_head.apply(parameters.cross_entropy_params,jnp.concatenate([observation.reshape((-1,self.actor_observation_size)), transformer_encoding,jax.lax.stop_gradient(height_encoded)],axis=-1))

        return log_prob, loc, scale, predicted_distribution


        

