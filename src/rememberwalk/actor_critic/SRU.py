import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, lecun_normal
from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic
from rememberwalk.actor_critic.MLP import Actor, Critic, HeightEncoder
import optax
import flax

# --------------------------------------------------------------------------- #
# Spatially-Enhanced Recurrent Unit (SRU), Yang et al. 2025, arXiv:2506.05997  #
#                                                                              #
# NOTE: To be a drop-in replacement for the LSTM ablation, this file keeps the #
# exact same interface and field names as the LSTM actor-critic               #
# (``lstm_parameters``, ``last_lstm_state``, ``lstm_state``, ``update_lstm``,   #
# carry = (c, h)). Only the recurrent cell differs, so the PPO loss / scan     #
# code does not need to change.                                                #
# --------------------------------------------------------------------------- #


@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    lstm_parameters: jnp.ndarray          # SRU parameters (name kept for trainer compatibility)
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
    last_lstm_state: jnp.ndarray


@flax.struct.dataclass
class Transition:
    done: jnp.ndarray
    truncation: jnp.ndarray
    actor_observation: jnp.ndarray
    actor_height_observation: jnp.ndarray
    critic_observation: jnp.ndarray
    critic_height_observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    critic_next_observation: jnp.ndarray
    critic_next_height_observation: jnp.ndarray
    log_prob: jnp.ndarray
    true_distribution: jnp.ndarray
    linear_velocity: jnp.ndarray
    lstm_state: jnp.ndarray
    noisy_true_pos: jnp.ndarray


class SRUCell(nn.Module):
    """Single SRU step.

    Standard LSTM gates plus an input-dependent spatial term s_t that multiplies
    the candidate pre-activation element-wise:

        i_t = sigmoid(W_xi x_t + W_hi h_{t-1} + b_i)
        f_t = sigmoid(W_xf x_t + W_hf h_{t-1} + b_f)
        o_t = sigmoid(W_xo x_t + W_ho h_{t-1} + b_o)
        s_t = W_xs x_t + b_s
        g_t = tanh( s_t * (W_xg x_t + W_hg h_{t-1} + b_g) )

    use_refined_gate=True  ("SRU-Ours", refined gate of Gu et al. 2020):
        r_t = i_t * (1 - (1 - f_t)^2) + (1 - i_t) * f_t^2
        c_t = r_t * c_{t-1} + (1 - r_t) * g_t
    use_refined_gate=False ("SRU-LSTM"):
        c_t = f_t * c_{t-1} + i_t * g_t

        h_t = o_t * tanh(c_t)

    Carry layout is (c, h), identical to flax's LSTM cells.
    """
    features: int
    use_refined_gate: bool = True
    spatial_bias_init: float = 1.0  # s_t ~ 1 at init -> cell starts close to a plain LSTM

    @nn.compact
    def __call__(self, carry, x):
        c, h = carry

        # Fused gate pre-activations: [i, f, g, o]
        input_proj = nn.Dense(4 * self.features, kernel_init=lecun_normal(), bias_init=constant(0.0), name="input_proj")(x)
        hidden_proj = nn.Dense(4 * self.features, use_bias=False, kernel_init=orthogonal(), name="hidden_proj")(h)
        z = input_proj + hidden_proj
        i_pre, f_pre, g_pre, o_pre = jnp.split(z, 4, axis=-1)

        # Spatial transformation term (input only, as in the paper)
        s = nn.Dense(self.features, kernel_init=lecun_normal(), bias_init=constant(self.spatial_bias_init), name="spatial_proj")(x)

        i = nn.sigmoid(i_pre)
        f = nn.sigmoid(f_pre)
        o = nn.sigmoid(o_pre)
        g = jnp.tanh(s * g_pre)

        if self.use_refined_gate:
            r = i * (1.0 - (1.0 - f) ** 2) + (1.0 - i) * f ** 2
            new_c = r * c + (1.0 - r) * g
        else:
            new_c = f * c + i * g

        new_h = o * jnp.tanh(new_c)
        return (new_c, new_h), new_h


class SRU(nn.Module):
    """SRU core with the same pre/post-processing as the LSTM ablation.

    Two ELU dense layers in, one SRU step, one ELU dense projection out, so the
    heads receive a feature vector of size ``memory_output_features``.
    """
    hidden_size: int
    memory_output_features: int
    use_refined_gate: bool = True

    @nn.compact
    def __call__(self, carry, x):
        # Input projection (identical to the LSTM ablation).
        y = nn.Dense(256)(x)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.elu(y)

        # Single SRU step.
        new_carry, h = SRUCell(features=self.hidden_size, use_refined_gate=self.use_refined_gate)(carry, y)

        # Output projection to the feature size consumed by the heads.
        memory_encoding = nn.Dense(self.memory_output_features)(h)
        memory_encoding = nn.elu(memory_encoding)
        return new_carry, memory_encoding


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


class ActorCritic(BaseActorCritic):

    def __init__(self, config: dict):
        super().__init__()
        actor_observation_size            = config["actor_observation_size"]
        actor_output_size                 = config["actor_output_size"]
        value_observation_size            = config["value_observation_size"]
        height_sample_size                = config["height_sample_size"]
        critic_height_sample_size         = config["critic_height_sample_size"]
        height_encoder_size               = config["height_encoder_size"]
        distribution_size                 = config["distribution_size"]
        # SRU sizes; fall back to the LSTM / transformer keys so existing configs keep working.
        lstm_hidden_size                  = config.get("sru_hidden_size", config.get("lstm_hidden_size", config.get("transformer_hidden_size", 256)))
        lstm_memory_output_features       = config.get("sru_memory_output_features", config.get("lstm_memory_output_features", config.get("transformer_memory_output_features")))
        use_refined_gate                  = config.get("sru_use_refined_gate", True)  # True: SRU-Ours, False: SRU-LSTM
        min_scale                         = config.get("min_scale", 0.001)
        max_scale                         = config.get("max_scale", 2.0)
        seed                              = config.get("seed", 0)
        actor_layer_sizes                 = config.get("actor_layer_sizes", [512, 256, 128])
        critic_layer_sizes                = config.get("critic_layer_sizes", [512, 256, 128])
        actor_height_encoder_layer_sizes  = config.get("actor_height_encoder_layer_sizes", [256, 128])
        critic_height_encoder_layer_sizes = config.get("critic_height_encoder_layer_sizes", [512, 256])

        self.min_scale = min_scale
        self.max_scale = max_scale
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key, 11)

        self.actor_observation_size = actor_observation_size
        self.value_observation_size = value_observation_size
        self.actor_output_size = actor_output_size
        self.action_size = int(actor_output_size / 2)
        self.height_sample_size = height_sample_size
        self.critic_height_sample_size = critic_height_sample_size
        self.height_encoder_size = height_encoder_size
        self.distribution_size = distribution_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_memory_output_features = lstm_memory_output_features
        self.use_refined_gate = use_refined_gate
        # Per-timestep SRU input: [observation, height_encoded, pred_linear_velocity(3), true_pos(3)]
        self.lstm_input_size = actor_observation_size + height_encoder_size + 3 + 3

        # Init Actor
        self.actor = Actor(actor_output_size=int(actor_output_size / 2), layer_sizes=actor_layer_sizes)
        self.init_actor_nn_params = self.actor.init(
            init_keys[0], jnp.zeros((1, actor_observation_size + height_encoder_size + lstm_memory_output_features))
        )

        # Init Critic
        self.critic = Critic(layer_sizes=critic_layer_sizes)
        self.init_critic_nn_params = self.critic.init(
            init_keys[1], jnp.zeros((1, value_observation_size + 2 * height_encoder_size))
        )

        # Init SRU (attribute name ``lstm`` kept for trainer compatibility)
        self.lstm = SRU(hidden_size=lstm_hidden_size, memory_output_features=lstm_memory_output_features, use_refined_gate=use_refined_gate)
        dummy_carry = (jnp.zeros((1, lstm_hidden_size)), jnp.zeros((1, lstm_hidden_size)))
        self.init_lstm_nn_params = self.lstm.init(init_keys[2], dummy_carry, jnp.zeros((1, self.lstm_input_size)))

        # Init Cross Entropy Head
        self.cross_entropy_head = CrossEntropyHead(int(self.distribution_size))
        self.init_cross_entropy_head_nn_params = self.cross_entropy_head.init(
            init_keys[3],
            jnp.zeros((1, self.actor_observation_size + self.height_encoder_size + self.lstm_memory_output_features)),
        )

        # Init Linear Velocity Head
        self.linear_velocity_head = LinearVelocityHead()
        self.init_linear_velocity_head_nn_params = self.linear_velocity_head.init(
            init_keys[4], jnp.zeros((1, self.actor_observation_size))
        )

        # Init Height Encoders
        self.height_encoder = HeightEncoder(height_encoder_size=height_encoder_size, layer_sizes=actor_height_encoder_layer_sizes)
        self.critic_height_encoder = HeightEncoder(height_encoder_size=2 * height_encoder_size, layer_sizes=critic_height_encoder_layer_sizes)
        self.init_height_encoder_nn_params = self.height_encoder.init(init_keys[6], jnp.zeros((1, height_sample_size)))
        self.init_critic_height_encoder_nn_params = self.critic_height_encoder.init(init_keys[7], jnp.zeros((1, critic_height_sample_size)))

        self.params = Params(
            policy_parameters=self.init_actor_nn_params,
            value_parameters=self.init_critic_nn_params,
            lstm_parameters=self.init_lstm_nn_params,
            cross_entropy_params=self.init_cross_entropy_head_nn_params,
            height_encoder_parameters=self.init_height_encoder_nn_params,
            critic_height_encoder_parameters=self.init_critic_height_encoder_nn_params,
            linear_velocity_parameters=self.init_linear_velocity_head_nn_params,
        )
        self.labels = Params(
            policy_parameters="actor",
            value_parameters="critic",
            lstm_parameters="actor",
            cross_entropy_params="head",
            height_encoder_parameters="actor",
            critic_height_encoder_parameters="critic",
            linear_velocity_parameters="head",
        )

    def create_optimizer_and_training_state(self, train_cfg: dict):
        """Create the optimizer and the initial TrainingState."""
        learning_rate = train_cfg["learning_rate"]
        critic_learning_rate = train_cfg["critic_learning_rate"]
        clip_norm = train_cfg["optimizer_clip_norm"]
        eps = train_cfg["optimizer_eps"]
        weight_decay = train_cfg["optimizer_weight_decay"]
        nr_envs = train_cfg["nr_envs"]

        actor_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate, eps=eps))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=critic_learning_rate, eps=eps, weight_decay=weight_decay))
        head_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=train_cfg.get("head_learning_rate"), eps=eps, weight_decay=weight_decay))

        optimizer = optax.multi_transform(
            transforms={'actor': actor_optimizer, 'critic': critic_optimizer, "head": head_optimizer},
            param_labels=self.labels,
        )

        initial_lstm_state = (jnp.zeros((nr_envs, self.lstm_hidden_size)), jnp.zeros((nr_envs, self.lstm_hidden_size)))

        training_state = TrainingState(
            optimizer_state=optimizer.init(self.params),
            params=self.params,
            training_steps=jnp.zeros((), dtype=jnp.float32),
            last_lstm_state=initial_lstm_state,
        )
        return optimizer, training_state

    def create_dictionaries(self, eval_cfg: dict):
        """Return initialised visualizer and eval dicts (carry = (c, h))."""
        nr_eval_envs = eval_cfg.get("nr_eval_envs", 1)
        empty_state_single = (jnp.zeros((1, self.lstm_hidden_size)), jnp.zeros((1, self.lstm_hidden_size)))
        empty_state_batch = (jnp.zeros((nr_eval_envs, self.lstm_hidden_size)), jnp.zeros((nr_eval_envs, self.lstm_hidden_size)))

        visualizer_dict = {
            "action":     jnp.zeros((self.action_size,)),
            "lstm_state": empty_state_single,
        }
        eval_dict = {
            "lstm_state": empty_state_batch,
        }
        return visualizer_dict, eval_dict

    def update_lstm(self, parameters: Params, lstm_state, observation: jnp.ndarray,
                    height_encoded: jnp.ndarray, pred_linear_velocity: jnp.ndarray, true_pos: jnp.ndarray):
        """Advance the SRU by exactly one step (name kept for trainer compatibility).

        Returns
        -------
        new_lstm_state : tuple (c, h)
        lstm_encoding  : (-1, lstm_memory_output_features)
        """
        lstm_input = jnp.concatenate(
            [
                observation.reshape((-1, self.actor_observation_size)),
                height_encoded.reshape((-1, self.height_encoder_size)),
                pred_linear_velocity.reshape((-1, 3)),
                true_pos.reshape((-1, 3)),
            ],
            axis=-1,
        )
        new_lstm_state, lstm_encoding = self.lstm.apply(parameters.lstm_parameters, lstm_state, lstm_input)
        lstm_encoding = lstm_encoding.reshape((-1, self.lstm_memory_output_features))
        return new_lstm_state, lstm_encoding

    # ------------------------------------------------------------------ #
    # Acting / evaluation                                                 #
    # ------------------------------------------------------------------ #
    def sample_action(self, parameters: Params, observation: jnp.ndarray, height_observation: jnp.ndarray,
                      lstm_state, true_pos, key: jnp.ndarray):
        height_encoded = self.height_encoder.apply(parameters.height_encoder_parameters, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        # Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(parameters.linear_velocity_parameters, observation.reshape((-1, self.actor_observation_size)))

        # SRU recurrent step
        new_lstm_state, lstm_encoding = self.update_lstm(parameters, lstm_state, observation, height_encoded, pred_linear_vel, true_pos)

        loc, log_scale = self.actor.apply(
            parameters.policy_parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )
        log_scale = jnp.clip(log_scale, jnp.log(self.min_scale), jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        action = jax.random.normal(key, shape=loc.shape) * scale + loc
        log_prob = self.log_prob(action, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)

        # Cross Entropy Head
        predicted_distribution = self.cross_entropy_head.apply(
            parameters.cross_entropy_params,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), lstm_encoding, jax.lax.stop_gradient(height_encoded)], axis=-1),
        )
        return action, scale, log_prob, height_encoded, predicted_distribution, pred_linear_vel, new_lstm_state

    def get_deterministic_action(self, parameters: Params, observation: jnp.ndarray, height_observation: jnp.ndarray, lstm_state, true_pos: jnp.ndarray):
        height_encoded = self.height_encoder.apply(parameters.height_encoder_parameters, height_observation.reshape(-1, self.height_sample_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        # Linear velocity step
        pred_linear_vel = self.linear_velocity_head.apply(parameters.linear_velocity_parameters, observation.reshape((-1, self.actor_observation_size)))

        # SRU recurrent step
        new_lstm_state, lstm_encoding = self.update_lstm(parameters, lstm_state, observation, height_encoded, pred_linear_vel, true_pos)

        loc, scale = self.actor.apply(
            parameters.policy_parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )

        # Cross Entropy Head
        predicted_distribution = self.cross_entropy_head.apply(
            parameters.cross_entropy_params,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), lstm_encoding, jax.lax.stop_gradient(height_encoded)], axis=-1),
        )

        return loc, predicted_distribution, height_encoded, pred_linear_vel, new_lstm_state

    def evaluate_actions(self, parameters: Params, observation: jnp.ndarray, height_observation: jnp.ndarray):
        value_nn_params = parameters.value_parameters
        height_encoder_params = parameters.critic_height_encoder_parameters
        height_encoded = self.critic_height_encoder.apply(height_encoder_params, height_observation.reshape(-1, self.critic_height_sample_size))
        height_encoded = height_encoded.reshape((-1, int(self.critic_height_encoder.height_encoder_size)))
        value = self.critic.apply(value_nn_params, jnp.concatenate([observation.reshape((-1, self.value_observation_size)), height_encoded], axis=-1))
        return value

    def log_prob_from_observation(self, actions, parameters: Params, observation: jnp.ndarray,
                                  height_observation: jnp.ndarray, lstm_encoding: jnp.ndarray):
        """Apply the heads given an ALREADY-UNROLLED ``lstm_encoding``.

        The recurrent core is unrolled separately (by scanning ``update_lstm``)
        and its output is passed in here, so the head losses never re-run the SRU.
        """
        actions = actions.reshape((-1, self.action_size))
        lstm_encoding = lstm_encoding.reshape((-1, self.lstm_memory_output_features))

        height_encoded = self.height_encoder.apply(parameters.height_encoder_parameters, height_observation.reshape((-1, self.height_sample_size)))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        loc, log_scale = self.actor.apply(
            parameters.policy_parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )
        log_scale = jnp.clip(log_scale, jnp.log(self.min_scale), jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        log_prob = self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)

        # Cross Entropy Head
        predicted_distribution = self.cross_entropy_head.apply(
            parameters.cross_entropy_params,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), lstm_encoding, jax.lax.stop_gradient(height_encoded)], axis=-1),
        )

        return log_prob, loc, scale, predicted_distribution