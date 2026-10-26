import jax
import jax.numpy as jnp
import flax.linen as nn
from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic
import optax
import flax


@flax.struct.dataclass
class Params:
    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    lstm_parameters: jnp.ndarray
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


class LSTM(nn.Module):
    """Standard LSTM core.

    The pre-processing (two ELU-activated dense layers) and the single output
    projection mirror the Transformer ``Encoder``, so downstream heads receive a feature vector
    of the same size (``memory_output_features``). One ``__call__`` advances the
    recurrent state by exactly one step, which makes it trivial to unroll the
    core on its own via ``jax.lax.scan`` in the loss function.
    """
    hidden_size: int
    memory_output_features: int

    @nn.compact
    def __call__(self, carry, x):
        # Input projection (analogous to the Encoder's pre-encoder dense stack).
        y = nn.Dense(256)(x)
        y = nn.elu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.elu(y)

        # Single LSTM step.
        new_carry, h = nn.OptimizedLSTMCell(features=self.hidden_size)(carry, y)

        # Output projection to the feature size consumed by the heads.
        memory_encoding = nn.Dense(self.memory_output_features)(h)
        memory_encoding = nn.elu(memory_encoding)
        return new_carry, memory_encoding


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
        lstm_hidden_size, lstm_memory_output_features = self.core_sizes(config)
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
        # Per-timestep LSTM input: [observation, height_encoded, pred_linear_velocity(3)]
        # (same composition as the transformer's per-timestep input).
        self.lstm_input_size = actor_observation_size + height_encoder_size + 3 + 3

        # Init actor, critic, height encoders and heads
        self.init_shared_networks(
            keys={"actor": init_keys[0], "critic": init_keys[1], "cross_entropy": init_keys[3], "linear_velocity": init_keys[4],
                  "height_encoder": init_keys[6], "critic_height_encoder": init_keys[7]},
            actor_memory_size=lstm_memory_output_features,
            head_memory_size=self.lstm_memory_output_features,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_height_encoder_layer_sizes=actor_height_encoder_layer_sizes,
            critic_height_encoder_layer_sizes=critic_height_encoder_layer_sizes,
        )

        # Init the recurrent core
        self.lstm = self.make_core(config)
        dummy_carry = (jnp.zeros((1, lstm_hidden_size)), jnp.zeros((1, lstm_hidden_size)))
        self.init_lstm_nn_params = self.lstm.init(init_keys[2], dummy_carry, jnp.zeros((1, self.lstm_input_size)))

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

    def core_sizes(self, config: dict):
        """(hidden size, output feature size) of the core. Falls back to the transformer keys so existing configs keep working."""
        lstm_hidden_size = config.get("lstm_hidden_size", config.get("transformer_hidden_size", 256))
        lstm_memory_output_features = config.get("lstm_memory_output_features", config.get("transformer_memory_output_features"))
        return lstm_hidden_size, lstm_memory_output_features

    def make_core(self, config: dict) -> nn.Module:
        return LSTM(hidden_size=self.lstm_hidden_size, memory_output_features=self.lstm_memory_output_features)

    def create_optimizer_and_training_state(self, train_cfg: dict):
        nr_envs = train_cfg["nr_envs"]
        optimizer = self.create_optimizer(train_cfg)

        initial_lstm_state = (jnp.zeros((nr_envs, self.lstm_hidden_size)),jnp.zeros((nr_envs, self.lstm_hidden_size)))

        training_state = TrainingState(
            optimizer_state=optimizer.init(self.params),
            params=self.params,
            training_steps=jnp.zeros((), dtype=jnp.float32),
            last_lstm_state=initial_lstm_state,
        )
        return optimizer, training_state

    def create_dictionaries(self, eval_cfg: dict):
        """Return initialised visualizer and eval dicts.

        Both dicts carry an ``lstm_state`` so callbacks can maintain the
        recurrent core across steps without depending on a particular state type.
        """
        nr_eval_envs = eval_cfg.get("nr_eval_envs", 1)
        empty_state_single = (jnp.zeros((1, self.lstm_hidden_size)),jnp.zeros((1, self.lstm_hidden_size)))
        empty_state_batch = (jnp.zeros((nr_eval_envs, self.lstm_hidden_size)),jnp.zeros((nr_eval_envs, self.lstm_hidden_size)))

        visualizer_dict = {
            "action":     jnp.zeros((self.action_size,)),
            "lstm_state": empty_state_single,
        }
        eval_dict = {
            "lstm_state": empty_state_batch,
        }
        return visualizer_dict, eval_dict

    def update_lstm(self, parameters: Params, lstm_state, observation: jnp.ndarray,
                    height_encoded: jnp.ndarray, pred_linear_velocity: jnp.ndarray,true_pos:jnp.ndarray):
        """Advance the LSTM by exactly one step.

        Mirrors ``update_matrix_memory``: it takes the current recurrent state,
        does one update, and returns the new state plus the encoding that the
        heads consume. Because it is a pure single step it can be unrolled with
        ``jax.lax.scan`` in the loss, independently of every head.

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
                true_pos.reshape((-1,3))
            ],
            axis=-1,
        )
        new_lstm_state, lstm_encoding = self.lstm.apply(parameters.lstm_parameters, lstm_state, lstm_input)
        lstm_encoding = lstm_encoding.reshape((-1, self.lstm_memory_output_features))
        return new_lstm_state, lstm_encoding

    def sample_action(self, parameters: Params, observation: jnp.ndarray, height_observation: jnp.ndarray,
                      lstm_state,true_pos, key: jnp.ndarray):
        height_encoded = self.encode_height(parameters, height_observation)
        pred_linear_vel = self.predict_linear_velocity(parameters, observation)
        new_lstm_state, lstm_encoding = self.update_lstm(parameters, lstm_state, observation, height_encoded, pred_linear_vel,true_pos)

        loc, scale = self.actor_distribution(
            parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )
        action, log_prob = self.sample_from_distribution(loc, scale, key)

        predicted_distribution = self.predict_distribution(parameters, observation, lstm_encoding, height_encoded)
        return action, scale, log_prob, height_encoded, predicted_distribution, pred_linear_vel, new_lstm_state

    def get_deterministic_action(self, parameters: Params, observation: jnp.ndarray, height_observation: jnp.ndarray, lstm_state,true_pos:jnp.ndarray):
        height_encoded = self.encode_height(parameters, height_observation)
        pred_linear_vel = self.predict_linear_velocity(parameters, observation)
        new_lstm_state, lstm_encoding = self.update_lstm(parameters, lstm_state, observation, height_encoded, pred_linear_vel,true_pos)

        loc, scale = self.actor.apply(
            parameters.policy_parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )

        predicted_distribution = self.predict_distribution(parameters, observation, lstm_encoding, height_encoded)
        return loc, predicted_distribution, height_encoded, pred_linear_vel, new_lstm_state

    def log_prob_from_observation(self, actions, parameters: Params, observation: jnp.ndarray,
                                  height_observation: jnp.ndarray, lstm_encoding: jnp.ndarray):
        """Apply the heads given an ALREADY-UNROLLED ``lstm_encoding``.

        The recurrent core is unrolled separately (by scanning ``update_lstm``)
        and its output is passed in here, so the head losses never re-run the core.
        """
        actions = actions.reshape((-1, self.action_size))
        lstm_encoding = lstm_encoding.reshape((-1, self.lstm_memory_output_features))
        height_encoded = self.encode_height(parameters, height_observation)

        loc, scale = self.actor_distribution(
            parameters,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), height_encoded, lstm_encoding], axis=-1),
        )
        log_prob = self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)

        predicted_distribution = self.predict_distribution(parameters, observation, lstm_encoding, height_encoded)
        return log_prob, loc, scale, predicted_distribution
