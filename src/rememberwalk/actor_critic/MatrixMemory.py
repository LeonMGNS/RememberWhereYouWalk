"""LSMM: actor-critic with a learned spatial matrix memory (``actor_critic.MatrixMemory.ActorCritic``).

The memory is an (H, W, C) grid (``matrix_memory_shape``) over the maze, anchored in the world frame. Every
step writes a (h, w, c) window (``matrix_memory_update_window_shape``) around the robot's noisy position:

    M_local <- (M_local * F(x) + G(x)) / (max(M_local * F(x) + G(x)) + 0.02)

F (calibration) and G (update) are MLPs on x = [height encoding, proprioception, window, window index,
yaw, predicted linear velocity]. A CNN encodes the whole memory into a feature vector for the actor and the
map-reconstruction (cross entropy) head.
"""

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.linen.initializers import constant, orthogonal

from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic


@flax.struct.dataclass
class Params:
    """Network parameters. Field names are part of the checkpoint format: do not rename."""

    policy_parameters: jnp.ndarray
    value_parameters: jnp.ndarray
    cross_entropy_params: jnp.ndarray
    height_encoder_parameters: jnp.ndarray
    critic_height_encoder_parameters: jnp.ndarray
    f_parameters: jnp.ndarray
    g_parameters: jnp.ndarray
    matrix_memory_encoder_parameters: jnp.ndarray
    linear_velocity_parameters: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: Params
    training_steps: jnp.ndarray
    last_matrix_memory: jnp.ndarray


@flax.struct.dataclass
class Transition:
    """One rollout step, as stored for the PPO update (leading axes: unroll step, env)."""

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
    matrix_memory: jnp.ndarray
    noisy_true_pos: jnp.ndarray
    linear_velocity: jnp.ndarray


class WindowUpdateMLP(nn.Module):
    """Maps the local memory window and its context to a (window_size,) gate in [0, 1]. Used for F and G."""

    window_size: int

    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        def dense_layer(units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)):
            return nn.Dense(units, kernel_init=kernel_init, bias_init=bias_init)

        y = dense_layer(128)(x)
        y = nn.elu(y)
        y = dense_layer(128)(y)
        y = nn.elu(y)
        y = dense_layer(self.window_size)(y)
        y = nn.sigmoid(y)
        return y


class MatrixMemoryEncoder(nn.Module):
    """CNN + MLP: (batch, H, W, C) matrix memory -> (batch, output_feature_size) feature vector."""

    output_feature_size: int

    @nn.compact
    def __call__(self, matrix_memory):
        # Reusable layer setups for consistency
        ortho_init = orthogonal(jnp.sqrt(2))
        zero_bias = constant(0.0)

        def conv_layer(features, kernel_size, strides):
            return nn.Conv(
                features=features,
                kernel_size=kernel_size,
                strides=strides,
                kernel_init=ortho_init,  # Consistent initialization
                bias_init=zero_bias,
            )

        def dense_layer(units):
            return nn.Dense(features=units, kernel_init=ortho_init, bias_init=zero_bias)

        # --- Feature Extraction (Convolutions) ---
        y = conv_layer(8, (5, 5), (4, 4))(matrix_memory)
        y = nn.elu(y)
        y = conv_layer(16, (3, 3), (2, 2))(y)
        y = nn.elu(y)

        # --- Flattening ---
        # Safely flatten the last 3 dimensions (H, W, C) regardless of batch/time dims
        y = y.reshape((y.shape[0], -1))
        # --- MLP Projection ---
        y = nn.LayerNorm()(y)

        y = dense_layer(256)(y)
        y = nn.elu(y)

        y = dense_layer(128)(y)
        y = nn.elu(y)

        y = dense_layer(self.output_feature_size)(y)

        return y


class ActorCritic(BaseActorCritic):
    """LSMM actor-critic.

    Args:
        actor_critic_dict: sizes from ``generate_setup`` (observation sizes, ``matrix_memory_shape``,
            ``matrix_memory_update_window_shape``, ``height_field_in_meters_shape``, ...).
        seed: PRNG seed of the parameter initialisation. The key index of every network is fixed
            (actor 0, critic 1, height encoders 7/8, heads 9/10, F/G 12/13, memory encoder 15).
        *_layer_sizes: hidden sizes of the actor, critic and the two height encoders.
    """

    def __init__(
        self,
        actor_critic_dict: dict,
        seed=0,
        actor_layer_sizes=(512, 256, 128),
        critic_layer_sizes=(512, 256, 128),
        actor_height_encoder_layer_sizes=(256, 128),
        critic_height_encoder_layer_sizes=(512, 256),
    ):

        # Init BaseActorCritic
        super().__init__()
        # Set object variables
        init_key = jax.random.PRNGKey(seed)
        init_keys = jax.random.split(init_key, 16)
        self.actor_observation_size = actor_critic_dict["actor_observation_size"]
        self.value_observation_size = actor_critic_dict["value_observation_size"]
        self.actor_output_size = actor_critic_dict["actor_output_size"]
        self.action_size = int(self.actor_output_size / 2)
        self.distribution_size = actor_critic_dict["distribution_size"]
        self.height_encoder_size = actor_critic_dict["height_encoder_size"]
        self.height_sample_size = actor_critic_dict["height_sample_size"]
        self.critic_height_sample_size = actor_critic_dict["critic_height_sample_size"]
        self.min_scale = actor_critic_dict["min_scale"]
        self.max_scale = actor_critic_dict["max_scale"]
        self.matrix_memory_update_window_shape = actor_critic_dict["matrix_memory_update_window_shape"]
        self.matrix_memory_update_window_size = (
            self.matrix_memory_update_window_shape[0]
            * self.matrix_memory_update_window_shape[1]
            * self.matrix_memory_update_window_shape[2]
        )
        self.matrix_memory_shape = actor_critic_dict["matrix_memory_shape"]
        self.matrix_memory_size = (
            self.matrix_memory_shape[0] * self.matrix_memory_shape[1] * self.matrix_memory_shape[2]
        )
        self.matrix_memory_feature_vector_size = actor_critic_dict["matrix_memory_feature_vector_size"]
        self.height_field_in_meters_shape = actor_critic_dict["height_field_in_meters_shape"]

        # Init actor, critic, height encoders and heads
        self.init_shared_networks(
            keys={
                "actor": init_keys[0],
                "critic": init_keys[1],
                "height_encoder": init_keys[7],
                "critic_height_encoder": init_keys[8],
                "cross_entropy": init_keys[9],
                "linear_velocity": init_keys[10],
            },
            actor_memory_size=self.matrix_memory_feature_vector_size,
            head_memory_size=self.matrix_memory_feature_vector_size,
            actor_layer_sizes=actor_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            actor_height_encoder_layer_sizes=actor_height_encoder_layer_sizes,
            critic_height_encoder_layer_sizes=critic_height_encoder_layer_sizes,
        )

        # Init F (calibration) and G (update) functions
        self.update_input_size = (
            self.actor_observation_size + self.height_encoder_size + self.matrix_memory_update_window_size + 7
        )
        self.init_update_networks(init_keys)

        # Init Matrix Memory Encoder
        self.matrix_memory_encoder = MatrixMemoryEncoder(self.matrix_memory_feature_vector_size)
        self.init_matrix_memory_encoder_nn_params = self.matrix_memory_encoder.init(
            init_keys[15],
            jnp.zeros(
                (1, self.matrix_memory_shape[0], self.matrix_memory_shape[1], self.matrix_memory_shape[2])
            ),
        )
        self.params, self.labels = self.make_params_and_labels()

    def init_update_networks(self, init_keys):
        """Builds F and G (overridden by MatrixMemoryAdditive, which has no F)."""
        self.F = WindowUpdateMLP(self.matrix_memory_update_window_size)
        self.init_F_nn_params = self.F.init(init_keys[12], jnp.zeros((1, self.update_input_size)))
        self.G = WindowUpdateMLP(self.matrix_memory_update_window_size)
        self.init_G_nn_params = self.G.init(init_keys[13], jnp.zeros((1, self.update_input_size)))

    def make_params_and_labels(self):
        """Returns (Params, optimizer labels): "actor", "critic" or "head" per parameter group."""
        params = Params(
            policy_parameters=self.init_actor_nn_params,
            value_parameters=self.init_critic_nn_params,
            cross_entropy_params=self.init_cross_entropy_head_nn_params,
            height_encoder_parameters=self.init_height_encoder_nn_params,
            critic_height_encoder_parameters=self.init_critic_height_encoder_nn_params,
            f_parameters=self.init_F_nn_params,
            g_parameters=self.init_G_nn_params,
            matrix_memory_encoder_parameters=self.init_matrix_memory_encoder_nn_params,
            linear_velocity_parameters=self.init_linear_velocity_head_nn_params,
        )
        labels = Params(
            policy_parameters="actor",
            value_parameters="critic",
            cross_entropy_params="head",
            height_encoder_parameters="actor",
            critic_height_encoder_parameters="critic",
            f_parameters="actor",
            g_parameters="actor",
            matrix_memory_encoder_parameters="actor",
            linear_velocity_parameters="head",
        )
        return params, labels

    def create_optimizer_and_training_state(self, train_cfg: dict):
        """Returns (optimizer, TrainingState) with the memory of all ``nr_envs`` envs initialised to 0.5."""
        memory_shape = self.matrix_memory_shape
        nr_envs = train_cfg["nr_envs"]
        optimizer = self.create_optimizer(train_cfg)
        # Create the initial memory state
        initial_matrix_memory = jnp.ones((nr_envs, memory_shape[0], memory_shape[1], memory_shape[2])) * 0.5

        # Create training state
        training_state = TrainingState(
            optimizer_state=optimizer.init(self.params),
            params=self.params,
            training_steps=jnp.zeros((), dtype=jnp.float32),
            last_matrix_memory=initial_matrix_memory,
        )
        return optimizer, training_state

    def create_dictionaries(self, eval_cfg):
        """Returns (visualizer_dict, eval_dict) with the initial memory of 1 and ``nr_eval_envs`` envs."""

        visualizer_dict = {
            "matrix_memory": jnp.ones(
                (1, self.matrix_memory_shape[0], self.matrix_memory_shape[1], self.matrix_memory_shape[2])
            )
            * 0.5,
            "action": jnp.zeros((self.action_size,)),
        }
        eval_dict = {
            "matrix_memory": jnp.ones(
                (
                    eval_cfg["nr_eval_envs"],
                    self.matrix_memory_shape[0],
                    self.matrix_memory_shape[1],
                    self.matrix_memory_shape[2],
                )
            )
            * 0.5
        }

        eval_dict["predicted_distribution"] = jnp.zeros((eval_cfg["nr_eval_envs"], self.distribution_size))
        eval_dict["pred_delta_loc"] = jnp.zeros((eval_cfg["nr_eval_envs"], 3))
        return visualizer_dict, eval_dict

    def apply_window_update(self, parameters: Params, local_memory: jnp.ndarray, make_update_input):
        """M_local <- M_local * F(x) + G(x)

        ``make_update_input()`` builds x. F and G each build their own copy on purpose: sharing one
        concatenation changes the gradient summation order and breaks bit-exact reproduction.
        """
        update_matrix = self.G.apply(parameters.g_parameters, make_update_input()).reshape(local_memory.shape)
        calibration_matrix = self.F.apply(parameters.f_parameters, make_update_input()).reshape(
            local_memory.shape
        )
        return local_memory * calibration_matrix + update_matrix

    def update_matrix_memory(
        self,
        parameters: Params,
        matrix_memory: jnp.ndarray,
        height_encoded: jnp.ndarray,
        proprio_observation: jnp.ndarray,
        true_noisy_pos: jnp.ndarray,
        pred_linear_velocity: jnp.ndarray,
    ):
        """Writes one env's step into its memory.

        Args:
            matrix_memory: (H, W, C) memory of one env.
            height_encoded: (height_encoder_size,) encoded height map.
            proprio_observation: (actor_observation_size,) actor observation.
            true_noisy_pos: (3,) noisy x, y (m, maze frame) and yaw (rad).
            pred_linear_velocity: (3,) output of the linear velocity head.

        Returns:
            (H, W, C) updated memory; only the window around the position changes.
        """
        H, W, C = self.matrix_memory_shape
        h, w, c = self.matrix_memory_update_window_shape

        # Index creation
        shape_m = jnp.asarray(self.height_field_in_meters_shape)
        memory_shape = jnp.asarray(self.matrix_memory_shape)
        idx_in_matrix_memory = (true_noisy_pos[:2] / shape_m + 0.5) * memory_shape[:2]
        idxs_2d = jnp.floor(idx_in_matrix_memory[:2]).astype(jnp.int32)
        half = jnp.array([h // 2, w // 2, 0], dtype=jnp.int32)
        idxs_3d = jnp.append(idxs_2d, 0) - half
        min_bounds = jnp.array([0, 0, 0], dtype=jnp.int32)
        max_bounds = jnp.array([H - h, W - w, 0], dtype=jnp.int32)
        idxs = jnp.clip(idxs_3d, min_bounds, max_bounds)
        # Get local memory
        local_memory = jax.lax.dynamic_slice(matrix_memory, idxs, (h, w, c))

        # Matrix Memory Update
        window = local_memory

        def make_update_input():
            return jnp.concatenate(
                [
                    height_encoded.reshape((self.height_encoder_size,)),
                    proprio_observation.reshape((self.actor_observation_size,)),
                    window.flatten(),
                    idxs,
                    true_noisy_pos[2].reshape((1,)),
                    pred_linear_velocity,
                ],
                axis=-1,
            )

        local_memory = self.apply_window_update(parameters, local_memory, make_update_input)

        max_local_memory = jnp.max(local_memory) + 0.02

        local_memory = local_memory / max_local_memory

        matrix_memory = jax.lax.dynamic_update_slice(matrix_memory, local_memory, idxs)
        return matrix_memory

    def step_memory(
        self,
        parameters: Params,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        matrix_memory: jnp.ndarray,
        true_noisy_pos: jnp.ndarray,
    ):
        """Writes the current step into the memory of every env and encodes it.

        Returns:
            (height_encoded (batch, height_encoder_size), memory features
            (batch, matrix_memory_feature_vector_size), new memory (batch, H, W, C)).
        """
        height_encoded = self.encode_height(parameters, height_observation)
        pred_linear_vel = self.predict_linear_velocity(parameters, observation)
        new_matrix_memory = jax.vmap(self.update_matrix_memory, (None, 0, 0, 0, 0, 0))(
            parameters,
            matrix_memory,
            height_encoded,
            observation.reshape((-1, self.actor_observation_size)),
            true_noisy_pos,
            pred_linear_vel,
        )
        matrix_memory_feature_vector = self.matrix_memory_encoder.apply(
            parameters.matrix_memory_encoder_parameters, new_matrix_memory
        )
        return height_encoded, matrix_memory_feature_vector, new_matrix_memory

    def sample_action(
        self,
        parameters: Params,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        matrix_memory: jnp.ndarray,
        true_noisy_pos: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """Samples an action for a batch of envs and updates their memory.

        Args:
            observation: (batch, actor_observation_size).
            height_observation: (batch, height_sample_size).
            matrix_memory: (batch, H, W, C).
            true_noisy_pos: (batch, 3).
            key: PRNG key of the action noise.

        Returns:
            (action (batch, action_size), scale (batch, action_size), log_prob (batch, 1),
            predicted map logits (batch, distribution_size), new memory (batch, H, W, C)).
        """
        height_encoded, matrix_memory_feature_vector, new_matrix_memory = self.step_memory(
            parameters, observation, height_observation, matrix_memory, true_noisy_pos
        )

        actor_input = jnp.concatenate(
            [
                observation.reshape((-1, self.actor_observation_size)),
                height_encoded,
                matrix_memory_feature_vector,
            ],
            axis=-1,
        )
        loc, scale = self.actor_distribution(parameters, actor_input)
        action, log_prob = self.sample_from_distribution(loc, scale, key)

        predicted_distribution = self.predict_distribution(
            parameters, observation, matrix_memory_feature_vector, height_encoded
        )
        return action, scale, log_prob, predicted_distribution, new_matrix_memory

    def get_deterministic_action(
        self,
        parameters: Params,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        matrix_memory: jnp.ndarray,
        true_noisy_pos: jnp.ndarray,
    ):
        """Like ``sample_action`` but returns the mean action: (loc, predicted map logits, new memory)."""
        height_encoded, matrix_memory_feature_vector, new_matrix_memory = self.step_memory(
            parameters, observation, height_observation, matrix_memory, true_noisy_pos
        )

        actor_input = jnp.concatenate(
            [
                observation.reshape((-1, self.actor_observation_size)),
                height_encoded,
                matrix_memory_feature_vector,
            ],
            axis=-1,
        )
        loc, log_scale = self.actor.apply(parameters.policy_parameters, actor_input)

        predicted_distribution = self.predict_distribution(
            parameters, observation, matrix_memory_feature_vector, height_encoded
        )
        return loc, predicted_distribution, new_matrix_memory

    def log_prob_from_observation(
        self,
        actions: jnp.ndarray,
        parameters: Params,
        observation: jnp.ndarray,
        height_encoded: jnp.ndarray,
        matrix_memory: jnp.ndarray,
    ):
        """Re-evaluates stored actions under the current parameters (PPO update).

        The memory is not stepped again: ``matrix_memory`` is the already updated memory of each step.

        Returns:
            (log_prob (batch, 1), loc, scale, predicted map logits (batch, distribution_size)).
        """
        actions = actions.reshape((-1, self.action_size))
        height_encoded = height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

        matrix_memory_feature_vector = self.matrix_memory_encoder.apply(
            parameters.matrix_memory_encoder_parameters, matrix_memory
        )

        actor_input = jnp.concatenate(
            [
                observation.reshape((-1, self.actor_observation_size)),
                height_encoded,
                matrix_memory_feature_vector,
            ],
            axis=-1,
        )
        loc, scale = self.actor_distribution(parameters, actor_input)
        log_prob = self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)

        predicted_distribution = self.predict_distribution(
            parameters, observation, matrix_memory_feature_vector, height_encoded
        )
        return log_prob, loc, scale, predicted_distribution
