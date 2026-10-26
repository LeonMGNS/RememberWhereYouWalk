import jax
import jax.numpy as jnp
import optax
from rememberwalk.actor_critic.MLP import Actor, Critic, HeightEncoder
from rememberwalk.actor_critic.heads import CrossEntropyHead, LinearVelocityHead


class BaseActorCritic:
    """Gaussian policy math plus the network glue shared by every approach.

    Subclasses set ``actor``, ``critic``, ``height_encoder``, ``critic_height_encoder``,
    ``cross_entropy_head``, ``linear_velocity_head``, ``params``, ``labels`` and the size
    attributes used below.
    """

    def log_prob(self, actions, loc, scale):
        log_unnormalized = -0.5 * jnp.square((actions - loc) / scale)
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        return jnp.sum(log_unnormalized - log_normalization, axis=-1)

    def entropy(self, loc, scale):
        log_normalization = 0.5 * jnp.log(2.0 * jnp.pi) + jnp.log(scale)
        entropy = 0.5 + log_normalization
        entropy = jnp.sum(entropy, axis=-1)
        return entropy

    def init_shared_networks(self, keys: dict, actor_memory_size: int, head_memory_size: int,
                             actor_layer_sizes, critic_layer_sizes,
                             actor_height_encoder_layer_sizes, critic_height_encoder_layer_sizes):
        """Builds and initialises the actor, critic, both height encoders and the two auxiliary heads.

        ``keys`` maps "actor", "critic", "cross_entropy", "linear_velocity", "height_encoder" and
        "critic_height_encoder" to the PRNG key of each network (the indices differ per approach).
        """
        self.actor = Actor(actor_output_size=self.action_size, layer_sizes=actor_layer_sizes)
        self.init_actor_nn_params = self.actor.init(keys["actor"], jnp.zeros((1, self.actor_observation_size + self.height_encoder_size + actor_memory_size)))

        self.critic = Critic(layer_sizes=critic_layer_sizes)
        self.init_critic_nn_params = self.critic.init(keys["critic"], jnp.zeros((1, self.value_observation_size + 2 * self.height_encoder_size)))

        self.cross_entropy_head = CrossEntropyHead(int(self.distribution_size))
        self.init_cross_entropy_head_nn_params = self.cross_entropy_head.init(keys["cross_entropy"], jnp.zeros((1, self.actor_observation_size + self.height_encoder_size + head_memory_size)))

        self.linear_velocity_head = LinearVelocityHead()
        self.init_linear_velocity_head_nn_params = self.linear_velocity_head.init(keys["linear_velocity"], jnp.zeros((1, self.actor_observation_size)))

        self.height_encoder = HeightEncoder(height_encoder_size=self.height_encoder_size, layer_sizes=actor_height_encoder_layer_sizes)
        self.critic_height_encoder = HeightEncoder(height_encoder_size=2 * self.height_encoder_size, layer_sizes=critic_height_encoder_layer_sizes)
        self.init_height_encoder_nn_params = self.height_encoder.init(keys["height_encoder"], jnp.zeros((1, self.height_sample_size)))
        self.init_critic_height_encoder_nn_params = self.critic_height_encoder.init(keys["critic_height_encoder"], jnp.zeros((1, self.critic_height_sample_size)))

    def create_optimizer(self, train_cfg: dict):
        """Adam for the actor, AdamW for the critic and the auxiliary heads, selected by ``self.labels``."""
        learning_rate = train_cfg["learning_rate"]
        critic_learning_rate = train_cfg["critic_learning_rate"]
        clip_norm = train_cfg["optimizer_clip_norm"]
        eps = train_cfg["optimizer_eps"]
        weight_decay = train_cfg["optimizer_weight_decay"]

        actor_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=learning_rate, eps=eps))
        critic_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=critic_learning_rate, eps=eps, weight_decay=weight_decay))
        head_optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adamw(learning_rate=train_cfg.get("head_learning_rate"), eps=eps, weight_decay=weight_decay))

        return optax.multi_transform(
            transforms={'actor': actor_optimizer, 'critic': critic_optimizer, "head": head_optimizer},
            param_labels=self.labels,
        )

    def encode_height(self, parameters, height_observation: jnp.ndarray):
        height_encoded = self.height_encoder.apply(parameters.height_encoder_parameters, height_observation.reshape(-1, self.height_sample_size))
        return height_encoded.reshape((-1, self.height_encoder.height_encoder_size))

    def predict_linear_velocity(self, parameters, observation: jnp.ndarray):
        return self.linear_velocity_head.apply(parameters.linear_velocity_parameters, observation.reshape((-1, self.actor_observation_size)))

    def predict_distribution(self, parameters, observation: jnp.ndarray, memory_encoding: jnp.ndarray, height_encoded: jnp.ndarray):
        """Cross entropy head; the height encoding is detached so the head does not train the height encoder."""
        return self.cross_entropy_head.apply(
            parameters.cross_entropy_params,
            jnp.concatenate([observation.reshape((-1, self.actor_observation_size)), memory_encoding, jax.lax.stop_gradient(height_encoded)], axis=-1),
        )

    def actor_distribution(self, parameters, actor_input: jnp.ndarray):
        """Returns (loc, scale) of the Gaussian policy with the scale clipped to [min_scale, max_scale]."""
        loc, log_scale = self.actor.apply(parameters.policy_parameters, actor_input)
        log_scale = jnp.clip(log_scale, jnp.log(self.min_scale), jnp.log(self.max_scale))
        scale = jnp.exp(log_scale)
        return loc, scale

    def sample_from_distribution(self, loc: jnp.ndarray, scale: jnp.ndarray, key: jnp.ndarray):
        action = jax.random.normal(key, shape=loc.shape) * scale + loc
        log_prob = self.log_prob(action, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)
        return action, log_prob

    def evaluate_actions(self, parameters, observation: jnp.ndarray, height_observation: jnp.ndarray):
        value_nn_params = parameters.value_parameters
        height_encoder_params = parameters.critic_height_encoder_parameters
        height_encoded = self.critic_height_encoder.apply(height_encoder_params, height_observation.reshape(-1, self.critic_height_sample_size))
        height_encoded = height_encoded.reshape((-1, int(self.critic_height_encoder.height_encoder_size)))
        value = self.critic.apply(value_nn_params, jnp.concatenate([observation.reshape((-1, self.value_observation_size)), height_encoded], axis=-1))
        return value
