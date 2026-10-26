"""Shared base of the history-based approaches (Transformer, FixedWindow).

The memory is a window of past inputs. Each history entry is
[actor observation, height encoding, predicted linear velocity (3), noisy x, y, yaw (3)].
"""

import flax
import jax.numpy as jnp
import optax

from rememberwalk.actor_critic.BaseActorCritic import BaseActorCritic


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""

    optimizer_state: optax.OptState
    params: flax.struct.PyTreeNode
    training_steps: jnp.ndarray


@flax.struct.dataclass
class Transition:
    """One rollout step, as stored for the PPO update (leading axes: unroll step, env)."""

    done: jnp.ndarray
    truncation: jnp.ndarray
    actor_observation: jnp.ndarray
    actor_height_observation: jnp.ndarray
    critic_observation: jnp.ndarray
    critic_height_observation: jnp.ndarray
    delta_position: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    critic_next_observation: jnp.ndarray
    critic_next_height_observation: jnp.ndarray
    log_prob: jnp.ndarray
    true_distribution: jnp.ndarray
    linear_velocity: jnp.ndarray
    noisy_true_pos: jnp.ndarray


class HistoryActorCritic(BaseActorCritic):
    """Actor-critic whose memory is the window of the last ``sequence_length`` step inputs.

    Subclasses build the networks, set ``sequence_length``, ``history_input_size`` and
    ``memory_output_features`` and implement ``encode_history``.
    """

    def encode_history(self, parameters, observation_history: jnp.ndarray) -> jnp.ndarray:
        """(batch, sequence_length, history_input_size) -> (batch, memory_output_features)"""
        raise NotImplementedError

    def create_optimizer_and_training_state(self, train_cfg: dict):
        """Returns (optimizer, TrainingState); the history lives in the env state, not here."""
        optimizer = self.create_optimizer(train_cfg)
        training_state = TrainingState(
            optimizer_state=optimizer.init(self.params),
            params=self.params,
            training_steps=jnp.zeros((), dtype=jnp.float32),
        )
        return optimizer, training_state

    def create_dictionaries(self, eval_cfg: dict):
        """Return initialised visualizer and eval dicts.

        Both dicts include ``observation_history`` so callbacks can maintain
        the context window across steps without depending on a
        particular state type.
        """
        nr_eval_envs = eval_cfg.get("nr_eval_envs", 1)
        empty_history_single = jnp.zeros((self.sequence_length, self.history_input_size))
        empty_history_batch = jnp.zeros((nr_eval_envs, self.sequence_length, self.history_input_size))

        visualizer_dict = {
            "action": jnp.zeros((self.action_size,)),
            "observation_history": empty_history_single,
        }
        eval_dict = {
            "observation_history": empty_history_batch,
        }
        return visualizer_dict, eval_dict

    def _encode(self, parameters, observation_history: jnp.ndarray):
        history_encoding = self.encode_history(
            parameters, observation_history.reshape(-1, self.sequence_length, self.history_input_size)
        )
        return history_encoding.reshape((-1, self.memory_output_features))

    def sample_action(
        self,
        parameters,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        observation_history: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """Samples an action for a batch of envs.

        Args:
            observation: (batch, actor_observation_size).
            height_observation: (batch, height_sample_size).
            observation_history: (batch, sequence_length, history_input_size).
            key: PRNG key of the action noise.

        Returns:
            (action, scale, log_prob (batch, 1), height_encoded, predicted map logits,
            predicted linear velocity).
            The caller appends the new history entry.
        """
        height_encoded = self.encode_height(parameters, height_observation)
        pred_linear_vel = self.predict_linear_velocity(parameters, observation)
        history_encoding = self._encode(parameters, observation_history)

        loc, scale = self.actor_distribution(
            parameters,
            jnp.concatenate(
                [observation.reshape((-1, self.actor_observation_size)), height_encoded, history_encoding],
                axis=-1,
            ),
        )
        action, log_prob = self.sample_from_distribution(loc, scale, key)

        predicted_distribution = self.predict_distribution(
            parameters, observation, history_encoding, height_encoded
        )
        return action, scale, log_prob, height_encoded, predicted_distribution, pred_linear_vel

    def get_deterministic_action(
        self,
        parameters,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        observation_history: jnp.ndarray,
    ):
        """Mean action: (loc, predicted map logits, height_encoded, predicted linear velocity)."""
        height_encoded = self.encode_height(parameters, height_observation)
        pred_linear_vel = self.predict_linear_velocity(parameters, observation)
        history_encoding = self._encode(parameters, observation_history)

        loc, scale = self.actor.apply(
            parameters.policy_parameters,
            jnp.concatenate(
                [observation.reshape((-1, self.actor_observation_size)), height_encoded, history_encoding],
                axis=-1,
            ),
        )

        predicted_distribution = self.predict_distribution(
            parameters, observation, history_encoding, height_encoded
        )
        return loc, predicted_distribution, height_encoded, pred_linear_vel

    def log_prob_from_observation(
        self,
        actions,
        parameters,
        observation: jnp.ndarray,
        height_observation: jnp.ndarray,
        observation_history: jnp.ndarray,
    ):
        """Re-evaluates stored actions under the current parameters (PPO update).

        Returns:
            (log_prob (batch, 1), loc, scale, predicted map logits (batch, distribution_size)).
        """
        actions = actions.reshape((-1, self.action_size))
        height_encoded = self.encode_height(parameters, height_observation)
        history_encoding = self._encode(parameters, observation_history)

        loc, scale = self.actor_distribution(
            parameters,
            jnp.concatenate(
                [observation.reshape((-1, self.actor_observation_size)), height_encoded, history_encoding],
                axis=-1,
            ),
        )
        log_prob = self.log_prob(actions, loc, scale)
        log_prob = jnp.expand_dims(log_prob, axis=-1)

        predicted_distribution = self.predict_distribution(
            parameters, observation, history_encoding, height_encoded
        )
        return log_prob, loc, scale, predicted_distribution
