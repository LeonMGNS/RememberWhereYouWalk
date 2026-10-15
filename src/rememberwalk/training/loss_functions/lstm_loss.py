import jax.numpy as jnp
from rememberwalk.actor_critic.LSTM import Params, Transition
from rememberwalk.training.ppo import PPOTrainer
import jax
from typing import Tuple
import optax


def loss_fn(ppo_trainer: PPOTrainer, params: Params, transitions: Transition, training_steps: jnp.ndarray, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
    
    #Unpack Transition
    rewards = jnp.expand_dims(transitions.reward, axis=-1)
    dones = jnp.expand_dims(transitions.done, axis=-1)
    truncations = jnp.expand_dims(transitions.truncation, axis=-1)
    data_collection_log_prob = transitions.log_prob
    actor_actions = transitions.action
    actor_observation = transitions.actor_observation
    actor_height_observation = transitions.actor_height_observation
    lstm_state = transitions.lstm_state  # tuple (c, h), each (T, B, hidden) after the swapaxes
    critic_observation = transitions.critic_observation
    critic_height_observation = transitions.critic_height_observation
    critic_next_height_observations = transitions.critic_next_height_observation
    critic_next_observation = transitions.critic_next_observation
    true_distributions = transitions.true_distribution
    true_linear_velocities = transitions.linear_velocity
    noisy_true_pos = transitions.noisy_true_pos

    # Encode height observations (needed as part of the LSTM input)
    heights_encoded = jax.vmap(ppo_trainer.actor_critic.height_encoder.apply, (None, 0))(params.height_encoder_parameters, actor_height_observation)

    # Predict linear velocity (fed into the LSTM and supervised below)
    pred_linear_velocities = jax.vmap(ppo_trainer.actor_critic.linear_velocity_head.apply, (None, 0))(params.linear_velocity_parameters, actor_observation)

    # Forward LSTM unroll (this replaces the matrix-memory forward scan)
    def forward_scan(carry, scan_input):
        current_lstm_state = carry
        actor_obs, height_encoded, done, linear_velocity,noisy_true_pos = scan_input
        new_lstm_state, lstm_encoding = ppo_trainer.actor_critic.update_lstm(params, current_lstm_state, actor_obs, height_encoded, linear_velocity,noisy_true_pos)
        # Reset the carry at episode boundaries (zero init), like the memory reset to 0.5
        c, h = new_lstm_state
        keep = 1.0 - done
        new_lstm_state = (c * keep, h * keep)
        return new_lstm_state, lstm_encoding

    init_lstm_state = (lstm_state[0][0], lstm_state[1][0])  # (c[0], h[0]) -> each (B, hidden)
    forward_scan_input = (actor_observation, heights_encoded, dones, jax.lax.stop_gradient(pred_linear_velocities),noisy_true_pos)
    _, lstm_encodings = jax.lax.scan(forward_scan, init_lstm_state, forward_scan_input, unroll=False)

    # Compute Heads (actor / localization / cross-entropy) from the unrolled encodings
    current_log_prob, current_loc, current_scale, predicted_distribution = jax.vmap(
        ppo_trainer.actor_critic.log_prob_from_observation, (0, None, 0, 0, 0)
    )(actor_actions, params, actor_observation, actor_height_observation, lstm_encodings)

    # Interpolate Discount Factor
    discount_factor = ppo_trainer.min_discount_factor
    
    # Compute Critic
    baseline = jax.vmap(ppo_trainer.actor_critic.evaluate_actions, (None, 0, 0))(params, critic_observation, critic_height_observation)
    bootstrap_value = ppo_trainer.actor_critic.evaluate_actions(params, critic_observation[-1], critic_next_height_observations[-1])
    terminated_bootstrap_value = jnp.zeros_like(bootstrap_value)
    bootstrap_value = jnp.where(dones[-1], terminated_bootstrap_value, bootstrap_value)
    next_value = jax.lax.stop_gradient(jnp.concatenate([baseline[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0))

    # Compute GAE
    termination = dones * (1 - truncations)
    truncation_mask = 1 - truncations
    deltas = rewards + discount_factor * next_value * (1 - termination) - baseline
    acc = jnp.zeros_like(deltas[0])
    def compute_gae(carry, target):
        lambda_, acc = carry
        truncation_mask, delta, termination = target
        acc = delta + discount_factor * (1 - termination) * truncation_mask * lambda_ * acc
        return (lambda_, acc), (acc)
    _, accs = jax.lax.scan(compute_gae, (ppo_trainer.gae_lambda, acc), (truncation_mask, deltas, termination), length=int(termination.shape[0]), reverse=True)

    #Compute advantages
    advantages = jax.lax.stop_gradient(accs)
    vs = jax.lax.stop_gradient(accs + baseline)  # Use this for value loss

    # Advantage normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_prob_diff = current_log_prob - data_collection_log_prob
    rho_s = jnp.exp(log_prob_diff)

    # Compute Policy Surrogate Loss
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_trainer.clipping_epsilon, 1 + ppo_trainer.clipping_epsilon) * advantages
    policy_loss = ppo_trainer.training_config["policy_loss_scaling_factor"] * -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = ppo_trainer.training_config["value_loss_scaling_factor"] * jnp.mean(v_error * v_error) * 0.5

    # Entropy reward
    entropy_cost  = interpolate_entropy_cost(ppo_trainer.min_entropy_cost, ppo_trainer.max_entropy_cost, training_steps,float(ppo_trainer.training_steps))
    entropy_vals = ppo_trainer.actor_critic.entropy(current_loc, current_scale)
    entropy = jnp.mean(entropy_vals)
    entropy_loss = entropy_cost * -entropy

    # BCE Loss
    bce_error = jnp.mean(compute_bce_loss(predicted_distribution, true_distributions))
    bce_loss = ppo_trainer.training_config["bce_loss_scaling_factor"] * jnp.mean(compute_bce_loss(predicted_distribution, true_distributions))

    # Linear Velocity loss
    linear_velocity_error = true_linear_velocities - pred_linear_velocities
    linear_velocity_error_log = jnp.mean(jnp.abs(linear_velocity_error))    
    linear_velocity_error_loss = ppo_trainer.training_config["linear_velocity_loss_scaling_factor"] * jnp.mean(linear_velocity_error * linear_velocity_error) * 0.5


    # KL divergence
    logratio = current_log_prob - data_collection_log_prob
    ratio = jnp.exp(logratio)
    approx_kl_div = jnp.mean((ratio - 1) - logratio)
    total_loss = policy_loss + v_loss + entropy_loss + bce_loss  + linear_velocity_error_loss
    losses = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "entropy": entropy,
        "approx_kl_div": approx_kl_div,
        "bce_loss": bce_loss,
        "bce_error": bce_error,
        "discount_factor": discount_factor,
        "linear_velocity_error_log": linear_velocity_error_log,
    }
    return total_loss, (losses, key)


def compute_bce_loss(predicted_distribution: jnp.ndarray, true_distribution: jnp.ndarray):
    loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution, true_distribution)
    loss = jnp.mean(loss, axis=-1)
    return loss


def interpolate_param(min_val, max_val, curr_coeff):
    new_value = (1 - curr_coeff) * min_val + curr_coeff * max_val  # Interpolates in [min_val, max_val]
    return jnp.clip(new_value, min_val, max_val)


def interpolate_entropy_cost(min_entropy_cost, max_entropy_cost, training_steps, max_training_steps=6_000_000_000.0):
    interpolation_parameter = jnp.clip(training_steps / max_training_steps, 0, 1)
    entropy_cost = max_entropy_cost - interpolation_parameter * (max_entropy_cost - min_entropy_cost)
    return entropy_cost