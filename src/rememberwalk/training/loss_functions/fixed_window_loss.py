import jax.numpy as jnp
from rememberwalk.actor_critic.Transformer import Params, Transition
from rememberwalk.training.ppo import PPOTrainer
import jax
from typing import Tuple
import optax

def create_observation_history(encoded_observations: jnp.ndarray, dones: jnp.ndarray, seq_len: int) -> jnp.ndarray:
    """Reconstruct the causal observation history used by the transformer.

    For each timestep ``t`` the transformer received a window of the ``seq_len``
    most-recent encoded observations *before* the action at step ``t`` was taken.
    Episode boundaries (``done`` flags) zero-out the carry so that a new episode
    always starts with an empty history.

    Args:
        encoded_observations: ``(nr_envs, T, enc_size)`` – concatenation of
            actor observations and encoded height maps, ordered by env then time.
        dones:               ``(nr_envs, T)`` – ``True`` when the episode ended
            at the corresponding step (used to reset the history carry).
        seq_len:             Transformer sequence length.

    Returns:
        ``(nr_envs, T, seq_len, enc_size)`` – the observation history that was
        available *before* each action, matching what was used during data
        collection.
    """
    nr_envs, T, enc_size = encoded_observations.shape
    # Transpose so lax.scan iterates over timesteps (T-first).
    obs_T  = encoded_observations.swapaxes(0, 1)   # (T, nr_envs, enc_size)
    done_T = dones.swapaxes(0, 1)                  # (T, nr_envs)

    def step(carry, inputs):
        history = carry                             # (nr_envs, seq_len, enc_size)
        obs, done = inputs                          # (nr_envs, enc_size), (nr_envs,)
        # The history *before* adding the current obs is what the transformer saw.
        output = history
        # Roll and insert the current obs at the last position.
        new_history = jnp.roll(history, shift=-1, axis=1)
        new_history = new_history.at[:, -1, :].set(obs)
        # Zero-out the carry after a done so the next episode starts fresh.
        next_carry = jnp.where(done[:, None, None], jnp.zeros_like(new_history), new_history)
        return next_carry, output

    init = jnp.zeros((nr_envs, seq_len, enc_size))
    _, histories = jax.lax.scan(step, init, (obs_T, done_T))  # histories: (T, nr_envs, seq_len, enc_size)
    return histories.swapaxes(0, 1)                            # (nr_envs, T, seq_len, enc_size)


def loss_fn(ppo_trainer: PPOTrainer, params: Params, transitions: Transition, training_steps: jnp.ndarray, key: jnp.ndarray) -> Tuple:
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)

    rewards       = jnp.expand_dims(transitions.reward, axis=-1)
    dones         = jnp.expand_dims(transitions.done,   axis=-1)
    truncations   = jnp.expand_dims(transitions.truncation, axis=-1)
    data_log_prob = transitions.log_prob
    actor_actions = transitions.action
    actor_obs     = transitions.actor_observation
    actor_h_obs   = transitions.actor_height_observation
    critic_obs    = transitions.critic_observation
    critic_h_obs  = transitions.critic_height_observation
    critic_next_h = transitions.critic_next_height_observation
    critic_next   = transitions.critic_next_observation
    true_lin_vel  = transitions.linear_velocity
    true_dist     = transitions.true_distribution
    true_pos      = transitions.delta_position
    noisy_true_pos= transitions.noisy_true_pos

    seq_len = ppo_trainer.actor_critic.window_sequence_length

    # Encode heights with the *current* parameters so gradients flow back through
    # the height encoder.  The result shape is (nr_envs, T, encoded_height_size).
    height_encoded = ppo_trainer.actor_critic.height_encoder.apply(
        params.height_encoder_parameters, actor_h_obs
    )
    pred_linear_vel = ppo_trainer.actor_critic.linear_velocity_head.apply(params.linear_velocity_parameters,actor_obs)

    # Build the [actor_obs || encoded_height] sequence and reconstruct causal histories.
    encoded_seq = jnp.concatenate([actor_obs, height_encoded,jax.lax.stop_gradient(pred_linear_vel),noisy_true_pos], axis=-1)  # (nr_envs, T, input_size)
    obs_histories = create_observation_history(encoded_seq, transitions.done, seq_len)
    # obs_histories: (nr_envs, T, seq_len, input_size)

    # Recompute log-probabilities with the current policy parameters.
    current_log_prob, current_loc, current_scale,predicted_distribution = jax.vmap(
        ppo_trainer.actor_critic.log_prob_from_observation, (0, None, 0, 0,0)
    )(actor_actions, params, actor_obs, actor_h_obs, obs_histories)

    baseline         = jax.vmap(ppo_trainer.actor_critic.evaluate_actions, (None, 0, 0))(params, critic_obs, critic_h_obs)
    bootstrap_value  = ppo_trainer.actor_critic.evaluate_actions(params, critic_next[-1], critic_next_h[-1])

    terminated_bv    = jnp.zeros_like(bootstrap_value)
    bootstrap_value  = jnp.where(dones[-1], terminated_bv, bootstrap_value)
    next_value       = jnp.concatenate([baseline[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)

    termination      = dones * (1 - truncations)
    trunc_mask       = 1 - truncations
    gamma            = ppo_trainer.discount_factor
    deltas           = rewards + gamma * next_value * (1 - termination) - baseline
    deltas          *= trunc_mask
    acc              = jnp.zeros_like(deltas[0])

    def compute_gae(carry, target):
        lambda_, acc = carry
        tm, delta, term = target
        acc = delta + gamma * (1 - term) * tm * lambda_ * acc
        return (lambda_, acc), acc

    _, accs = jax.lax.scan(
        compute_gae,
        (ppo_trainer.gae_lambda, acc),
        (trunc_mask, deltas, termination),
        length=int(termination.shape[0]),
        reverse=True,
    )
    vs          = jax.lax.stop_gradient(accs + baseline)
    advantages  = jax.lax.stop_gradient(accs)
    advantages  = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    log_diff    = jnp.clip(current_log_prob - data_log_prob, -20.0, 20.0)
    rho         = jnp.exp(log_diff)
    surrogate1  = rho * advantages
    surrogate2  = jnp.clip(rho, 1 - ppo_trainer.clipping_epsilon, 1 + ppo_trainer.clipping_epsilon) * advantages
    policy_loss = ppo_trainer.training_config["policy_loss_scaling_factor"] * -jnp.mean(jnp.minimum(surrogate1, surrogate2))

    v_loss      = ppo_trainer.training_config["value_loss_scaling_factor"] * jnp.mean((vs - baseline) ** 2) * 0.5

    entropy_cost  = interpolate_entropy_cost(ppo_trainer.min_entropy_cost, ppo_trainer.max_entropy_cost, training_steps,float(ppo_trainer.training_steps))
    entropy       = jnp.mean(ppo_trainer.actor_critic.entropy(current_loc, current_scale))
    entropy_loss  = entropy_cost * -entropy

    # Head losses
    # BCE Loss
    bce_error = jnp.mean(compute_bce_loss(predicted_distribution,true_dist))
    bce_loss = ppo_trainer.training_config["bce_loss_scaling_factor"] * jnp.mean(compute_bce_loss(predicted_distribution,true_dist))

    #Linear Velocity loss 
    linear_velocity_error = true_lin_vel - pred_linear_vel
    linear_velocity_error_log = jnp.mean(jnp.abs(linear_velocity_error))
    linear_velocity_error_loss = ppo_trainer.training_config["linear_velocity_loss_scaling_factor"] * jnp.mean(linear_velocity_error*linear_velocity_error)*0.5
    

    approx_kl   = jnp.mean((rho - 1) - log_diff)
    total_loss  = policy_loss + v_loss + entropy_loss  +  bce_loss + linear_velocity_error_loss

    losses = {
        "total_loss":    total_loss,
        "policy_loss":   policy_loss,
        "v_loss":        v_loss,
        "entropy_loss":  entropy_loss,
        "approx_kl_div": approx_kl,
        "bce_loss": bce_loss,
        "bce_error": bce_error,
        "linear_velocity_error_log": linear_velocity_error_log,
    }
    return total_loss, (losses, key)


def interpolate_entropy_cost(min_entropy_cost,max_entropy_cost,training_steps,max_training_steps=6_000_000_000.0):
    interpolation_parameter = jnp.clip(training_steps/max_training_steps,0,1)
    entropy_cost = max_entropy_cost - interpolation_parameter*(max_entropy_cost-min_entropy_cost)
    return entropy_cost

def compute_bce_loss(predicted_distribution: jnp.ndarray,true_distribution:jnp.ndarray):
    loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,true_distribution)
    loss = jnp.mean(loss,axis=-1)
    return loss
