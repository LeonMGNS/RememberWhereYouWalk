import jax.numpy as jnp
from rememberwalk.actor_critic.MatrixMemory import Params, Transition
from rememberwalk.training.ppo import PPOTrainer
from rememberwalk.training.bce import compute_bce_loss
from rememberwalk.training.loss_functions.common import compute_gae, interpolate_entropy_cost
import jax
from typing import Tuple

def loss_fn(ppo_trainer:PPOTrainer,params: Params, transitions: Transition,training_steps:jnp.ndarray,key:jnp.ndarray)-> Tuple[jnp.ndarray,jnp.ndarray] :
    transitions = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 0, 1), transitions)
    rewards = jnp.expand_dims(transitions.reward, axis=-1)
    dones = jnp.expand_dims(transitions.done, axis=-1)
    truncations = jnp.expand_dims(transitions.truncation, axis=-1)
    data_collection_log_prob = transitions.log_prob
    actor_actions = transitions.action
    actor_observation = transitions.actor_observation
    actor_height_observation = transitions.actor_height_observation
    matrix_memory = transitions.matrix_memory

    critic_observation = transitions.critic_observation
    critic_height_observation = transitions.critic_height_observation
    critic_next_height_observations = transitions.critic_next_height_observation
    critic_next_observation = transitions.critic_next_observation
    true_distributions = transitions.true_distribution
    true_linear_velocities = transitions.linear_velocity
    noisy_true_pos = transitions.noisy_true_pos
    
    #Encode height observations
    heights_encoded = jax.vmap(ppo_trainer.actor_critic.height_encoder.apply,(None,0))(params.height_encoder_parameters,actor_height_observation)

    #Predict linear velocity
    pred_linear_velocities = jax.vmap(ppo_trainer.actor_critic.linear_velocity_head.apply,(None,0))(params.linear_velocity_parameters,actor_observation)

    #Forward Matrix  Memory Part
    def forward_scan(carry,scan_input):
        current_matrix_memory,key = carry
        actor_obs, height_encoded, done,noisy_true_pos,linear_velocity = scan_input
        updated_matrix_memory = jax.vmap(ppo_trainer.actor_critic.update_matrix_memory,(None,0,0,0,0,0))(params,current_matrix_memory,height_encoded,actor_obs,noisy_true_pos,linear_velocity)
        new_matrix_memory = jnp.where(done[:,:,None,None],jnp.ones_like(updated_matrix_memory)*0.5,updated_matrix_memory)
        return (new_matrix_memory,key),(updated_matrix_memory)
    forward_carry = (matrix_memory[0],key)
    forward_scan_input= (actor_observation,heights_encoded,dones,noisy_true_pos,jax.lax.stop_gradient(pred_linear_velocities))
    (new_matrix_memory,key),(matrix_memory) = jax.lax.scan(forward_scan,forward_carry,forward_scan_input,unroll=False)

    #Compute Heads
    current_log_prob,current_loc,current_scale,predicted_distribution = jax.vmap(ppo_trainer.actor_critic.log_prob_from_observation,(0,None,0,0,0))(actor_actions,params,actor_observation,heights_encoded,matrix_memory)
    
    discount_factor = ppo_trainer.discount_factor
    #Compute Critic
    baseline = jax.vmap(ppo_trainer.actor_critic.evaluate_actions,(None,0,0))(params, critic_observation,critic_height_observation)
    bootstrap_value = ppo_trainer.actor_critic.evaluate_actions(params, critic_next_observation[-1],critic_next_height_observations[-1])
    terminated_bootstrap_value = jnp.zeros_like(bootstrap_value)
    bootstrap_value = jnp.where(dones[-1],terminated_bootstrap_value,bootstrap_value)
    next_value = jax.lax.stop_gradient(jnp.concatenate([baseline[1:],jnp.expand_dims(bootstrap_value, 0)],axis=0))

    # Compuate GAE
    accs = compute_gae(rewards, baseline, next_value, dones, truncations, discount_factor, ppo_trainer.gae_lambda)
    
    advantages = jax.lax.stop_gradient(accs)
    vs = jax.lax.stop_gradient(accs + baseline) # Use this for value loss
    
    # Advantage normalization
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    log_prob_diff = current_log_prob - data_collection_log_prob
    rho_s = jnp.exp(log_prob_diff)

    #Compute Policy Surrogate Loss
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1-ppo_trainer.clipping_epsilon, 1+ppo_trainer.clipping_epsilon) * advantages
    policy_loss = ppo_trainer.training_config["policy_loss_scaling_factor"] * -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
    
    # Value function loss
    v_error =vs - baseline
    v_loss = ppo_trainer.training_config["value_loss_scaling_factor"] * jnp.mean(v_error * v_error) * 0.5

    # Entropy reward
    entropy_cost  = interpolate_entropy_cost(ppo_trainer.min_entropy_cost, ppo_trainer.max_entropy_cost, training_steps,float(ppo_trainer.training_steps))
    entropy_vals= ppo_trainer.actor_critic.entropy(current_loc, current_scale)
    entropy = jnp.mean(entropy_vals)
    entropy_loss = entropy_cost * -entropy
        
    # BCE Loss
    bce_error = jnp.mean(compute_bce_loss(predicted_distribution,true_distributions))
    bce_loss = ppo_trainer.training_config["bce_loss_scaling_factor"] * jnp.mean(compute_bce_loss(predicted_distribution,true_distributions))

    #Linear Velocity loss 
    linear_velocity_error = true_linear_velocities - pred_linear_velocities
    linear_velocity_error_log = jnp.mean(jnp.abs(linear_velocity_error))
    linear_velocity_error_loss = ppo_trainer.training_config["linear_velocity_loss_scaling_factor"] * jnp.mean(linear_velocity_error*linear_velocity_error)*0.5
    
    # KL divergence
    logratio = current_log_prob - data_collection_log_prob
    ratio = jnp.exp(logratio)
    approx_kl_div = jnp.mean((ratio - 1) - logratio)
    total_loss =  policy_loss +  v_loss + entropy_loss +  bce_loss + linear_velocity_error_loss
    losses = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "entropy": entropy,
        "approx_kl_div": approx_kl_div,
        "bce_loss": bce_loss,
        "bce_error": bce_error,
        "discount_factor":discount_factor,
        "linear_velocity_error_log": linear_velocity_error_log,
    }
    return total_loss, (losses,key)
