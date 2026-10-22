import jax.numpy as jnp
import optax
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from typing import Tuple
import jax

def experiment_callback(actor_critic,eval_dict:dict,env:SimpleMazeEnv) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:
    """Deterministic eval callback. Uses get deterministc action from Actor Critic for eval

    Args:
        eval_dict (dict): eval_dict contatining all important information for stepping the given architecture
        key (jnp.ndarray): jax random key to keep randomness

    Returns:
        Tuple[dict|jnp.ndarray]: (eval_dict with updated values,updated random key)
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    obs_history = eval_dict["observation_history"]
    obs_history = jnp.where(state.done[:, None, None], jnp.zeros_like(obs_history), obs_history)
    action,predicted_distribution,height_encoded,pred_linear_vel= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,obs_history)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel,state.noisy_true_pos], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    
    eval_dict["observation_history"] = new_obs_history
    
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = compute_bce_reward(predicted_distribution,state.maze_description,env)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    metrics["cell_visits"] = state.cell_visits
    metrics["predicted_distribution"] = predicted_distribution
    metrics["true_distribution"] = state.maze_description
    metrics["pos"] =  state.delta_pos
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["action"] = action
    return eval_dict,metrics

def deterministic_eval_callback(actor_critic,eval_dict:dict,env:SimpleMazeEnv) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:
    """Deterministic eval callback. Uses get deterministc action from Actor Critic for eval

    Args:
        eval_dict (dict): eval_dict contatining all important information for stepping the given architecture
        key (jnp.ndarray): jax random key to keep randomness

    Returns:
        Tuple[dict|jnp.ndarray]: (eval_dict with updated values,updated random key)
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]

    obs_history = eval_dict["observation_history"]
    obs_history = jnp.where(state.done[:, None, None], jnp.zeros_like(obs_history), obs_history)
    action,predicted_distribution,height_encoded,pred_linear_vel= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,obs_history)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel,state.noisy_true_pos], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5*bce_error)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    #metrics["cell_visits"] = state.cell_visits
    #metrics["predicted_distribution"] = predicted_distribution
    #metrics["true_distribution"] = state.maze_description
    #metrics["pos"] =  state.delta_pos
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["action"] = action
    return eval_dict,metrics

def probabilistic_eval_callback(actor_critic,eval_dict:dict,env:SimpleMazeEnv,key:jnp.ndarray) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:
    """Probabilistic eval callback. Uses sampled action from actor critic. (Same as in step_fn during training)

    Args:
        eval_dict (dict): eval_dict contatining all important information for stepping the given architecture
        key (jnp.ndarray): jax random key to keep randomness

    Returns:
        Tuple[dict|jnp.ndarray]: (eval_dict with updated values,updated random key)
    """

    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    obs_history = eval_dict["observation_history"]
    obs_history = jnp.where(state.done[:, None, None], jnp.zeros_like(obs_history), obs_history)
    action,predicted_distribution,height_encoded,pred_linear_vel= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,obs_history)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel,state.noisy_true_pos], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5*bce_error)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    metrics["pos"] =  state.delta_pos
    metrics["predicted_distribution"] = predicted_distribution
    metrics["true_distribution"] = state.maze_description
    eval_dict["state"] = state
    eval_dict["action"] = action
    eval_dict["predicted_distribution"] = predicted_distribution
    eval_dict["true_distribution"] = state.maze_description
    return eval_dict,metrics,key


def visualizer_callback(actor_critic,visualizer_dict:dict,key:jnp.ndarray) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:

    state = visualizer_dict["state"]
    parameters = visualizer_dict["parameters"]
    env = visualizer_dict["env"]
    state = visualizer_dict["state"]
    parameters = visualizer_dict["parameters"]
    obs_history = visualizer_dict["observation_history"]
    obs_history = jnp.where(state.done[:, None, None], jnp.zeros_like(obs_history), obs_history)
    action,predicted_distribution,height_encoded,predicted_delta_position,pred_linear_vel= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,obs_history)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5*bce_error)
    metrics = state.metrics
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    visualizer_dict["state"] = state
    visualizer_dict["action"] = action
    visualizer_dict["predicted_distribution"] = predicted_distribution
    visualizer_dict["true_distribution"] = state.maze_description
    visualizer_dict["pred_delta_loc"] =  predicted_delta_position
    return visualizer_dict,key


def compute_bce_loss(predicted_distribution: jnp.ndarray,true_distribution:jnp.ndarray):
    loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,true_distribution)
    loss = jnp.mean(loss,axis=-1)
    return loss


def compute_bce_reward(predicted_distribution, true_distribution,env):
    loss = compute_bce_loss(predicted_distribution, true_distribution)
    bce_reward = (
        env.dt
        * env.parameters["reward_parameters"]["bce_reward"]
        * jnp.clip(0.30 - loss, 0.0, 0.30) / 0.30  # normalize to [0, 1]
    )
    return bce_reward