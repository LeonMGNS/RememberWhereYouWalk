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
    matrix_memory = eval_dict["matrix_memory"]
    action,predicted_distribution,matrix_memory_updated= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,matrix_memory,state.noisy_true_pos)
    matrix_memory = matrix_memory_updated
    matrix_memory = jnp.where(state.done[:,None,None,None],jnp.ones_like(matrix_memory)*0.5,matrix_memory)
    
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = compute_bce_reward(predicted_distribution,state.maze_description,env)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    # Same metric set as the other experiment callbacks; the experiment plots read all of them
    metrics["cell_visits"] = state.cell_visits
    metrics["predicted_distribution"] = predicted_distribution
    metrics["true_distribution"] = state.maze_description
    metrics["pos"] =  state.delta_pos
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["matrix_memory"]=matrix_memory
    eval_dict["action"] = action
    eval_dict["predicted_distribution"] = predicted_distribution
    eval_dict["pred_delta_loc"] =  state.delta_pos
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
    matrix_memory = eval_dict["matrix_memory"]
    action,predicted_distribution,matrix_memory_updated= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,matrix_memory,state.noisy_true_pos)
    matrix_memory = matrix_memory_updated
    matrix_memory = jnp.where(state.done[:,None,None,None],jnp.ones_like(matrix_memory)*0.5,matrix_memory)
    
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5*bce_error)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["matrix_memory"]=matrix_memory
    eval_dict["action"] = action
    eval_dict["predicted_distribution"] = predicted_distribution
    eval_dict["pred_delta_loc"] =  state.delta_pos
    return eval_dict,metrics

def visualizer_callback(action_jit,visualizer_dict:dict,key:jnp.ndarray) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:

    state = visualizer_dict["state"]
    parameters = visualizer_dict["parameters"]
    matrix_memory = visualizer_dict["matrix_memory"]
    action,predicted_dsitribution,matrix_memory= action_jit(parameters,state.obs,state.height_obs,matrix_memory,state.noisy_true_pos.reshape((-1,3)))
    matrix_memory = jnp.where(state.done,jnp.ones_like(matrix_memory)*0.5,matrix_memory)
    key,_ = jax.random.split(key)
    action = action[0]
    key,_ = jax.random.split(key)
    visualizer_dict["state"] = state
    visualizer_dict["matrix_memory"]=matrix_memory
    visualizer_dict["action"] = action
    visualizer_dict["predicted_distribution"] = predicted_dsitribution
    visualizer_dict["pred_delta_loc"] = state.delta_pos.reshape((1,3))
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