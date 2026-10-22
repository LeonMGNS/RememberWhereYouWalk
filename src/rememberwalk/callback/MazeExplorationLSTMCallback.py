import jax.numpy as jnp
import optax
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from typing import Tuple
import jax
import mujoco

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
    lstm_state = eval_dict["lstm_state"]
    action,predicted_distribution,_,_,new_lstm_state= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,lstm_state,state.noisy_true_pos)
    
    # Visualizing current goal heading
    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done[:,None], jnp.zeros_like(x), x),
        new_lstm_state,
    )
    
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
    eval_dict["lstm_state"]=updated_lstm_state
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
    lstm_state = eval_dict["lstm_state"]
    action,predicted_distribution,_,_,new_lstm_state= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,lstm_state,state.noisy_true_pos)
    
    # Visualizing current goal heading
    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done[:,None], jnp.zeros_like(x), x),
        new_lstm_state,
    )
    
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
    eval_dict["lstm_state"]=updated_lstm_state
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
    lstm_state = eval_dict["lstm_state"]
    action,_,_,_,predicted_distribution,new_pred_loc,_,new_lstm_state= actor_critic.sample_action(parameters,state.obs,state.height_obs,lstm_state,state.noisy_true_pos,key)
    
    # Visualizing current goal heading
    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done[:,None], jnp.zeros_like(x), x),
        new_lstm_state,
    )
    
    #Compute bce error/reward
    bce_error = jnp.mean(optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,state.maze_description),axis=-1)
    bce_reward = env.dt * env.parameters["reward_parameters"]["bce_reward"] * jnp.exp(-5*bce_error)
    metrics = state.metrics
    metrics["bce_error"] =  bce_error
    metrics["bce_reward"] = bce_reward
    eval_dict["state"] = state
    eval_dict["lstm_state"]=updated_lstm_state
    eval_dict["action"] = action
    eval_dict["predicted_distribution"] = predicted_distribution
    eval_dict["pred_pos"] =  new_pred_loc
    return eval_dict,metrics,key


def visualizer_callback(actor_critic,visualizer_dict:dict,key:jnp.ndarray) -> Tuple[jnp.ndarray|jnp.ndarray|jnp.ndarray]:

    state = visualizer_dict["state"]
    parameters = visualizer_dict["parameters"]
    env = visualizer_dict["env"]
    lstm_state = visualizer_dict["lstm_state"]
    action,predicted_distribution,_,_,new_lstm_state= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,lstm_state,state.noisy_true_pos)
    
    action = action[0]
    # Visualizing current goal heading
    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done, jnp.zeros_like(x), x),
        new_lstm_state,
    )
    goal_flag_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_flag")
    goal_flag_pos = jnp.zeros(3)
    goal_flag_pos = goal_flag_pos.at[:2].set(state.goal_position)
    goal_flag_pos = goal_flag_pos.at[2].set(2.0)
    state = state.replace(model=state.model.replace(geom_pos=state.model.geom_pos.at[goal_flag_id].set(goal_flag_pos)))
    visualizer_dict["state"] = state
    visualizer_dict["lstm_state"]=updated_lstm_state
    visualizer_dict["action"] = action
    visualizer_dict["predicted_distribution"] = predicted_distribution
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