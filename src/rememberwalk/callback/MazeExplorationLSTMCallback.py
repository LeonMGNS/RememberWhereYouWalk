import jax.numpy as jnp
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.training.bce import compute_bce_loss, compute_bce_reward, compute_exponential_bce_reward
from typing import Tuple
import jax
import mujoco


def _eval_step(actor_critic,eval_dict:dict,env:SimpleMazeEnv,full_metrics:bool) -> Tuple[dict,dict]:
    """One deterministic policy step for all envs; the recurrent state is zeroed where an episode ended.

    ``full_metrics`` adds the per-step maps and positions the experiment plots need, and logs the
    training BCE reward instead of the exponential one.
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    lstm_state = eval_dict["lstm_state"]
    action,predicted_distribution,_,_,new_lstm_state= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,lstm_state,state.noisy_true_pos)

    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done[:,None], jnp.zeros_like(x), x),
        new_lstm_state,
    )

    #Compute bce error/reward
    bce_error = compute_bce_loss(predicted_distribution,state.maze_description)
    metrics = dict(state.metrics)
    metrics["bce_error"] =  bce_error
    if full_metrics:
        metrics["bce_reward"] = compute_bce_reward(predicted_distribution,state.maze_description,env)
        metrics["cell_visits"] = state.cell_visits
        metrics["predicted_distribution"] = predicted_distribution
        metrics["true_distribution"] = state.maze_description
        metrics["pos"] =  state.delta_pos
    else:
        metrics["bce_reward"] = compute_exponential_bce_reward(bce_error,env)
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["lstm_state"]=updated_lstm_state
    eval_dict["action"] = action
    return eval_dict,metrics


def experiment_callback(actor_critic,eval_dict:dict,env:SimpleMazeEnv) -> Tuple[dict,dict]:
    """Eval step of the experiment rollouts (experiments/data_creation.py): full metric set."""
    return _eval_step(actor_critic,eval_dict,env,full_metrics=True)


def deterministic_eval_callback(actor_critic,eval_dict:dict,env:SimpleMazeEnv) -> Tuple[dict,dict]:
    """Eval step of the training-time deterministic evaluation (PPOTrainer): scalar metrics only."""
    return _eval_step(actor_critic,eval_dict,env,full_metrics=False)


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
