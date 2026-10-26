import jax.numpy as jnp
import optax
from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.training.bce import compute_bce_loss, compute_bce_reward, compute_exponential_bce_reward
from typing import Tuple


def _eval_step(actor_critic,eval_dict:dict,env:SimpleMazeEnv,full_metrics:bool) -> Tuple[dict,dict]:
    """One deterministic policy step for all envs on the observation history (cleared where an episode ended).

    ``full_metrics`` adds the per-step maps and positions the experiment plots need, and logs the
    training BCE reward instead of the exponential one.
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    obs_history = eval_dict["observation_history"]
    obs_history = jnp.where(state.done[:, None, None], jnp.zeros_like(obs_history), obs_history)
    action,predicted_distribution,height_encoded,pred_linear_vel= actor_critic.get_deterministic_action(parameters,state.obs,state.height_obs,obs_history)
    new_obs_entry = jnp.concatenate([state.obs, height_encoded,pred_linear_vel,state.noisy_true_pos], axis=-1)
    new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
    new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
    if full_metrics:
        # B11 (RELEASE_CLEANUP_PLAN.md): only the experiment rollout writes the history back; the
        # training-time eval keeps acting on the empty history. Kept as is until decided in P5.
        eval_dict["observation_history"] = new_obs_history

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
