"""Per-step eval callbacks of the LSTM and SRU (``imports.experiment_callback``,
``deterministic_eval_callback``, ``visualizer_callback``).

A callback runs the deterministic policy for all envs of ``eval_dict``, keeps the approach's memory (the
recurrent (c, h) state, zeroed at episode end) and returns the per-step metrics. The experiment callback
adds what the paper plots need.
"""

import jax
import jax.numpy as jnp
import mujoco

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.training.bce import compute_bce_loss, compute_bce_reward, compute_exponential_bce_reward


def _eval_step(actor_critic, eval_dict: dict, env: SimpleMazeEnv, full_metrics: bool) -> tuple[dict, dict]:
    """One deterministic policy step for all envs; the recurrent state is zeroed where an episode ended.

    ``full_metrics`` adds the per-step maps and positions the experiment plots need, and logs the
    training BCE reward instead of the exponential one.
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    lstm_state = eval_dict["lstm_state"]
    action, predicted_distribution, _, _, new_lstm_state = actor_critic.get_deterministic_action(
        parameters, state.obs, state.height_obs, lstm_state, state.noisy_true_pos
    )

    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done[:, None], jnp.zeros_like(x), x),
        new_lstm_state,
    )

    # Compute bce error/reward
    bce_error = compute_bce_loss(predicted_distribution, state.maze_description)
    metrics = dict(state.metrics)
    metrics["bce_error"] = bce_error
    if full_metrics:
        metrics["bce_reward"] = compute_bce_reward(predicted_distribution, state.maze_description, env)
        metrics["cell_visits"] = state.cell_visits
        metrics["predicted_distribution"] = predicted_distribution
        metrics["true_distribution"] = state.maze_description
        metrics["pos"] = state.delta_pos
    else:
        metrics["bce_reward"] = compute_exponential_bce_reward(bce_error, env)
    metrics["step"] = state.n_steps
    eval_dict["state"] = state
    eval_dict["lstm_state"] = updated_lstm_state
    eval_dict["action"] = action
    return eval_dict, metrics


def experiment_callback(actor_critic, eval_dict: dict, env: SimpleMazeEnv) -> tuple[dict, dict]:
    """Eval step of the experiment rollouts (experiments/data_creation.py): full metric set."""
    return _eval_step(actor_critic, eval_dict, env, full_metrics=True)


def deterministic_eval_callback(actor_critic, eval_dict: dict, env: SimpleMazeEnv) -> tuple[dict, dict]:
    """Eval step of the training-time deterministic evaluation (PPOTrainer): scalar metrics only."""
    return _eval_step(actor_critic, eval_dict, env, full_metrics=False)


def visualizer_callback(action_jit, visualizer_dict: dict, key: jnp.ndarray) -> tuple[dict, jnp.ndarray]:
    """Single-env step of scripts/visualize.py; ``action_jit`` is the jitted ``get_deterministic_action``."""
    state = visualizer_dict["state"]
    parameters = visualizer_dict["parameters"]
    env = visualizer_dict["env"]
    lstm_state = visualizer_dict["lstm_state"]
    action, predicted_distribution, _, _, new_lstm_state = action_jit(
        parameters, state.obs, state.height_obs, lstm_state, state.noisy_true_pos.reshape((-1, 3))
    )

    action = action[0]
    updated_lstm_state = jax.tree_util.tree_map(
        lambda x: jnp.where(state.done, jnp.zeros_like(x), x),
        new_lstm_state,
    )
    # Visualizing current goal heading
    goal_flag_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "goal_flag")
    goal_flag_pos = jnp.zeros(3)
    goal_flag_pos = goal_flag_pos.at[:2].set(state.goal_position)
    goal_flag_pos = goal_flag_pos.at[2].set(2.0)
    state = state.replace(
        model=state.model.replace(geom_pos=state.model.geom_pos.at[goal_flag_id].set(goal_flag_pos))
    )
    visualizer_dict["state"] = state
    visualizer_dict["lstm_state"] = updated_lstm_state
    visualizer_dict["action"] = action
    visualizer_dict["predicted_distribution"] = predicted_distribution
    visualizer_dict["pred_delta_loc"] = state.delta_pos.reshape((1, 3))
    return visualizer_dict, key
