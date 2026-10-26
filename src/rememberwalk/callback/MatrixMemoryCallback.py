"""Per-step eval callbacks of the LSMM approaches (``imports.experiment_callback``,
``deterministic_eval_callback``, ``visualizer_callback``).

A callback runs the deterministic policy for all envs of ``eval_dict``, keeps the approach's memory (the
matrix memory, reset to 0.5 at episode end) and returns the per-step metrics. The experiment callback adds
what the paper plots need.
"""

import jax
import jax.numpy as jnp

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.training.bce import compute_bce_loss, compute_bce_reward, compute_exponential_bce_reward


def _eval_step(actor_critic, eval_dict: dict, env: SimpleMazeEnv, full_metrics: bool) -> tuple[dict, dict]:
    """One deterministic policy step for all envs; the matrix memory is reset to 0.5 where an episode ended.

    ``full_metrics`` adds the per-step maps and positions the experiment plots need, and logs the
    training BCE reward instead of the exponential one.
    """
    state = eval_dict["state"]
    parameters = eval_dict["parameters"]
    matrix_memory = eval_dict["matrix_memory"]
    action, predicted_distribution, matrix_memory_updated = actor_critic.get_deterministic_action(
        parameters, state.obs, state.height_obs, matrix_memory, state.noisy_true_pos
    )
    matrix_memory = matrix_memory_updated
    matrix_memory = jnp.where(
        state.done[:, None, None, None], jnp.ones_like(matrix_memory) * 0.5, matrix_memory
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
    eval_dict["matrix_memory"] = matrix_memory
    eval_dict["action"] = action
    eval_dict["predicted_distribution"] = predicted_distribution
    eval_dict["pred_delta_loc"] = state.delta_pos
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
    matrix_memory = visualizer_dict["matrix_memory"]
    action, predicted_distribution, matrix_memory = action_jit(
        parameters, state.obs, state.height_obs, matrix_memory, state.noisy_true_pos.reshape((-1, 3))
    )
    matrix_memory = jnp.where(state.done, jnp.ones_like(matrix_memory) * 0.5, matrix_memory)
    key, _ = jax.random.split(key)
    action = action[0]
    key, _ = jax.random.split(key)
    visualizer_dict["state"] = state
    visualizer_dict["matrix_memory"] = matrix_memory
    visualizer_dict["action"] = action
    visualizer_dict["predicted_distribution"] = predicted_distribution
    visualizer_dict["pred_delta_loc"] = state.delta_pos.reshape((1, 3))
    return visualizer_dict, key
