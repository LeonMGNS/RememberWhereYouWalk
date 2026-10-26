"""Rollout of the LSMM approaches (``imports.collect_trajectories_fn``).

The matrix memory is reset to 0.5 at episode ends.
"""

import jax
import jax.numpy as jnp

from rememberwalk.actor_critic.MatrixMemory import TrainingState, Transition
from rememberwalk.states.MazeState import MazeState
from rememberwalk.training.bce import compute_bce_reward
from rememberwalk.training.collect_trajectories_functions.linear_velocity_target import linear_velocity_target
from rememberwalk.training.ppo import PPOTrainer


def collect_trajectories(ppo_trainer: PPOTrainer, env_state: MazeState, training_state: TrainingState, key):
    """Steps all envs ``unroll_length`` times with the sampled policy (jitted ``lax.scan``).

    The exploration reward from the map prediction (``compute_bce_reward``) is added in exploration
    episodes.

    Returns:
        (env_state, training_state with the last memory, key, Transition (unroll step, env, ...), metrics,
        summed rewards, episode lengths).
    """

    def one_step(carry, _):
        # Unpack carry
        """One policy step and env step for all envs."""
        old_state, old_matrix_memory, key = carry

        # Sample action
        action, scale, log_prob, predicted_distribution, matrix_memory = (
            ppo_trainer.actor_critic.sample_action(
                training_state.params,
                old_state.obs,
                old_state.height_obs,
                old_matrix_memory,
                old_state.noisy_true_pos,
                key,
            )
        )

        # Compute Wall prediction reward
        auxiliary_reward = old_state.no_goal_position * compute_bce_reward(
            predicted_distribution, old_state.maze_description, ppo_trainer.environment
        )

        # Update curriculum boundaries
        updated_max_allowed_curr_coeff = jnp.median(old_state.curr_coeff)
        updated_max_allowed_terrain_curr_coeff = jnp.median(old_state.terrain_curr_coeff)
        old_state = old_state.replace(
            max_allowed_curr_coeff=old_state.max_allowed_curr_coeff.at[:].set(updated_max_allowed_curr_coeff),
            max_allowed_terrain_curr_coeff=old_state.max_allowed_terrain_curr_coeff.at[:].set(
                updated_max_allowed_terrain_curr_coeff
            ),
        )

        # Environment step
        key, step_key = jax.random.split(key)
        env_keys = jax.random.split(step_key, ppo_trainer.nr_envs)
        env_state, env_keys = jax.vmap(ppo_trainer.environment.step)(old_state, action, env_keys)

        # Reset Matrix Memory
        matrix_memory = jnp.where(
            env_state.done[:, None, None, None], jnp.ones_like(matrix_memory) * 0.5, matrix_memory
        )

        # Logging
        summed_reward = jnp.where(env_state.done, env_state.summed_reward, 0.0)
        episode_length = jnp.where(env_state.done, env_state.n_steps, 0.0)
        metrics = dict(env_state.metrics)
        metrics["auxiliary_reward"] = auxiliary_reward
        metrics["scale"] = jnp.mean(scale, axis=-1)
        linear_velocity = jax.vmap(linear_velocity_target)(
            old_state.pipeline_state.qpos, old_state.pipeline_state.qvel
        )

        # Create Transition
        transition = Transition(
            done=env_state.done,
            truncation=env_state.truncation,
            actor_observation=old_state.obs,
            action=action,
            reward=env_state.reward + auxiliary_reward,
            actor_height_observation=old_state.height_obs,
            critic_next_observation=env_state.critic_obs,
            log_prob=log_prob,
            critic_next_height_observation=env_state.critic_height_obs,
            critic_height_observation=old_state.critic_height_obs,
            critic_observation=old_state.critic_obs,
            true_distribution=old_state.maze_description,
            matrix_memory=old_matrix_memory,
            linear_velocity=linear_velocity,
            noisy_true_pos=old_state.noisy_true_pos,
        )
        return (env_state, matrix_memory, env_keys[-1]), (transition, metrics, summed_reward, episode_length)

    matrix_memory = training_state.last_matrix_memory
    carry, output = jax.lax.scan(
        one_step, (env_state, matrix_memory, key), (), ppo_trainer.unroll_length, unroll=10
    )
    env_state, matrix_memory, key = carry
    training_state = training_state.replace(last_matrix_memory=matrix_memory)
    transitions, training_metrics, summed_reward, episode_length = output
    key, new_key = jax.random.split(key)
    return env_state, training_state, new_key, transitions, training_metrics, summed_reward, episode_length
