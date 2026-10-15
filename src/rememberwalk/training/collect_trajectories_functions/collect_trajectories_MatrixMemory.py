import jax.numpy as jnp
from rememberwalk.actor_critic.MatrixMemory import Transition,TrainingState
from rememberwalk.states.MazeState import MazeState
from rememberwalk.training.ppo import PPOTrainer
import jax
from typing import Tuple
import optax

def collect_trajectories(ppo_trainer:PPOTrainer,env_state:MazeState,training_state:TrainingState,key):

    def one_step(carry, _):
        #Unpack carry
        old_state,old_matrix_memory, key = carry

        #Sample action
        action,scale, log_prob,predicted_dsitribution,matrix_memory = ppo_trainer.actor_critic.sample_action(training_state.params, old_state.obs, old_state.height_obs,old_matrix_memory,old_state.noisy_true_pos, key)
        
        #Compute Wall prediction reward
        auxiliary_reward = old_state.no_goal_position * compute_bce_reward(ppo_trainer,predicted_dsitribution,old_state.maze_description)
        
        #Update curriculum boundaries
        updated_max_allowed_curr_coeff = jnp.median(old_state.curr_coeff) 
        updated_max_allowed_terrain_curr_coeff = jnp.median(old_state.terrain_curr_coeff)
        old_state = old_state.replace(
            max_allowed_curr_coeff=old_state.max_allowed_curr_coeff.at[:].set(updated_max_allowed_curr_coeff),
            max_allowed_terrain_curr_coeff= old_state.max_allowed_terrain_curr_coeff.at[:].set(updated_max_allowed_terrain_curr_coeff),
        )

        #Environment step
        key, step_key = jax.random.split(key)
        env_keys = jax.random.split(step_key, ppo_trainer.nr_envs)
        env_state, env_keys = jax.vmap(ppo_trainer.environment.step)(old_state, action, env_keys)

        #Reset Matrix Memory
        matrix_memory = jnp.where(env_state.done[:, None, None, None],jnp.ones_like(matrix_memory)*0.5,matrix_memory)

        #Logging
        summed_reward = jnp.where(env_state.done, env_state.summed_reward, 0.0)
        episode_length = jnp.where(env_state.done, env_state.n_steps, 0.0)
        metrics = dict(env_state.metrics)
        metrics["auxiliary_reward"] =  auxiliary_reward
        metrics["scale"] = jnp.mean(scale,axis=-1)
        linear_velocity = jnp.concat([old_state.pipeline_state.qvel[:,:2],old_state.pipeline_state.qvel[:,5].reshape((-1,1))],axis=-1)

        #Create Transition
        transition = Transition(
            done=env_state.done,
            truncation=env_state.truncation,
            actor_observation=old_state.obs,
            action=action,
            reward=env_state.reward + auxiliary_reward ,
            actor_height_observation=old_state.height_obs,
            critic_next_observation=env_state.critic_obs,
            log_prob=log_prob,
            critic_next_height_observation=env_state.critic_height_obs,
            critic_height_observation=old_state.critic_height_obs,
            critic_observation=old_state.critic_obs,
            true_distribution= old_state.maze_description,
            matrix_memory=old_matrix_memory,
            linear_velocity=linear_velocity,
            noisy_true_pos = old_state.noisy_true_pos
        )
        return (env_state,matrix_memory,env_keys[-1]), (transition, metrics, summed_reward, episode_length)
    
    matrix_memory = training_state.last_matrix_memory
    carry, output = jax.lax.scan(one_step, (env_state,matrix_memory,key), (), ppo_trainer.unroll_length,unroll=10)
    env_state,matrix_memory, key = carry
    training_state = training_state.replace(last_matrix_memory=matrix_memory)
    transitions, training_metrics, summed_reward, episode_length = output
    key, new_key= jax.random.split(key)
    return env_state,training_state ,new_key, transitions, training_metrics, summed_reward, episode_length


def compute_bce_reward(ppo_trainer, predicted_distribution, true_distribution):
    loss = compute_bce_loss(predicted_distribution, true_distribution)
    bce_reward = (
        ppo_trainer.environment.dt
        * ppo_trainer.environment.parameters["reward_parameters"]["bce_reward"]
        * jnp.clip(0.30 - loss, 0.0, 0.30) / 0.30  # normalize to [0, 1]
    )
    return bce_reward

def compute_bce_loss(predicted_distribution: jnp.ndarray,true_distribution:jnp.ndarray):
    loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,true_distribution)
    loss = jnp.mean(loss,axis=-1)
    return loss