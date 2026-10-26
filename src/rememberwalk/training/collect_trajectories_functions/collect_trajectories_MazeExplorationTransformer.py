import jax.numpy as jnp
from rememberwalk.actor_critic.Transformer import Transition, TrainingState
from rememberwalk.states.MazeState import MazeState
from rememberwalk.training.ppo import PPOTrainer
from rememberwalk.training.bce import compute_bce_reward
from rememberwalk.training.collect_trajectories_functions.linear_velocity_target import linear_velocity_target
import jax


def collect_trajectories(ppo_trainer: PPOTrainer, env_state: MazeState , training_state: TrainingState, key):

    def one_step(carry, _):

        #Unpack Carry
        old_state, key = carry
        obs_history = old_state.observation_history

        # Sample action
        action,scale, log_prob, height_encoded,predicted_distribution,pred_linear_vel = ppo_trainer.actor_critic.sample_action(
            training_state.params, old_state.obs, old_state.height_obs,
            obs_history, key
        )

        # Update curriculum boundaries
        updated_max_allowed_curr_coeff = jnp.median(old_state.curr_coeff) 
        updated_max_allowed_terrain_curr_coeff = jnp.median(old_state.terrain_curr_coeff)
        old_state = old_state.replace(
            max_allowed_curr_coeff=old_state.max_allowed_curr_coeff.at[:].set(updated_max_allowed_curr_coeff),
            max_allowed_terrain_curr_coeff= old_state.max_allowed_terrain_curr_coeff.at[:].set(updated_max_allowed_terrain_curr_coeff),
        )
        key, step_key = jax.random.split(key)
        env_keys = jax.random.split(step_key, ppo_trainer.nr_envs)
        env_state, env_keys = jax.vmap(ppo_trainer.environment.step)(old_state, action, env_keys)

        # Update observation history: roll and add [actor_obs || encoded_height] at the last position.
        # Zero-out the whole history on episode done so the next episode starts clean.
        new_obs_entry = jnp.concatenate([old_state.obs, height_encoded,pred_linear_vel,old_state.noisy_true_pos], axis=-1)
        new_obs_history = jnp.roll(obs_history, shift=-1, axis=1)
        new_obs_history = new_obs_history.at[:, -1, :].set(new_obs_entry)
        new_obs_history = jnp.where(env_state.done[:, None, None], jnp.zeros_like(new_obs_history), new_obs_history)
        env_state = env_state.replace(observation_history=new_obs_history)
        
        #Compute Wall prediction reward
        auxiliary_reward = old_state.no_goal_position * compute_bce_reward(predicted_distribution,old_state.maze_description,ppo_trainer.environment)

        #Logging
        summed_reward = jnp.where(env_state.done, env_state.summed_reward, 0.0)
        episode_length = jnp.where(env_state.done, env_state.n_steps, 0.0)
        linear_velocity = jax.vmap(linear_velocity_target)(old_state.pipeline_state.qpos, old_state.pipeline_state.qvel)
        metrics = dict(env_state.metrics)
        metrics["auxiliary_reward"] =  auxiliary_reward
        metrics["max_loc_error"] = jnp.zeros_like(env_state.reward)
        metrics["avg_loc_error"] = jnp.zeros_like(env_state.reward)
        metrics["scale"] = jnp.mean(scale,axis=-1)
        metrics["use_pred_loc"] = env_state.use_pred_loc

        #Create Transition
        transition = Transition(
                done = env_state.done,
                truncation = env_state.truncation,
                actor_observation = old_state.obs,
                actor_height_observation = old_state.height_obs,
                critic_observation = old_state.critic_obs,
                critic_height_observation = old_state.critic_height_obs,
                delta_position = old_state.delta_pos,
                action = action,
                reward=env_state.reward + auxiliary_reward, 
                critic_next_observation = env_state.critic_obs,
                critic_next_height_observation =  env_state.critic_height_obs,
                log_prob = log_prob,
                true_distribution = old_state.maze_description,
                linear_velocity = linear_velocity,
                noisy_true_pos = old_state.noisy_true_pos
        )
        return (env_state, env_keys[-1]), (transition, metrics, summed_reward, episode_length)

    carry, output = jax.lax.scan(one_step, (env_state, key), (), ppo_trainer.unroll_length)
    env_state, key = carry
    transitions, training_metrics, summed_reward, episode_length = output
    key, new_key = jax.random.split(key)
    return env_state, training_state, new_key, transitions, training_metrics, summed_reward, episode_length
