"""PPO trainer shared by every approach.

The approach plugs in through the training config: ``actor_critic`` (networks), ``collect_trajectories_fn``
(rollout with the approach's memory), ``loss_fn`` (PPO + auxiliary head losses) and
``deterministic_eval_callback`` (evaluation).
"""

import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import serialization

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.utils.paths import ROOT_DIR


class PPOTrainer:
    """Runs PPO with vmapped MJX envs: rollout, minibatch SGD with KL early stopping, logging, evaluations
    at curriculum milestones and orbax checkpoints.

    Args:
        environment: the (single-env) environment; it is vmapped over ``nr_envs``.
        training_config: the ``training`` section of the setup from ``generate_setup``.
        ckpt_dir: directory of the rotating checkpoints; ``<ckpt_dir>_curr_complete`` keeps the first
            checkpoint with the curriculum complete.
        run: run name (log directory).
        seed: PRNG seed of the training.
    """

    def __init__(
        self,
        environment: SimpleMazeEnv,
        training_config: dict,
        ckpt_dir: str,
        run: str,
        seed: int = 0,
    ):

        # Set parameters needed in trainer

        # Training Parameters
        self.environment = environment
        self.training_config = training_config
        self.training_steps = training_config["training_steps"]
        self.debug = training_config["debug"]
        self.batch_size = training_config["batch_size"]
        self.actor_critic = training_config["actor_critic"]
        self.nr_envs = training_config["nr_envs"]
        self.optimizer = training_config["optimizer"]
        self.minibatch_size = training_config["minibatch_size"]
        self.discount_factor = training_config["discount_factor"]
        self.gae_lambda = training_config["gae_lambda"]
        self.clipping_epsilon = training_config["clipping_epsilon"]
        self.min_entropy_cost = training_config["min_entropy_cost"]
        self.max_entropy_cost = training_config["max_entropy_cost"]
        self.million_steps_per_logging = training_config["million_steps_per_logging"]
        self.million_steps_per_model_save = (
            training_config["million_steps_per_model_save"] / self.million_steps_per_logging
        )
        self.num_updates_per_batch = training_config["num_updates_per_batch"]
        self.kl_div_thr = training_config["kl_div_thr"]
        self.loss_fn = training_config["loss_fn"]
        self.transformer_mask = training_config["transformer_mask"]
        self.collect_trajectories_fn = training_config["collect_trajectories_fn"]
        self.initial_training_state = training_config["training_state"]
        assert self.batch_size % self.minibatch_size == 0
        assert self.batch_size % self.nr_envs == 0
        self.unroll_length = self.batch_size // self.nr_envs
        self.num_minibatches = self.batch_size // self.minibatch_size
        self.num_epochs = self.training_steps // self.batch_size
        self.num_training_steps_per_epoch = self.training_steps // self.num_epochs
        self.num_training_steps_per_minibatch = self.num_training_steps_per_epoch // (
            self.num_minibatches * self.num_updates_per_batch
        )
        self.seed = seed

        # Evalutaion parameters
        self.evaluation_at_curriculum_steps = training_config["evaluation_at_curriculum_steps"]
        self.nr_eval_envs = training_config["nr_eval_envs"]
        self.deterministic_eval_callback = training_config["deterministic_eval_callback"]
        self.eval_dict = training_config["eval_dict"]

        # Init Objects needed in trainer
        options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
        self.checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.ckpt_dir, self.run = ckpt_dir, run
        self.logging_dir = f"{ROOT_DIR}/logs/{self.run}"
        self.checkpoint_manager = ocp.CheckpointManager(self.ckpt_dir, self.checkpointer, options)

        # Converged checkpoint
        self.secured_ckpt_dir = f"{self.ckpt_dir}_curr_complete"
        secured_options = ocp.CheckpointManagerOptions(max_to_keep=None, create=True)
        self.secured_checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
        self.secured_checkpoint_manager = ocp.CheckpointManager(
            self.secured_ckpt_dir, self.secured_checkpointer, secured_options
        )
        self._secured_ckpt_saved = False

        if self.debug:
            self.jitted_training_epoch = self.training_epoch
            self.jitted_reset_function = self.environment.reset
            self.evaluation_function = self.run_deterministic_evaluation
        else:
            self.jitted_training_epoch = jax.jit(self.training_epoch, static_argnums=())
            self.jitted_reset_function = jax.jit(self.environment.reset, static_argnums=())
            self.evaluation_function = jax.jit(self.run_deterministic_evaluation, static_argnums=())

    def save_ckpt(self, state, step: int):
        """Saves the TrainingState at ``step`` unless that step is already saved."""
        if self.checkpoint_manager.latest_step() == int(step):
            return
        state_dict = serialization.to_state_dict(state)
        self.checkpoint_manager.save(int(step), state_dict)

    def save_secured_ckpt(self, state, step: int):
        """Saves the TrainingState into the persistent curriculum-complete checkpoint directory."""
        state_dict = serialization.to_state_dict(state)
        self.secured_checkpoint_manager.save(int(step), state_dict)

    def train(self, logger):
        """Trains for ``training_steps`` env steps, logging to ``logger``. Returns the final TrainingState."""
        key = jax.random.PRNGKey(self.seed)

        # Initialize training environment state
        reset_keys = jax.random.split(key, self.nr_envs)
        key, _ = jax.random.split(key)
        env_state = jax.vmap(self.jitted_reset_function, (None, None, 0))(
            self.environment.parameters["initial_curr_coeff"],
            self.environment.parameters["initial_terrain_curr_coeff"],
            reset_keys,
        )

        # Init training loop carry
        training_state = self.initial_training_state
        carry = (env_state, training_state, key)
        latest_logging_step = 0

        # Start training loop
        carry, latest_logging_step = self.training_epoch_loop(carry, latest_logging_step, logger)
        env_state, training_state, key = carry
        self.save_ckpt(training_state, training_state.training_steps)
        self.checkpoint_manager.wait_until_finished()
        return training_state

    def training_epoch_loop(self, carry, latest_logging_step, logger):
        """Python loop over jitted epochs: logging every ``million_steps_per_logging`` and a deterministic
        evaluation each time the median curriculum passes the next ``evaluation_at_curriculum_steps`` value.
        """
        logging_counter = 0
        env_state, training_state, key = carry
        trained_steps = 0
        next_eval_curriculum_counter = 0
        eval_key = jax.random.PRNGKey(int(time.time()))
        while trained_steps < self.training_steps:
            (
                env_state,
                training_state,
                key,
                optimize_metrics,
                training_metrics,
                summed_reward,
                episode_length,
            ) = self.jitted_training_epoch(env_state, training_state, key)
            trained_steps = float(training_state.training_steps)
            print(f"Trained Steps: {trained_steps}")

            # Secure the checkpoint once the curriculum is complete (all envs at 1.0).
            if (not self._secured_ckpt_saved) and (float(jnp.mean(env_state.curr_coeff)) >= 1.0):
                self.save_secured_ckpt(training_state, training_state.training_steps)  # persistent copy
                self.save_ckpt(training_state, training_state.training_steps)  # alongside the rotating ones
                self._secured_ckpt_saved = True
                print(f"Secured curriculum-complete checkpoint at step {trained_steps}")

            if (trained_steps - latest_logging_step) >= (self.million_steps_per_logging * 1e6):
                logging_counter += 1
                latest_logging_step = float(training_state.training_steps)
                training_metrics, eval_key = self.compute_training_metrics(
                    training_metrics, summed_reward, episode_length, eval_key
                )
                optimize_metrics = self.compute_optimize_metrics(optimize_metrics)
                logger.log(training_metrics, step=int(training_state.training_steps))
                logger.log(optimize_metrics, step=int(training_state.training_steps))
                if logging_counter % self.million_steps_per_model_save == 0:
                    self.save_ckpt(training_state, training_state.training_steps)

            if (next_eval_curriculum_counter < len(self.evaluation_at_curriculum_steps)) and (
                jnp.median(env_state.curr_coeff)
                >= self.evaluation_at_curriculum_steps[next_eval_curriculum_counter]
            ):
                eval_curr_coeff = self.evaluation_at_curriculum_steps[next_eval_curriculum_counter]
                eval_terrain_curr_coeff = jnp.median(env_state.terrain_curr_coeff)
                eval_reset_keys = jax.random.split(key, self.nr_eval_envs)
                key, _ = jax.random.split(key)
                eval_env_state = jax.vmap(self.jitted_reset_function, (None, None, 0))(
                    eval_curr_coeff, eval_terrain_curr_coeff, eval_reset_keys
                )
                eval_dict = deepcopy(self.eval_dict)
                eval_dict["state"] = eval_env_state
                eval_dict["parameters"] = training_state.params

                eval_dict["action"] = jnp.zeros((self.nr_eval_envs, self.actor_critic.action_size))
                if hasattr(self.actor_critic, "distribution_size"):
                    eval_dict["predicted_distribution"] = jnp.zeros(
                        (self.nr_eval_envs, self.actor_critic.distribution_size)
                    )
                    eval_dict["pred_delta_loc"] = jnp.zeros((self.nr_eval_envs, 3))
                if hasattr(self.actor_critic, "env_distribution_size"):
                    eval_dict["predicted_env_distribution"] = jnp.zeros(
                        (self.nr_eval_envs, self.actor_critic.env_distribution_size)
                    )

                eval_metrics, eval_episode_lengths, eval_summed_rewards = self.evaluation_function(
                    eval_dict, key
                )
                evaluation_metrics = self.compute_evaluation_metrics(
                    eval_metrics, eval_summed_rewards, eval_episode_lengths
                )
                logger.log(evaluation_metrics, step=int(training_state.training_steps))
                next_eval_curriculum_counter += 1
        return (env_state, training_state, key), latest_logging_step

    def training_epoch(self, env_state, training_state, key):
        """One rollout of unroll_length steps and the PPO update on it (jitted)."""
        env_state, training_state, key, transitions, training_metrics, summed_reward, episode_length = (
            self.collect_trajectories_fn(self, env_state, training_state, key)
        )
        training_state, key, optimize_metrics = self.optimize(training_state, transitions, key)
        return (
            env_state,
            training_state,
            key,
            optimize_metrics,
            training_metrics,
            summed_reward,
            episode_length,
        )

    def minibatch_sgd_step(self, carry, transitions):
        """One gradient step on a minibatch. Once the approximate KL exceeds ``kl_div_thr``, all further
        updates of this epoch are skipped.
        """
        training_state, kl_div_stop, key = carry
        (loss, aux_data), grads = jax.value_and_grad(self.loss_fn, has_aux=True, argnums=1)(
            self, training_state.params, transitions, training_state.training_steps, key
        )
        params_update, optimizer_state = self.optimizer.update(
            grads, training_state.optimizer_state, params=training_state.params
        )
        params = optax.apply_updates(training_state.params, params_update)
        new_training_state = training_state.replace(optimizer_state=optimizer_state, params=params)
        loss_metrics = aux_data[0]
        key = aux_data[1]
        kl_div_stop = (jnp.abs(jnp.mean(loss_metrics["approx_kl_div"])) > self.kl_div_thr) | kl_div_stop
        loss_metrics["kl_div_stop"] = kl_div_stop
        training_state = jax.lax.cond(kl_div_stop, lambda: training_state, lambda: new_training_state)
        # Count the collected env steps even if the KL early stop rejected the update
        training_state = training_state.replace(
            training_steps=training_state.training_steps + self.num_training_steps_per_minibatch
        )
        return (training_state, kl_div_stop, key), loss_metrics

    def sgd_step(self, carry, _):
        """One pass over all minibatches."""
        training_state, transitions, kl_div_stop, key = carry
        (new_training_state, kl_div_stop, key), metrics = jax.lax.scan(
            self.minibatch_sgd_step,
            (training_state, kl_div_stop, key),
            transitions,
            length=self.num_minibatches,
        )
        return (new_training_state, transitions, kl_div_stop, key), metrics

    def optimize(self, training_state, transitions, key):
        """Shuffles the rollout over envs into minibatches and runs ``num_updates_per_batch`` passes. Returns
        (TrainingState, key, metrics).
        """

        def process_transitions(transitions):
            transitions = jnp.swapaxes(transitions, 0, 1)
            shuffled_transitions = jax.random.permutation(key, transitions, axis=0)
            original_shape = shuffled_transitions.shape
            # New shape: (num_minibatches, minibatch_size, unroll_length, features...)
            new_shape = (self.num_minibatches, -1) + original_shape[1:]
            shuffled_transitions = jnp.reshape(shuffled_transitions, new_shape)
            return shuffled_transitions

        processed_transitions = jax.tree_util.tree_map(process_transitions, transitions)
        key, _ = jax.random.split(key)
        kl_div_stop = False
        carry = (training_state, processed_transitions, kl_div_stop, key)
        carry, metrics = jax.lax.scan(self.sgd_step, carry, (), length=self.num_updates_per_batch)
        new_training_state, transitions, kl_div_stop, key = carry
        return new_training_state, key, metrics

    def run_deterministic_evaluation(self, eval_dict: dict, key):
        """Deterministic rollout of ``episode_length`` steps of all eval envs.

        Metrics of envs whose episode already ended are masked to 0.

        Returns:
            (per-step metrics, episode lengths (nr_eval_envs,), episode returns (nr_eval_envs,)).
        """
        # Initialize evaluation environment state
        states_already_terminated = jnp.zeros((self.nr_eval_envs,), dtype=jnp.int32)
        episode_lengths = jnp.zeros((self.nr_eval_envs,), dtype=jnp.int32)
        summed_rewards = jnp.zeros((self.nr_eval_envs,), dtype=jnp.float32)

        def eval_step(carry, _):
            eval_dict, state_already_terminated, episode_length, summed_reward, key = carry
            eval_dict, metrics = self.deterministic_eval_callback(
                self.actor_critic, eval_dict, self.environment
            )
            state_keys = jax.random.split(key, self.nr_eval_envs)
            next_state, state_keys = jax.vmap(self.environment.step)(
                eval_dict["state"], eval_dict["action"], state_keys
            )
            eval_dict["state"] = next_state
            episode_length = jnp.where(state_already_terminated, episode_length, next_state.n_steps)
            masked_metrics = jax.tree_util.tree_map(
                lambda metric: jnp.where(state_already_terminated, 0.0, metric), metrics
            )
            just_finished = (
                eval_dict["state"].done | (eval_dict["state"].n_steps >= self.environment.episode_length - 1)
            ) & (~state_already_terminated.astype(jnp.bool_))
            summed_reward = jnp.where(just_finished, eval_dict["state"].summed_reward, summed_reward)
            state_already_terminated = jnp.where(
                eval_dict["state"].done, jnp.ones_like(state_already_terminated), state_already_terminated
            )
            return (
                eval_dict,
                state_already_terminated,
                episode_length,
                summed_reward,
                state_keys[-1],
            ), masked_metrics

        (eval_dict, states_already_terminated, episode_lengths, summed_rewards, key), metrics = jax.lax.scan(
            eval_step,
            (eval_dict, states_already_terminated, episode_lengths, summed_rewards, key),
            None,
            length=self.environment.episode_length,
        )
        return metrics, episode_lengths, summed_rewards

    def compute_training_metrics(self, metrics: dict, summed_reward, episode_length, eval_key):
        """Mean/std of the rollout metrics for wandb."""
        result = {}
        for key, value in metrics.items():
            if key == "loc_log_distribution":
                eval_key, subkey = jax.random.split(eval_key)
                leading_shape = value.shape[:-1]
                flat_idx = jax.random.randint(
                    subkey, shape=(), minval=0, maxval=jnp.prod(jnp.array(leading_shape))
                )
                single_distribution = jnp.exp(value.reshape(-1, value.shape[-1])[flat_idx])  # (dist_shape,)
                img_shape = jnp.sqrt(single_distribution.shape[0]).astype(jnp.int32)
                result[f"Rollout/{key}"] = wandb.Image(
                    np.array(single_distribution.reshape((img_shape, img_shape)))
                )
            else:
                mean_value = jnp.mean(value)
                std_value = jnp.std(value)
                result[f"Rollout/{key}_mean"] = mean_value
                result[f"Rollout/{key}_std"] = std_value

        summed_reward = jnp.nan_to_num(summed_reward, nan=0.0, posinf=0.0, neginf=0.0)
        non_zero_summed_reward_idx = jnp.nonzero(summed_reward)
        non_zero_episode_length_idx = jnp.nonzero(episode_length)
        result["Rollout/summed_reward_mean"] = jnp.mean(summed_reward[non_zero_summed_reward_idx])
        result["Rollout/episode_length_mean"] = jnp.mean(episode_length[non_zero_episode_length_idx])
        result["Rollout/summed_reward_std"] = jnp.std(summed_reward[non_zero_summed_reward_idx])
        result["Rollout/episode_length_std"] = jnp.std(episode_length[non_zero_episode_length_idx])
        return result, eval_key

    def compute_evaluation_metrics(self, metrics: dict, summed_reward, episode_length):
        """Summary of an evaluation for wandb."""
        step_indices = jnp.arange(self.environment.episode_length)[:, None]
        mask = step_indices < episode_length
        result = {}
        for key, value in metrics.items():
            mean_value = jnp.mean(value[mask])
            std_value = jnp.std(value[mask])
            result[f"Eval/{key}_mean"] = mean_value
            result[f"Eval/{key}_std"] = std_value

        summed_reward = jnp.nan_to_num(summed_reward, nan=0.0, posinf=0.0, neginf=0.0)
        result["Eval/summed_reward_mean"] = jnp.mean(summed_reward)
        result["Eval/episode_length_mean"] = jnp.mean(episode_length)
        result["Eval/summed_reward_std"] = jnp.std(summed_reward)
        result["Eval/episode_length_std"] = jnp.std(episode_length)
        return result

    def compute_optimize_metrics(self, metrics: dict):
        """Mean of the loss metrics for wandb."""
        result = {}
        for key, value in metrics.items():
            mean_value = jnp.mean(value)
            std_value = jnp.std(value)
            result[f"Optimization/{key}_mean"] = mean_value
            result[f"Optimization/{key}_std"] = std_value
        return result
