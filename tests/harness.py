"""Shared helpers for the release-cleanup safety net (see RELEASE_CLEANUP_PLAN.md, Phase 0).

The tests build every training config in a shrunken form (few envs, short episodes), run a
fixed-seed workload and compare the result against golden files recorded on the
``pre-cleanup`` tag. Golden files are hardware and JAX-version specific: record them with
``pytest tests --update-golden`` on the machine you test on, before changing any code.
"""
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from rememberwalk.utils.paths import CKPT_DIR, CONFIG_DIR

GOLDEN_DIR = Path(__file__).parent / "golden"

# Every training config of the paper; lsmm_deployment covers the small-maze variants
CONFIGS = ["lsmm", "lsmm_additive", "lsmm_deployment", "lstm", "sru", "transformer", "fixed_window"]

# Original (thesis) paper checkpoints the goldens were recorded with, and their training config.
# experiment_checkpoints/ now holds the retrained runs, which have different weights.
CHECKPOINTS = {
    "LSMM": ("lsmm", "experiment_checkpoints_old/LSMM/policy_1"),
    "LSTM": ("lstm", "experiment_checkpoints_old/LSTM/policy_1"),
    "FixedWindow": ("fixed_window", "experiment_checkpoints_old/FixedWindow/policy_1"),
    "Transformer": ("transformer", "experiment_checkpoints_old/Transformer/policy_1"),
}

# Shrunken workload: the tests run on CPU, where compile time dominates
NR_ENVS = 2
UNROLL_LENGTH = 4
EPISODE_LENGTH = 20
NR_MAZES = 16
SEED = 0
CURR_COEFF = 0.5
TERRAIN_CURR_COEFF = 0.5


def load_config(name: str) -> dict:
    """Training config ``configs/<name>.yaml`` shrunk to a tiny, deterministic workload."""
    config = OmegaConf.to_container(OmegaConf.load(CONFIG_DIR / f"{name}.yaml"), resolve=True)
    config["env"]["episode_length"] = EPISODE_LENGTH
    training = config["training"]
    training.update(
        seed=SEED,
        debug=False,
        upload_to_wandb=False,
        nr_envs=NR_ENVS,
        batch_size=NR_ENVS * UNROLL_LENGTH,
        minibatch_size=NR_ENVS * UNROLL_LENGTH // 2,
        num_updates_per_batch=2,
        training_steps=NR_ENVS * UNROLL_LENGTH,
        nr_mazes=NR_MAZES,
    )
    config["eval"]["nr_eval_envs"] = NR_ENVS
    return config


def build(name: str, ckpt_dir: Path):
    """Setup dict, environment and PPOTrainer for the shrunken config ``name``."""
    from rememberwalk.training.ppo import PPOTrainer
    from rememberwalk.utils.generate_setup import generate_setup

    setup = generate_setup(load_config(name))
    env_setup = setup["env"]
    environment = env_setup["mode"](env_setup)
    trainer = PPOTrainer(environment, setup["training"], str(ckpt_dir), "golden", seed=SEED)
    return setup, environment, trainer


def run_deterministic_eval(trainer, params) -> dict:
    """Deterministic evaluation exactly as PPOTrainer.training_epoch_loop sets it up."""
    key = jax.random.PRNGKey(SEED)
    reset_keys = jax.random.split(key, trainer.nr_eval_envs)
    key, _ = jax.random.split(key)
    state = jax.vmap(trainer.jitted_reset_function, (None, None, 0))(CURR_COEFF, TERRAIN_CURR_COEFF, reset_keys)
    eval_dict = deepcopy(trainer.eval_dict)
    eval_dict["state"] = state
    eval_dict["parameters"] = params
    eval_dict["action"] = jnp.zeros((trainer.nr_eval_envs, trainer.actor_critic.action_size))
    if hasattr(trainer.actor_critic, "distribution_size"):
        eval_dict["predicted_distribution"] = jnp.zeros((trainer.nr_eval_envs, trainer.actor_critic.distribution_size))
        eval_dict["pred_delta_loc"] = jnp.zeros((trainer.nr_eval_envs, 3))
    metrics, episode_lengths, summed_rewards = trainer.evaluation_function(eval_dict, key)
    out = {f"metrics/{k}": v for k, v in metrics.items()}
    out["episode_lengths"] = episode_lengths
    out["summed_rewards"] = summed_rewards
    return to_numpy(out)


def run_experiment_eval(setup, trainer, params) -> dict:
    """Deterministic rollout through the experiment_callback, as experiments/data_creation.py runs it."""
    from rememberwalk.experiments.data_creation import evaluation_fn
    from rememberwalk.experiments.rollout import build_eval_dict

    key = jax.random.PRNGKey(SEED)
    reset_keys = jax.random.split(key, trainer.nr_eval_envs)
    key, _ = jax.random.split(key)
    state = jax.vmap(trainer.jitted_reset_function, (None, None, 0))(CURR_COEFF, TERRAIN_CURR_COEFF, reset_keys)
    eval_dict = build_eval_dict(setup["training"], state, trainer.nr_eval_envs, trainer.actor_critic.action_size)
    eval_dict["parameters"] = params
    environment = trainer.environment
    eval_dict, metrics, _ = evaluation_fn(eval_dict, key, environment, trainer.actor_critic, environment.step,
                                          setup["training"]["experiment_callback"])
    out = {f"metrics/{k}": v for k, v in metrics.items()}
    out["episode_lengths"] = eval_dict["episode_lengths"]
    out["summed_rewards"] = eval_dict["summed_rewards"]
    return to_numpy(out)


def run_training_epoch(trainer) -> dict:
    """One jitted training epoch (rollout + PPO update) from a fixed-seed reset."""
    key = jax.random.PRNGKey(SEED)
    reset_keys = jax.random.split(key, trainer.nr_envs)
    key, _ = jax.random.split(key)
    env_state = jax.vmap(trainer.jitted_reset_function, (None, None, 0))(CURR_COEFF, TERRAIN_CURR_COEFF, reset_keys)
    env_state, training_state, key, optimize_metrics, training_metrics, summed_reward, episode_length = (
        trainer.jitted_training_epoch(env_state, trainer.initial_training_state, key))
    out = {f"optimize/{k}": v for k, v in optimize_metrics.items()}
    out.update({f"rollout/{k}": v for k, v in training_metrics.items()})
    out["summed_reward"] = summed_reward
    out["episode_length"] = episode_length
    out["obs"] = env_state.obs
    out.update({f"params/{k}": v for k, v in checksums(training_state.params).items()})
    return to_numpy(out)


def flatten(tree) -> dict:
    """``{"policy_parameters/params/Dense_0/kernel": leaf, ...}`` for any pytree."""
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return {jax.tree_util.keystr(path, simple=True, separator="/"): leaf for path, leaf in leaves}


def param_signature(params) -> dict:
    """Shape, dtype and value checksums per parameter leaf (small enough to keep in git)."""
    return {
        name: {"shape": list(np.shape(leaf)), "dtype": str(np.asarray(leaf).dtype),
               **{k: float(v) for k, v in _leaf_checksum(leaf).items()}}
        for name, leaf in flatten(params).items()
    }


def checksums(params) -> dict:
    """``{"<leaf>/sum": ..., "<leaf>/abs_sum": ...}`` as arrays, for npz golden files."""
    return {f"{name}/{k}": v for name, leaf in flatten(params).items() for k, v in _leaf_checksum(leaf).items()}


def _leaf_checksum(leaf) -> dict:
    values = np.asarray(leaf, dtype=np.float64)
    return {"sum": values.sum(), "abs_sum": np.abs(values).sum()}


def to_numpy(tree: dict) -> dict:
    return {k: np.asarray(v) for k, v in tree.items()}


def checkpoint_available(run_path: str) -> bool:
    return (CKPT_DIR / run_path).is_dir()
