"""Deterministic evaluation rollouts of every run in a suite.

For each approach and run, the policy is rolled out on the shared eval mazes and
``<output_dir>/data/<approach>/run_<i>/eval_data.pkl`` is written with the per-step
metrics of the approach's ``experiment_callback``. All figures are made from these files.
"""
import json
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from rememberwalk.experiments.rollout import build_environment, build_eval_dict, load_policy, reset_environments
from rememberwalk.experiments.suite import Suite, environment_key, experiment_config


def run_deterministic_evaluation(eval_dict: dict, key, env, actor_critic, step_fn, callback_fn):

    def eval_step(carry, _):

        def mask_single_env(is_terminated, single_env_metrics):
            return jax.tree_util.tree_map(
                lambda m: jnp.where(is_terminated, jnp.zeros_like(m), m),
                single_env_metrics
            )
        eval_dict, key = carry
        eval_dict, metrics = callback_fn(actor_critic, eval_dict, env)
        nr_envs = eval_dict["state"].done.shape[0]
        state_keys = jax.random.split(key, nr_envs)
        next_state, state_keys = jax.vmap(step_fn)(eval_dict["state"], eval_dict["action"], state_keys)
        eval_dict["state"] = next_state
        eval_dict["episode_lengths"] = jnp.where(eval_dict["envs_already_terminated"], eval_dict["episode_lengths"], next_state.n_steps)
        masked_metrics = jax.vmap(mask_single_env)(eval_dict["envs_already_terminated"], metrics)
        just_finished = ((eval_dict["state"].done | (eval_dict["state"].n_steps >= env.episode_length - 1)) & (~eval_dict["envs_already_terminated"].astype(jnp.bool_)))
        eval_dict["summed_rewards"] = jnp.where(just_finished, eval_dict["state"].summed_reward, eval_dict["summed_rewards"])
        eval_dict["envs_already_terminated"] = jnp.where(eval_dict["state"].done, jnp.ones_like(eval_dict["envs_already_terminated"]), eval_dict["envs_already_terminated"])
        return (eval_dict, state_keys[-1]), masked_metrics

    (eval_dict, key), metrics = jax.lax.scan(eval_step, (eval_dict, key), None, length=env.episode_length)
    return eval_dict, metrics, key


evaluation_fn = jax.jit(run_deterministic_evaluation, static_argnums=(2, 3, 4, 5))


def _created_from(out_dir):
    """Checkpoint an existing eval_data.pkl was made from, so a changed suite entry is re-run."""
    try:
        with open(out_dir / "run.json") as f:
            return json.load(f)["checkpoint"]
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


def create_data(suite: Suite, force: bool = False):
    for approach in suite.approaches:
        print(f"=== Data creation: {approach.name} ({len(approach.runs)} runs)")
        if approach.imported:
            # No checkpoints to roll out: the data was copied in, it only has to be there
            for i in range(len(approach.runs)):
                out_file = suite.data_dir(approach, i) / "eval_data.pkl"
                if not out_file.is_file():
                    raise FileNotFoundError(f"'{approach.name}' is imported, but {out_file} is missing")
                print(f"   run {i + 1}: imported {out_file}")
            continue
        environment, setup, env_state, env_keys, built_for = None, None, None, None, None
        for i, run_path in enumerate(approach.runs):
            out_dir = suite.data_dir(approach, i)
            out_file = out_dir / "eval_data.pkl"
            if out_file.is_file() and not force and _created_from(out_dir) == run_path:
                print(f"   run {i + 1}: {out_file} exists, skipping (use --force to recreate)")
                continue

            config = experiment_config(suite, approach, run_path)
            key = environment_key(config)
            if key != built_for:
                # Seeds of one approach share the compiled environment and rollout
                environment, setup = build_environment(config)
                env_state, env_keys = reset_environments(environment, suite.nr_envs, suite.randomization_key,
                                                         suite.curriculum_level, suite.terrain_curriculum_level)
                built_for = key
            training_setup = setup["training"]

            eval_dict = build_eval_dict(training_setup, env_state, suite.nr_envs, environment.action_size)
            eval_dict["parameters"] = load_policy(setup, run_path)
            eval_key, _ = jax.random.split(env_keys[-1])

            print(f"   run {i + 1}: rolling out {run_path}")
            eval_dict, metrics, _ = evaluation_fn(
                eval_dict, eval_key, environment, training_setup["actor_critic"], environment.step, training_setup["experiment_callback"]
            )
            save_dict = {
                "policy": run_path,
                "episode_lengths": np.asarray(eval_dict["episode_lengths"]),
                "summed_rewards": np.asarray(eval_dict["summed_rewards"]),
                "metrics": jax.tree.map(lambda x: np.asarray(x), metrics),
            }
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_file, "wb") as f:
                pickle.dump(save_dict, f)
            with open(out_dir / "run.json", "w") as f:
                json.dump({"approach": approach.name, "checkpoint": run_path}, f, indent=2)
            print(f"   run {i + 1}: saved {out_file}")
