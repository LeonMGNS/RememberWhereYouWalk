"""Shared evaluation rollout setup.

Data creation and the live recording both build their environment, reset keys and
eval dict through these functions. Same eval-mode maze generator, same
``PRNGKey(randomization_key)`` split over ``nr_envs``: env ``i`` is therefore the same
maze in every approach, which the showcase figure and the live video rely on.
"""

import jax
import jax.numpy as jnp

from rememberwalk.terrain.MazeStaticTerrainGenerator import MazeStaticGenerator
from rememberwalk.utils.checkpointing import restore_ckpt
from rememberwalk.utils.generate_setup import generate_setup
from rememberwalk.utils.paths import CKPT_DIR


def build_environment(config: dict):
    """Environment on the held-out eval mazes, plus the training and visualizer setups."""
    setup = generate_setup(config)
    env_setup = setup["env"]
    terrain_handler = config.get("imports", {}).get("terrain_handler", "")
    if terrain_handler.split(".")[-1] == "MazeStaticGenerator":
        env_setup["imports"]["TerrainGenerator"] = MazeStaticGenerator(
            env_setup["terrain_parameters"], from_file=True, eval_mode=True
        )
    environment = env_setup["mode"](env_setup)
    return environment, setup


def reset_environments(
    environment, nr_envs: int, randomization_key: int, curr_coeff: float, terrain_curr_coeff: float
):
    """Resets ``nr_envs`` envs with keys split from ``PRNGKey(randomization_key)``: env i is the same maze
    for every approach.

    Returns:
        (batched state, env keys).
    """
    env_keys = jax.random.split(jax.random.PRNGKey(randomization_key), nr_envs)
    state = jax.vmap(environment.reset, (None, None, 0))(curr_coeff, terrain_curr_coeff, env_keys)
    return state, env_keys


def build_eval_dict(training_setup: dict, state, nr_envs: int, action_size: int) -> dict:
    """Eval dict of the experiment rollout: the approach's initial memory, the state and zeroed counters."""
    eval_dict = dict(training_setup["eval_dict"])
    eval_dict["state"] = state
    eval_dict["envs_already_terminated"] = jnp.zeros((nr_envs,), dtype=jnp.int32)
    eval_dict["episode_lengths"] = jnp.zeros((nr_envs,), dtype=jnp.int32)
    eval_dict["summed_rewards"] = jnp.zeros((nr_envs,), dtype=jnp.float32)
    eval_dict["metrics"] = state.metrics
    eval_dict["action"] = jnp.zeros((nr_envs, action_size))
    return eval_dict


def load_policy(setup: dict, run_path: str):
    """Restore the policy parameters of ``run_path`` (relative to CKPT_DIR) onto the default device."""
    visualizer_setup = dict(setup["visualizer"])
    visualizer_setup["policy"] = run_path
    params = restore_ckpt(str(CKPT_DIR), setup["training"]["actor_critic"], visualizer_setup)
    if params is None:
        raise FileNotFoundError(f"Could not restore checkpoint {CKPT_DIR / run_path}")
    device = jax.devices()[0]
    params = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, device) if isinstance(x, jax.Array) else x, params
    )
    return visualizer_setup["params_type"](**params)
