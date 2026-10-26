"""Domain randomisation applied by the env (``imports.randomize_state_action_fn``,
``reset_randomize_state_fn``).
"""

import jax
import jax.numpy as jnp

from rememberwalk.env.SimpleMazeEnv import SimpleMazeEnv
from rememberwalk.states.MazeState import MazeState


def randomize_state_action(
    state: MazeState, action: jnp.ndarray, env: SimpleMazeEnv, key: jnp.ndarray
) -> tuple[MazeState, jnp.ndarray, jnp.ndarray]:
    """Per-step domain randomisation of one env: (state, action, key) -> (state, action, key).

    With probability ``randomization_prob`` each: re-sample the MJX model (masses, friction, ...), push the
    base, re-sample the PD gains / motor strength. The action delay is applied every step.
    """
    curr_randomize_prob = env.dr_parameters["randomization_prob"]
    # Model Randomization
    model, key = jax.lax.cond(
        jax.random.uniform(key) < curr_randomize_prob,
        lambda: env.DomainModel.randomize_model(
            state.model,
            key,
            state.curr_coeff,
        ),
        lambda: (state.model, key),
    )
    _, key = jax.random.split(key)
    # Perturbation (push) randomization
    pipeline_state, key = jax.lax.cond(
        jax.random.uniform(key) < curr_randomize_prob,
        lambda: env.DomainPerturbation.sample(state.pipeline_state, key, state.curr_coeff),
        lambda: (state.pipeline_state, key),
    )
    _, key = jax.random.split(key)
    # Delay Action
    action, action_delay, key = env.DelayAction.delay_action_mixed(state.action_delay, action, key)

    # Control Randomization
    _, key = jax.random.split(key)
    state, key = jax.lax.cond(
        jax.random.uniform(key) < curr_randomize_prob,
        lambda: env.control_randomization.sample(state, key, state.curr_coeff),
        lambda: (state, key),
    )
    state = state.replace(
        model=model,
        pipeline_state=pipeline_state,
        action_delay=action_delay,
    )
    return state, action, key


def reset_randomize_state(
    state: MazeState, env: SimpleMazeEnv, key: jnp.ndarray
) -> tuple[MazeState, jnp.ndarray]:
    """Randomisation at episode start: the initial pose (80 % of resets) and the height map noise."""

    pipeline_state, key = jax.lax.cond(
        jax.random.uniform(key) < 0.8,
        lambda: env.InitStateRand.init_state_rand(state, env, key),
        lambda: (state.pipeline_state, key),
    )

    _, key = jax.random.split(key)
    state = state.replace(pipeline_state=pipeline_state)
    state = env.terrain_observation.randomize_height_map_parameters(state, key)
    _, key = jax.random.split(key)

    return state, key
