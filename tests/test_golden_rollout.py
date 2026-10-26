"""Fixed-seed environment rollouts and PPO updates produce the recorded outputs.

The deterministic evaluation covers reset, observation, domain randomization, reward, done,
curriculum, terrain and the eval callback. The training epoch additionally covers trajectory
collection, the loss function and the optimizer.
"""

import numpy as np
import pytest
from harness import (
    CONFIGS,
    run_deterministic_eval,
    run_experiment_eval,
    run_training_epoch,
    run_visualizer_steps,
)

pytestmark = pytest.mark.slow


def test_rollout_is_repeatable(build_cache):
    """Precondition for exact goldens: two identical runs in one process match bit for bit."""
    _, _, trainer = build_cache("lsmm")
    params = trainer.initial_training_state.params
    first, second = run_deterministic_eval(trainer, params), run_deterministic_eval(trainer, params)
    differing = [k for k in first if not np.array_equal(first[k], second[k], equal_nan=True)]
    assert not differing, f"nondeterministic outputs (is --xla_gpu_deterministic_ops set?): {differing}"


@pytest.mark.parametrize("name", CONFIGS)
def test_deterministic_eval(name, build_cache, golden):
    _, _, trainer = build_cache(name)
    golden(f"eval/{name}.npz", run_deterministic_eval(trainer, trainer.initial_training_state.params))


@pytest.mark.parametrize("name", CONFIGS)
def test_experiment_eval(name, build_cache, golden):
    """The experiment_callback rollout that writes the paper's eval_data.pkl."""
    setup, _, trainer = build_cache(name)
    golden(
        f"experiment_eval/{name}.npz",
        run_experiment_eval(setup, trainer, trainer.initial_training_state.params),
    )


@pytest.mark.parametrize("name", CONFIGS)
def test_visualizer_callback(name, build_cache):
    """scripts/visualize.py's single-env loop runs for every approach.

    No golden: the viewer is interactive, only the callback contract is checked.
    """
    setup, _, trainer = build_cache(name)
    visualizer_dict = run_visualizer_steps(setup, trainer)
    action = np.asarray(visualizer_dict["action"])
    assert action.shape == (trainer.actor_critic.action_size,)
    assert np.all(np.isfinite(action))


@pytest.mark.parametrize("name", CONFIGS)
def test_training_epoch(name, build_cache, golden):
    _, _, trainer = build_cache(name)
    golden(f"training_epoch/{name}.npz", run_training_epoch(trainer))
