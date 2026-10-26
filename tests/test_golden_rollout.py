"""Fixed-seed environment rollouts and PPO updates produce the recorded outputs.

The deterministic evaluation covers reset, observation, domain randomization, reward, done,
curriculum, terrain and the eval callback. The training epoch additionally covers trajectory
collection, the loss function and the optimizer.
"""
import numpy as np
import pytest

from harness import CONFIGS, run_deterministic_eval, run_training_epoch

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
def test_training_epoch(name, build_cache, golden):
    _, _, trainer = build_cache(name)
    golden(f"training_epoch/{name}.npz", run_training_epoch(trainer))
