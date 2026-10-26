"""The trained paper checkpoints still load and the restored policies behave as recorded.

Skipped when checkpoints/experiment_checkpoints_old is not present (e.g. on the cluster).
"""
import jax
import pytest

from harness import CHECKPOINTS, checkpoint_available, flatten, param_signature, run_deterministic_eval

pytestmark = pytest.mark.checkpoint

CASES = [pytest.param(config, run, id=approach) for approach, (config, run) in CHECKPOINTS.items()]


def _restore(build_cache, config, run):
    if not checkpoint_available(run):
        pytest.skip(f"checkpoint {run} not available")
    from rememberwalk.experiments.rollout import load_policy

    setup, _, trainer = build_cache(config)
    return setup, trainer, load_policy(setup, run)


@pytest.mark.parametrize("config, run", CASES)
def test_restore(config, run, build_cache, golden):
    setup, _, params = _restore(build_cache, config, run)
    init_params = setup["training"]["actor_critic"].params
    restored, initial = flatten(params), flatten(init_params)
    assert sorted(restored) == sorted(initial), "restored parameter tree differs from the network"
    for name, leaf in initial.items():
        assert restored[name].shape == leaf.shape, f"{name}: {restored[name].shape} != {leaf.shape}"
    golden(f"checkpoint/{config}.json", param_signature(params))


@pytest.mark.slow
@pytest.mark.parametrize("config, run", CASES)
def test_restored_policy_rollout(config, run, build_cache, golden):
    _, trainer, params = _restore(build_cache, config, run)
    params = jax.tree_util.tree_map(jax.numpy.asarray, params)
    golden(f"checkpoint_eval/{config}.npz", run_deterministic_eval(trainer, params))
