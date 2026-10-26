"""The networks of every config keep their parameter tree, shapes and initial values.

Catches renamed Params fields, changed layer structure (breaks loading the paper checkpoints)
and changed PRNG key usage during initialisation (breaks reproducing the paper runs).
"""
import jax
import pytest

from harness import CONFIGS, flatten, param_signature


@pytest.mark.parametrize("name", CONFIGS)
def test_initial_params(name, build_cache, golden):
    setup, _, _ = build_cache(name)
    golden(f"param_tree/{name}.json", param_signature(setup["training"]["actor_critic"].params))


@pytest.mark.parametrize("name", CONFIGS)
def test_training_state_layout(name, build_cache, golden):
    """Shapes of the full TrainingState (optimizer state, memory buffers), no values."""
    setup, _, _ = build_cache(name)
    layout = {path: {"shape": list(leaf.shape), "dtype": str(leaf.dtype)}
              for path, leaf in flatten(setup["training"]["training_state"]).items()}
    golden(f"training_state/{name}.json", layout)


@pytest.mark.parametrize("name", CONFIGS)
def test_optimizer_labels_match_params(name, build_cache):
    """optax.multi_transform needs one label per Params field."""
    actor_critic = build_cache(name)[0]["training"]["actor_critic"]
    params_fields = jax.tree_util.tree_structure(actor_critic.params, is_leaf=lambda x: isinstance(x, dict))
    labels_fields = jax.tree_util.tree_structure(actor_critic.labels)
    assert params_fields.num_leaves == labels_fields.num_leaves
