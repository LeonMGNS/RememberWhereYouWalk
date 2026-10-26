"""Pytest plumbing for the golden-output safety net.

    pytest tests --update-golden      # record goldens (do this on the pre-cleanup tag)
    pytest tests                      # compare against the recorded goldens
    pytest tests -m "not slow"        # only the fast structural checks

Tolerances for float comparisons can be loosened with GOLDEN_RTOL / GOLDEN_ATOL (default:
bit-exact). Everything else (key sets, shapes, dtypes) is always compared exactly.

The tests run on CPU by default (deterministic, no GPU memory limits, works on login nodes).
Goldens are backend specific: to test on GPU, set JAX_PLATFORMS=cuda and record separate
goldens. GPU scatter/reduction kernels are nondeterministic by default (~1e-4 drift between
two identical runs); the deterministic-ops flag below makes GPU runs bit-identical.
"""

import os

# Must be set before JAX initialises its backend
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
if "--xla_gpu_deterministic_ops" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true").strip()

import gc
import json
import sys
from pathlib import Path

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import harness  # noqa: E402

RTOL = float(os.environ.get("GOLDEN_RTOL", 0.0))
ATOL = float(os.environ.get("GOLDEN_ATOL", 0.0))


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        help="Write the current outputs as golden files instead of comparing.",
    )


def pytest_collection_modifyitems(items):
    """Group the tests by config, so build_cache builds every config only once."""

    def config_of(item):
        params = getattr(item, "callspec", None)
        return params.params.get("name", params.params.get("config", "")) if params else ""

    order = {item: i for i, item in enumerate(items)}
    items.sort(key=lambda item: (config_of(item), order[item]))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: compiles and runs the environment (about a minute per config)")
    config.addinivalue_line("markers", "checkpoint: needs the trained checkpoints in checkpoints/")


@pytest.fixture(scope="session")
def build_cache(tmp_path_factory):
    """``build_cache(name)`` -> (setup, environment, trainer) of the most recent config.

    Only one config is kept alive: holding all seven with their compiled training epochs runs
    a 32 GB GPU out of memory. Switching config frees the old one and JAX's compile caches.
    """
    root = tmp_path_factory.mktemp("ckpt")
    current = {}

    def build(name):
        if name not in current:
            current.clear()
            gc.collect()
            jax.clear_caches()
            current[name] = harness.build(name, root / name)
        return current[name]

    return build


@pytest.fixture
def golden(request):
    """Compare a result against its golden file, or record it with --update-golden."""
    update = request.config.getoption("--update-golden")

    def check(relative: str, result):
        path = harness.GOLDEN_DIR / relative
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps(result, indent=1, sort_keys=True))
            else:
                np.savez_compressed(path, **result)
            return
        if not path.exists():
            pytest.fail(f"Golden file {path} is missing, record it with --update-golden")
        if path.suffix == ".json":
            _compare_signature(json.loads(path.read_text()), result)
        else:
            with np.load(path) as expected:
                _compare_arrays(dict(expected), result)

    return check


def _compare_signature(expected: dict, actual: dict):
    """Parameter signatures: names, shapes and dtypes exactly, checksums within tolerance."""
    assert sorted(actual) == sorted(expected), _key_diff(expected, actual)
    failures = []
    for name, exp in expected.items():
        act = actual[name]
        if act["shape"] != exp["shape"] or act["dtype"] != exp["dtype"]:
            failures.append(f"{name}: {act['shape']} {act['dtype']} != golden {exp['shape']} {exp['dtype']}")
            continue
        for key in ("sum", "abs_sum"):
            if key in exp and not np.isclose(act[key], exp[key], rtol=RTOL, atol=ATOL):
                failures.append(f"{name}/{key}: {act[key]!r} != golden {exp[key]!r}")
    assert not failures, "\n".join(failures)


def _compare_arrays(expected: dict, actual: dict):
    assert sorted(actual) == sorted(expected), _key_diff(expected, actual)
    failures = []
    for name, exp in expected.items():
        act = np.asarray(actual[name])
        if act.shape != exp.shape:
            failures.append(f"{name}: shape {act.shape} != golden {exp.shape}")
        elif not np.allclose(act, exp, rtol=RTOL, atol=ATOL, equal_nan=True):
            diff = np.nanmax(np.abs(act.astype(np.float64) - exp.astype(np.float64)))
            failures.append(f"{name}: max abs diff {diff:.3e}")
    assert not failures, "\n".join(failures)


def _key_diff(expected, actual) -> str:
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    return f"keys differ from golden. missing: {missing}  new: {extra}"
