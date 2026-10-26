"""Restoring, locating and exporting orbax checkpoints."""

import os
import time

import jax
import orbax.checkpoint as ocp
from flax import serialization
from flax.training import orbax_utils

from rememberwalk.utils.paths import CKPT_DIR


def restore_ckpt(ckpt_dir, actor_critic, visualizer_config, put_cpu=True, new_keys_to_ignore=()):
    # Get the state dictionary of your NEW structure
    """Restores the parameters of checkpoint ``visualizer_config["policy"]`` (relative to ``ckpt_dir``).

    The checkpoint is merged into the freshly initialised TrainingState, so state fields added after the
    checkpoint was written keep their initial values (``new_keys_to_ignore`` lists such top-level keys).

    Returns:
        The ``params`` dict (on the CPU if ``put_cpu``), or None if the checkpoint does not exist.
    """
    training_state = visualizer_config["training_state"]
    full_state_dict = serialization.to_state_dict(training_state)

    # 1. Create a copy and remove the newly added keys.
    # This perfectly matches the old checkpoint structure on disk.
    old_state_dict = dict(full_state_dict)

    # Put the exact top-level keys you added here.
    # Based on your error, 'last_error_vector' is one of them.

    for key in new_keys_to_ignore:
        if key in old_state_dict:
            del old_state_dict[key]

    # 2. Generate restore_args from the FILTERED dictionary.
    # This grabs the CPU sharding from your initialized arrays, fixing the Sharding error,
    # while ignoring the new keys so you don't get the Mismatch error.
    restore_args = orbax_utils.restore_args_from_target(old_state_dict)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    checkpoint_manager = ocp.CheckpointManager(ckpt_dir, checkpointer, options)

    # 3. Restore the partial state from disk
    restored_partial = checkpoint_manager.restore(
        visualizer_config["policy"], restore_kwargs={"restore_args": restore_args}
    )

    if restored_partial is None:
        print("Given checkpoint was not found")
        return None

    # 4. Helper to merge the old checkpoint back into your new full structure
    def recursive_merge(target, source):
        """Overwrites the values of ``target`` with those of ``source``, recursing into dicts."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                recursive_merge(target[key], value)
            else:
                # Overwrite the initialized value with the saved checkpoint value
                target[key] = value
        return target

    # 5. Merge! The new keys (like last_error_vector) keep their defaults,
    # while everything else gets updated with the checkpoint weights.
    restored = recursive_merge(full_state_dict, restored_partial)

    if put_cpu:
        return to_cpu(restored["params"])
    else:
        return restored


def to_cpu(tree):
    """Puts every jax array of a pytree on the CPU."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(x, jax.devices("cpu")[0]) if isinstance(x, jax.Array) else x, tree
    )


def get_dir_and_run_for_checkpointing():
    # Base checkpoints directory
    """Creates the checkpoint directory of a run and returns (directory, run id).

    checkpoints/sweep_<id>/run_<run id> inside a wandb sweep, checkpoints/non_sweep/run_<run id> for a
    wandb run, checkpoints/non_sweep/local_<unix time> otherwise.
    """
    base_checkpoint_dir = CKPT_DIR

    # Extract W&B environment variables
    sweep_id = os.environ.get("WANDB_SWEEP_ID")
    run_id = os.environ.get("WANDB_RUN_ID")

    if sweep_id:
        # 1. We are inside a Sweep
        # Group everything for this sweep inside its own parent folder
        target_dir = base_checkpoint_dir / f"sweep_{sweep_id}"

        # Since the parent folder specifies the sweep, we just use the run_id here
        run_identifier = run_id if run_id else str(int(time.time()))
        folder_name = f"run_{run_identifier}"

    else:
        # 2. We are NOT inside a Sweep
        # Route it to a dedicated non-sweep folder to keep things tidy
        target_dir = base_checkpoint_dir / "non_sweep"

        if run_id:
            # It's a standard W&B run (tracked, but not part of a sweep array)
            run_identifier = run_id
            folder_name = f"run_{run_identifier}"
        else:
            # 3. It's a completely local, untracked run
            run_identifier = str(int(time.time()))
            folder_name = f"local_{run_identifier}"

    # Combine the path: e.g., checkpoints/sweep_xyz/run_abc
    full_path = target_dir / folder_name

    # mkdir with parents=True creates all necessary intermediate folders automatically
    # exist_ok=True prevents crashes if multiple agents try to create the 'sweep_xyz' parent folder at the
    # exact same time
    full_path.mkdir(parents=True, exist_ok=True)

    return str(full_path), run_identifier


def extract_and_save_actor(full_state, export_dir):
    # 3. Extract strictly the Actor's parameters
    # (Adjust the keys based on your specific PyTree structure)
    """Saves only the parameters needed to run the LSMM policy (deployment) to ``export_dir``."""
    actor_params = {
        "policy_parameters": full_state["policy_parameters"],
        "cross_entropy_params": full_state["cross_entropy_params"],
        "height_encoder_parameters": full_state["height_encoder_parameters"],
        "f_parameters": full_state["f_parameters"],
        "g_parameters": full_state["g_parameters"],
        "matrix_memory_encoder_parameters": full_state["matrix_memory_encoder_parameters"],
        "linear_velocity_parameters": full_state["linear_velocity_parameters"],
    }

    # 4. Save the lightweight, Actor-only checkpoint
    export_checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    export_checkpointer.save(export_dir, actor_params, force=True)

    print(f"Lean Actor checkpoint saved to: {export_dir}")
