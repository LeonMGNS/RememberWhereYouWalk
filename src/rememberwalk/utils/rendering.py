"""Offscreen rendering of an env state for the viewer videos."""
import mujoco
import numpy as np
from mujoco import mjx


def render(eval_state,visualizer_config,env,renderer):
        data = mjx.get_data(env.model, eval_state.pipeline_state)
        # 1. Convert JAX array to NumPy and flatten it
        hfield_np = np.array(eval_state.model.hfield_data).flatten()

        # 2. Copy data IN-PLACE into the existing model buffer
        # This ensures the memory address remains the same for the renderer
        env.model.hfield_data[:] = hfield_np

        # 3. Explicitly upload to the renderer's GPU context
        mujoco.mjr_uploadHField(env.model, renderer._mjr_context, 0)

        # 4. Standard render sequence
        mujoco.mj_camlight(env.model, data)
        renderer.update_scene(data, 0)
        return renderer.render()


def generate_unique_path(dir_path, run):
    run = run.replace("/", "_")
    base_path = f"{dir_path}/videos/{run}"
    extension = ".mp4"
    output_path = f"{base_path}{extension}"
    return output_path
