
import os
from pathlib import Path
import fnmatch
import yaml
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
import optax
from rememberwalk.actor_critic.Transformer import TrainingState as TrainingStateTransformer
import jax.numpy as jnp
import jax
from flax.training import orbax_utils
from flax import serialization
from mujoco import mjx
import flax
import mujoco
import re
import time


def to_gpu(tree):
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cuda")[0]) if isinstance(x, jax.Array) else x, tree)

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




# def restore_ckpt(ckpt_dir,actor_critic,visualizer_config,put_cpu=True):
#     options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
#     checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
#     checkpoint_manager = ocp.CheckpointManager(ckpt_dir,checkpointer, options)
#     cpu_device = jax.devices("cpu")[0] 
#     cpu_sharding = jax.sharding.SingleDeviceSharding(cpu_device)
#     restore_args =  ocp.ArrayRestoreArgs( sharding=cpu_sharding, # explicit CPU sharding 
#                                                     global_shape=None # optional, Orbax infers if not given # no RestoreType anymore 
#                                                     )
#     actor_optimizer = optax.chain(
#         optax.clip_by_global_norm(1.0),  # Clip gradients with norm > 1.0
#         optax.adam(learning_rate=4e-5, eps=1e-5)    
#         )
#     critic_optimizer = optax.chain(
#             optax.clip_by_global_norm(1.0),  # Clip gradients with norm > 1.0
#             optax.adam(learning_rate=1e-5, eps=1e-5)    
#             )
#     labels = visualizer_config["labels"]
#     optimizer = optax.multi_transform(
#             transforms={'actor':actor_optimizer,'critic':critic_optimizer},
#             param_labels=labels,
#     )
#     training_state = visualizer_config["training_state"]
    
#     state_dict = serialization.to_state_dict(training_state)
#     print(state_dict.keys())

#     restore_args = orbax_utils.restore_args_from_target(state_dict)
#     options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
#     checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
#     checkpoint_manager = ocp.CheckpointManager(ckpt_dir, checkpointer, options)
#     restored = checkpoint_manager.restore(visualizer_config["policy"],restore_kwargs={'restore_args': restore_args})
#     if restored is None:
#         print("Given checkpoint was not found")
#     elif put_cpu:
#         return to_cpu(restored["params"])
#     elif not put_cpu:
#         return restored
    

def restore_ckpt(ckpt_dir, actor_critic, visualizer_config, put_cpu=True,new_keys_to_ignore=[]):
    # Get the state dictionary of your NEW structure 
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
        visualizer_config["policy"], 
        restore_kwargs={'restore_args': restore_args}
    )
    
    if restored_partial is None:
        print("Given checkpoint was not found")
        return None

    # 4. Helper to merge the old checkpoint back into your new full structure
    def recursive_merge(target, source):
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
        return jax.tree_util.tree_map(lambda x: jax.device_put(x, jax.devices("cpu")[0]) if isinstance(x, jax.Array) else x, tree)

# Directory settings
current_file_path = Path(__file__).resolve()
dir_path = str(current_file_path.parent.parent.parent.parent)
ckpt_dir = f"{dir_path}/checkpoints/"
save_ckpt_dir = f"{dir_path}/converged_checkpoints"

# Wandb config filter

allowed_types = (int, float, str, bool)

def filter_nested(obj):
    # Keep simple values
    if isinstance(obj, allowed_types):
        return obj

    # Recurse into dictionaries
    if isinstance(obj, dict):
        return {
            k: v_filtered
            for k, v in obj.items()
            if (v_filtered := filter_nested(v)) is not None
        }

    # Recurse into lists/tuples
    if isinstance(obj, (list, tuple)):
        result = [filter_nested(item) for item in obj]
        return [item for item in result if item is not None]

    # Everything else is removed
    return None



def generate_unique_path(dir_path, run):
    run = run.replace("/", "_")
    base_path = f"{dir_path}/videos/{run}"
    extension = ".mp4"
    output_path = f"{base_path}{extension}"
    return output_path



def get_dir_and_run_for_checkpointing():
    # Base checkpoints directory
    base_checkpoint_dir = Path(dir_path) / "checkpoints"
    
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
    # exist_ok=True prevents crashes if multiple agents try to create the 'sweep_xyz' parent folder at the exact same time
    full_path.mkdir(parents=True, exist_ok=True)
    
    return str(full_path), run_identifier

class EmptyLogger:

    def __init__(self):
        pass
        
    def log(self,metrics,step):
        return


def compute_bce_loss(predicted_distribution: jnp.ndarray,true_distribution):
        loss = optax.losses.sigmoid_binary_cross_entropy(predicted_distribution,true_distribution)
        loss = jnp.mean(loss)
        return loss

def unflatten_edge_vector(edge_vec, R, C):
    H = (R + 1) * C
    V = R * (C + 1)

    horizontal = edge_vec[:H].reshape(R+1, C)
    vertical   = edge_vec[H:H+V].reshape(R, C+1)
    return horizontal, vertical


def extract_and_save_actor(full_state, export_dir):
    
    # 3. Extract strictly the Actor's parameters
    # (Adjust the keys based on your specific PyTree structure)
    actor_params = {
        'policy_parameters': full_state['policy_parameters'],
        'cross_entropy_params': full_state['cross_entropy_params'],
        'height_encoder_parameters': full_state['height_encoder_parameters'],
        'f_parameters': full_state["f_parameters"],
        'g_parameters': full_state['g_parameters'],
        'matrix_memory_encoder_parameters': full_state["matrix_memory_encoder_parameters"],
        'linear_velocity_parameters': full_state["linear_velocity_parameters"]
    }
    
    # 4. Save the lightweight, Actor-only checkpoint
    export_checkpointer = ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    export_checkpointer.save(export_dir, actor_params, force=True)
    
    print(f"Lean Actor checkpoint saved to: {export_dir}")

def edges_to_wall_tensor(horizontal, vertical, R, C):
    # Create empty wall tensor
    # 0=N, 1=S, 2=W, 3=E
    wall = jnp.zeros((4, R, C), dtype=bool)

    # North walls
    wall = wall.at[0, :, :].set(horizontal[:-1, :])

    # South walls
    wall = wall.at[1, :, :].set(horizontal[1:, :])

    # West walls
    wall = wall.at[2, :, :].set(vertical[:, :-1])

    # East walls
    wall = wall.at[3, :, :].set(vertical[:, 1:])

    return wall


def get_maze_figure(fig, maze_pred_flat, maze_true_flat, R, C,pred_delta_pos,true_delta_pos):
    """Updates the figure and returns it as a NumPy RGB array."""
    # --- Convert flattened edge vectors into wall tensors ---
    # (Assuming jnp is available in your namespace as per your code)
    
    maze_pred = maze_pred_flat.reshape(R+1, C+1, 2)
    maze_true = maze_true_flat.reshape(R+1, C+1, 2)

    def draw_maze(ax, maze, title):
        maze = np.array(maze) # Convert to numpy for matplotlib
        H, W, _ = maze.shape
        H, W = H-1, W-1
        # Replicating your flip logic
        maze = np.flip(maze, axis=2)

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)

        north_outer_wall = maze[0, :, 0].flatten()
        west_outer_wall = maze[:, 0, 1].flatten()
        
        for r in range(H):
            for c in range(W):
                r_idx, c_idx = r+1, c+1
                S, E = maze[r_idx, c_idx]
                x0, x1 = c, c+1
                y0, y1 = r, r+1

                # 1-E/1-S logic for wall visibility
                ax.plot([x1, x1], [y0, y1+1], color=str(1-float(E)), linewidth=3)
                ax.plot([x0, x1+1], [y1, y1], color=str(1-float(S)), linewidth=3)

        for i in range(W):
            ax.plot([i, i+1], [0, 0], color=str(1-float(north_outer_wall[i+1])), linewidth=3)
            ax.plot([0, 0], [i, i+1], color=str(1-float(west_outer_wall[i+1])), linewidth=3)

        ax.set_xticks([])
        ax.set_yticks([])

    # Setup Axes
    if len(fig.axes) < 2:
        fig.clear()
        ax_pred = fig.add_subplot(3, 1, 1)
        ax_true = fig.add_subplot(3, 1, 2)
        ax_delta_pos = fig.add_subplot(3, 1, 3)
    else:
        ax_pred, ax_true,ax_delta_pos = fig.axes[:3]

    # Draw the mazes
    # (Note: make sure compute_bce_loss is defined in your scope)
    bce_loss = float(compute_bce_loss(maze_pred_flat, maze_true_flat))
    draw_maze(ax_pred, flax.linen.sigmoid(maze_pred), f"Predicted Maze (BCE: {bce_loss:.3f})")
    draw_maze(ax_true, maze_true, "True Maze")

    # Plot delta pos
    ax_delta_pos.clear()
    ax_delta_pos.quiver(true_delta_pos[1], -1*true_delta_pos[0], 
                    np.cos(true_delta_pos[2]-jnp.pi/2), np.sin(true_delta_pos[2]-jnp.pi/2), 
                    color='blue', label='True Delta', units='xy', scale=2.0, width=0.1)
    ax_delta_pos.quiver(pred_delta_pos[0,1], -1*pred_delta_pos[0,0],  
                    np.cos(pred_delta_pos[0,2]-jnp.pi/2), np.sin(pred_delta_pos[0,2]-jnp.pi/2), 
                    color='red', label='Pred Delta', units='xy', scale=2.0, width=0.1)
    
    ax_delta_pos.set_xlim([-5,5])
    ax_delta_pos.set_ylim([-5,5])
    ax_delta_pos.set_aspect('equal', adjustable='box')


    # --- CONVERSION TO NUMPY ARRAY ---
    fig.canvas.draw()
    
    # Grab the buffer from the renderer
    # buffer_rgba is standard for modern Matplotlib
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba_buffer.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Drop the Alpha channel to get RGB (standard for MuJoCo/Videos)
    return img[:, :, :3]

def get_maze_figure_matrix_memory(fig, maze_pred_flat, maze_true_flat, R, C,pred_delta_pos,true_delta_pos,matrix_memory):
    """Updates the figure and returns it as a NumPy RGB array."""
    # --- Convert flattened edge vectors into wall tensors ---
    # (Assuming jnp is available in your namespace as per your code)
    
    maze_pred = maze_pred_flat.reshape(R+1, C+1, 2)
    maze_true = maze_true_flat.reshape(R+1, C+1, 2)

    def draw_maze(ax, maze, title):
        maze = np.array(maze) # Convert to numpy for matplotlib
        H, W, _ = maze.shape
        H, W = H-1, W-1
        # Replicating your flip logic
        maze = np.flip(maze, axis=2)

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10)

        north_outer_wall = maze[0, :, 0].flatten()
        west_outer_wall = maze[:, 0, 1].flatten()
        
        for r in range(H):
            for c in range(W):
                r_idx, c_idx = r+1, c+1
                S, E = maze[r_idx, c_idx]
                x0, x1 = c, c+1
                y0, y1 = r, r+1

                # 1-E/1-S logic for wall visibility
                ax.plot([x1, x1], [y0, y1+1], color=str(1-float(E)), linewidth=3)
                ax.plot([x0, x1+1], [y1, y1], color=str(1-float(S)), linewidth=3)

        for i in range(W):
            ax.plot([i, i+1], [0, 0], color=str(1-float(north_outer_wall[i+1])), linewidth=3)
            ax.plot([0, 0], [i, i+1], color=str(1-float(west_outer_wall[i+1])), linewidth=3)

        ax.set_xticks([])
        ax.set_yticks([])

    # Setup Axes
    fig.clear()
    ax_pred = fig.add_subplot(4, 1, 1)
    ax_true = fig.add_subplot(4,1, 2)
    ax_delta_pos = fig.add_subplot(4, 1, 3)
    ax_matrix_memory_img = fig.add_subplot(4, 1, 4)

    # Draw the mazes
    # (Note: make sure compute_bce_loss is defined in your scope)
    bce_loss = float(compute_bce_loss(maze_pred_flat, maze_true_flat))
    draw_maze(ax_pred, flax.linen.sigmoid(maze_pred), f"Predicted Maze (BCE: {bce_loss:.3f})")
    draw_maze(ax_true, maze_true, "True Maze")

    # Plot delta pos
    ax_delta_pos.clear()
    ax_delta_pos.quiver(true_delta_pos[1], -1*true_delta_pos[0], 
                    np.cos(true_delta_pos[2]-jnp.pi/2), np.sin(true_delta_pos[2]-jnp.pi/2), 
                    color='blue', label='True Delta', units='xy', scale=2.0, width=0.1)
    ax_delta_pos.quiver(pred_delta_pos[0,1], -1*pred_delta_pos[0,0],  
                    np.cos(pred_delta_pos[0,2]-jnp.pi/2), np.sin(pred_delta_pos[0,2]-jnp.pi/2), 
                    color='red', label='Pred Delta', units='xy', scale=2.0, width=0.1)
    
    ax_delta_pos.set_xlim([-5,5])
    ax_delta_pos.set_ylim([-5,5])
    ax_delta_pos.set_aspect('equal', adjustable='box')
    #matrix_memory = (matrix_memory/jnp.max(matrix_memory))
    ax_matrix_memory_img.imshow(matrix_memory.squeeze(),cmap="viridis", 
    vmin=0.0, 
    vmax=1.0)
    # --- CONVERSION TO NUMPY ARRAY ---
    fig.canvas.draw()
    
    # Grab the buffer from the renderer
    # buffer_rgba is standard for modern Matplotlib
    rgba_buffer = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = rgba_buffer.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    
    # Drop the Alpha channel to get RGB (standard for MuJoCo/Videos)
    return img[:, :, :3]

def update_maze_figure(fig, maze_pred_flat, maze_true_flat, R, C):
    # --- Convert flattened edge vectors into wall tensors ---
    maze_pred = maze_pred_flat.reshape(R+1,C+1,2)
    maze_true = maze_true_flat.reshape(R+1,C+1,2)
    # --- Drawing function stays exactly the same ---
    def draw_maze(ax, maze, title):
        maze = jnp.array(maze)
        H, W,_ = maze.shape
        H,W = H-1,W-1
        maze = jnp.flip(maze, axis=2)

        ax.clear()
        ax.set_aspect("equal")
        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.invert_yaxis()
        ax.set_title(title)

        north_outer_wall = maze[0,:,0].flatten()
        west_outer_wall = maze[:,0,1].flatten()
        
        for r in range(H):
            for c in range(W):
                r_idx=r+1
                c_idx=c+1
                S,E = maze[r_idx, c_idx]
                x0, x1 = c, c+1
                y0, y1 = r, r+1

                ax.plot([x1, x1], [y0, y1+1], color=str(1-E), linewidth=5)
                ax.plot([x0, x1+1], [y1, y1], color=str(1-S), linewidth=5)

        for i in range(W):
            x_out_0,x_out_1 =i, i+1
            y_out_0,y_out_1 =i, i+1
            ax.plot([x_out_0, x_out_1], [0, 0], color=str(1-north_outer_wall[i+1]), linewidth=5)
            ax.plot([0,0], [y_out_0,y_out_1], color=str(1-west_outer_wall[i+1]), linewidth=5)

        ax.set_xticks([])
        ax.set_yticks([])

    # --- Ensure figure has two axes ---
    if len(fig.axes) < 2:
        fig.clear()
        ax_pred = fig.add_subplot(2, 1, 1)
        ax_true = fig.add_subplot(2, 1, 2)
    else:
        ax_pred, ax_true = fig.axes[:2]

    # Compute BCE loss
    bce_loss = compute_bce_loss(maze_pred_flat, maze_true_flat)

    # Draw
    draw_maze(ax_pred, maze_pred, f"Predicted Maze BCE Loss {bce_loss:.3f}")
    draw_maze(ax_true, maze_true, "True Maze (binary walls)")

    fig.tight_layout()
    fig.canvas.draw()
    fig.canvas.flush_events()





class Logger:

    def __init__(self, logging_size=1000000, obs_size=45, log_dir=None):
        self.logging_size = logging_size
        self.obs_size = obs_size
        self.counter = 0
        self.data = {}
        self.reset_log()
        current_file_path = Path(__file__).resolve()
        if log_dir is None:
            self.logging_dir = str(current_file_path.parent) + "/logs"
        else:
            self.logging_dir = log_dir

    def log(
        self,
        observation,
        action,
        joint_pos,
        joint_vel,
        target_vel,
        angular_vel,
        last_action,
    ):
        self.data["observation"][self.counter] = observation
        self.data["action"][self.counter] = action
        self.data["joint_pos"][self.counter] = joint_pos
        self.data["joint_vel"][self.counter] = joint_vel
        self.data["target_vel"][self.counter] = target_vel
        self.data["angular_vel"][self.counter] = angular_vel
        self.data["last_action"][self.counter] = last_action
        self.counter += 1

    def reset_log(self):
        self.counter = 0
        self.data["observation"] = np.zeros((self.logging_size, self.obs_size))
        self.data["action"] = np.zeros((self.logging_size, 12))
        self.data["joint_pos"] = np.zeros((self.logging_size, 12))
        self.data["joint_vel"] = np.zeros((self.logging_size, 12))
        self.data["target_vel"] = np.zeros((self.logging_size, 3))
        self.data["angular_vel"] = np.zeros((self.logging_size, 3))
        self.data["last_action"] = np.zeros((self.logging_size, 12))

    def save_log(self):
        log_number = self.get_next_log_number()
        if type(self.data["observation"]) != list:
            self.data["observation"] = self.data["observation"][: self.counter].tolist()
            self.data["action"] = self.data["action"][: self.counter].tolist()
            self.data["joint_pos"] = self.data["joint_pos"][: self.counter].tolist()
            self.data["joint_vel"] = self.data["joint_vel"][: self.counter].tolist()
            self.data["target_vel"] = self.data["target_vel"][: self.counter].tolist()
            self.data["angular_vel"] = self.data["angular_vel"][: self.counter].tolist()
            self.data["last_action"] = self.data["last_action"][: self.counter].tolist()
        else:
            self.data["observation"] = self.data["observation"][: self.counter]
            self.data["action"] = self.data["action"][: self.counter]
            self.data["joint_pos"] = self.data["joint_pos"][: self.counter]
            self.data["joint_vel"] = self.data["joint_vel"][: self.counter]
            self.data["target_vel"] = self.data["target_vel"][: self.counter]
            self.data["angular_vel"] = self.data["angular_vel"][: self.counter]
            self.data["last_action"] = self.data["last_action"][: self.counter]
        with open(f"{self.logging_dir}/log_{log_number}.yaml", "w") as file:
            yaml.dump(self.data, file)

    def get_next_log_number(self):
        def file_with_substring_exists(directory, substring):
            for filename in os.listdir(directory):
                if fnmatch.fnmatch(filename, f"*{substring}*"):
                    return True
            return False

        substring = "log_"
        if file_with_substring_exists(self.logging_dir, substring):
            p = Path(self.logging_dir)
            filenames = [str(file.name) for file in p.iterdir() if file.is_file()]
            logs = []
            for name in filenames:
                parts = name.split("_")
                if len(parts) > 1:
                    p = parts[1]
                    number = p.split(".")[0]
                    logs.append(int(number))
                else:
                    continue
            log = max(logs) + 1
        else:
            log = 0

        return log

def get_highest_maze_index(directory):
    """
    Scans the directory for 'mazes_i.pkl' and returns the highest i.
    Returns 0 if no matching files are found.
    """
    # Ensure the directory actually exists to avoid errors
    if not os.path.exists(directory):
        return 0

    # This regex looks for 'mazes_', captures digits (\d+), and ends with '.pkl'
    pattern = re.compile(r"mazes_(\d+)\.pkl")
    max_i = 0
    found = False

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            # Extract the integer from the first capture group
            current_i = int(match.group(1))
            if current_i > max_i:
                max_i = current_i
            found = True

    return max_i if found else 0