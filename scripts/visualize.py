import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".20"
import jax
import jax.numpy as jnp
import cv2
from mujoco import mjx
import orbax.checkpoint as ocp
import time
import mujoco
from rememberwalk.utils.Viewer import Viewer
from rememberwalk.utils.paths import CKPT_DIR
from rememberwalk.utils.script_utils import dir_path, generate_unique_path,update_maze_figure,restore_ckpt,render
from flax.training import orbax_utils
from flax import serialization
from rememberwalk.actor_critic.Transformer import TrainingState as TrainingStateTransformer
from rememberwalk.terrain.MazeStaticTerrainGenerator import MazeStaticGenerator
import optax
import matplotlib.pyplot as plt
import sys
import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from rememberwalk.utils.generate_setup import generate_setup
from rememberwalk.utils.maze_visual_runtime import MazePainter



class Visualizer:

    def __init__(self,visualizer_setup,env_setup,training_setup):
        self.env_config = env_setup
        self.visualizer_config = visualizer_setup
        self.training_config = training_setup
        self.curr_coeff = self.visualizer_config["curriculum_level"]
        self.terrain_curr_coeff = self.visualizer_config["terrain_curriculum_level"]
        self.env_config["initial_curr_coeff"] = self.curr_coeff
        self.visualizer_dict = self.visualizer_config["visualizer_dict"]
        self.seed = int(time.time())
        self.ckpt_dir = f"{CKPT_DIR}/"
        self.joystick = None
        self.cpu_device = jax.devices("cpu")
        self.actor_critic = self.training_config["actor_critic"]
        self.env = self.env_config["mode"](self.env_config)
        self.visualizer_dict["env"]= self.env
        ckpt_dir = f"{CKPT_DIR}/non_sweep"
        if visualizer_setup["policy"] == "None":
            self.policy_params = self.actor_critic.params
        else:
            params = restore_ckpt(ckpt_dir,self.actor_critic,visualizer_setup)
            self.policy_params = visualizer_setup["params_type"](**params)

        self.visualizer_dict["parameters"] = self.policy_params
        self.use_observation_history = self.visualizer_config["use_observation_history"]
        self.renderer = mujoco.Renderer(self.env.model, 480, 640)
        self.callback = self.visualizer_config["visualizer_callback_function"]

        # Paper-figure maze colouring. Off by default so existing configs and
        # older XMLs keep working unchanged.
        self.use_plot_coloring = True
        self.maze_painter = None
        if self.use_plot_coloring:
            try:
                self.maze_painter = MazePainter(self.env.model,orient="flip_rows")
            except ValueError as e:
                # Model predates the maze_tex texture -- warn and carry on
                # rather than killing a long visualisation run.
                print(f"** use_plot_coloring disabled: {e}")
                self.use_plot_coloring = False

        if not self.visualizer_config["video"]:
            self.run_viewer()
        else:
            self.create_video()
 
    def create_video(self):
        eval_seed = int(10)
        key= jax.random.PRNGKey(eval_seed)
        self.env.curr_coeff = 1.0
        
        #step_jit = jax.jit(self.env.step,backend="cpu")
        step_jit = self.env.step
        #reset_jit = jax.jit(self.env.reset,backend="cpu")
        reset_jit = self.env.reset
        state = reset_jit(key)
        state = state.replace(curr_coeff=self.curr_coeff, terrain_curr_coeff=self.terrain_curr_coeff)
        self.visualizer_dict["state"] = state
        eval_key, _ = jax.random.split(key)
        metrics = state.metrics
        images = []
        print("Before simulation loop")
        for i in range(self.env.episode_length):
            if self.visualizer_config["used_height_encoder"]:
                actions = self.actor_critic.get_deterministic_action(self.actor_parameters, self.height_encoder_parameters, state.obs, state.height_obs)
                self.env.model.hfield_data = state.model.hfield_data
            elif self.visualizer_config["used_gru"]:
                actions = self.actor_critic.get_deterministic_action(self.gru_params,state.obs, state.height_obs)
                self.env.model.hfield_data = state.model.hfield_data

            else:
                actions = self.actor_critic.get_deterministic_action(self.actor_parameters, state.obs)
            state, eval_key = step_jit(state, actions, eval_key)

            eval_key, _ = jax.random.split(eval_key)
            # Repaint before rendering. No-ops on frames where the heightfield
            # is unchanged, so it costs one array compare per step.
            if self.maze_painter is not None:
                self.maze_painter.paint(self.env.model, renderer=self.renderer)
            images.append(render(state,self.visualizer_config,self.env,self.renderer))
        fps = 50
        print("After simulation loop")
        print(len(images))
        self.renderer.close()
        output_path = generate_unique_path(dir_path,self.visualizer_config["policy"])
        height, width, layers = images[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for i in range(len(images)):
            out.write(images[i])
        out.release()

    def key_callback(self,keycode):
            if keycode == 257:
                self.reset_requested = True
            if keycode == 83:
                self.pause_sim = True

    def run_viewer(self):
        key = jax.random.PRNGKey(self.seed)
        eval_key = jax.random.PRNGKey(self.seed)
        step_jit = jax.jit(self.env.step,backend="cpu")
        reset_jit = jax.jit(self.env.reset,backend="cpu")
        self.env.parameters["intial_state"] = self.env.parameters["intial_state"].replace(curr_coeff=self.curr_coeff,terrain_curr_coeff=self.terrain_curr_coeff)
        state = reset_jit(self.curr_coeff,self.terrain_curr_coeff,eval_key)
        if self.visualizer_config["run_name"] == "Walking_MLP_Base":
            state = state.replace(curr_coeff=jnp.asarray(self.curr_coeff), terrain_curr_coeff=self.terrain_curr_coeff,done=jnp.array(True,jnp.bool_))
        else:
            state = state.replace(curr_coeff=jnp.asarray(self.curr_coeff), terrain_curr_coeff=self.terrain_curr_coeff,done=jnp.array(True,jnp.bool_),max_allowed_curr_coeff=self.curr_coeff,max_allowed_terrain_curr_coeff=self.terrain_curr_coeff)
        self.visualizer_dict["state"] = state
        self.visualizer_dict["max_loc_error"] = 0.0
        self.visualizer_dict["test_height_map"] = jnp.zeros_like(state.height_obs)
        key, _ = jax.random.split(key)
        pipeline_state = mjx.get_data(self.env.model, state.pipeline_state)
        action_jit = jax.jit(self.actor_critic.get_deterministic_action,backend="cpu")
        self.reset_requested = False
        self.pause_sim = False
        

        viewer = Viewer(self.env.model, pipeline_state,render_callback=self.key_callback)
        #self.maze_painter.calibrate(self.env.model, viewer=viewer.viewer)
        counter = 0
        while viewer.viewer.is_running():
            if counter == 0:
                input("Press any button to start sim")
                counter +=1
            
            if self.reset_requested:
                print("In Reset")
                self.visualizer_dict["state"] = self.visualizer_dict["state"].replace(n_steps=3500,curr_coeff=jnp.asarray(self.curr_coeff), terrain_curr_coeff=self.terrain_curr_coeff)
                self.reset_requested = False

            if self.pause_sim:
                input("Sim Paused: Press any key to resume")
                self.pause_sim = False
            time_start = float(time.time())
            self.visualizer_dict,key = self.callback(action_jit,self.visualizer_dict,key)
            state, key = step_jit(self.visualizer_dict["state"], self.visualizer_dict["action"], key)
            self.visualizer_dict["state"] = state
            self.env.model.geom_pos = state.model.geom_pos
            self.env.model.hfield_data = state.model.hfield_data
            viewer.viewer.update_hfield(0)
            if self.maze_painter is not None:
                self.maze_painter.paint(self.env.model, viewer=viewer.viewer)

            viewer.update(state.pipeline_state, state.model)
            mujoco.mj_camlight(self.env.model, pipeline_state)
            time_end = float(time.time())
            if (time_end - time_start) < 0.02:
                time.sleep(0.02 - (time_end - time_start))


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("No config file specified.\n"
                        "You must define a configuration file to run this script.\n"
                        "Example usage: python3 runner.py --config-name=Matrix_Memory_GR_Loc\n")

    
    #Load config
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    config["training"]["nr_envs"] = 1
    config["training"]["batch_size"] = 1
    config["training"]["minibatch_size"] = 1
    config["training"]["num_updates_per_batch"] = 1
    config["eval"]["nr_eval_envs"] = 1

    #Generate setup
    setup = generate_setup(config)
    visualizer_setup = setup["visualizer"]
    env_setup = setup["env"]
    training_setup = setup["training"]

    visualizer = Visualizer(visualizer_setup,env_setup,training_setup)




if __name__ == "__main__":
    main()