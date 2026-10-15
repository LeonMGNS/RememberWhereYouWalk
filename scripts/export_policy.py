from rememberwalk.utils.script_utils import dir_path,extract_and_save_actor,restore_ckpt,ckpt_dir
from omegaconf import DictConfig, OmegaConf
import hydra
from rememberwalk.utils.generate_setup import generate_setup

@hydra.main(version_base="1.3", config_path=f"{dir_path}/script/configs", config_name=None)
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
    ckpt_dir = f"{dir_path}/checkpoints/non_sweep"
    deployment_dir = f"{dir_path}/deployment_checkpoints/{visualizer_setup["deployment_name"]}"
    params = restore_ckpt(ckpt_dir,None,visualizer_setup)
    extract_and_save_actor(params,deployment_dir)



if __name__ == "__main__":
    main()