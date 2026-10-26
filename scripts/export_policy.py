from rememberwalk.utils.paths import ROOT_DIR
from rememberwalk.utils.checkpointing import extract_and_save_actor, restore_ckpt
from rememberwalk.utils.paths import CKPT_DIR
from omegaconf import DictConfig, OmegaConf
import hydra
from rememberwalk.utils.generate_setup import generate_setup

@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("No config file specified.\n"
                        "You must define a configuration file to run this script.\n"
                        "Example usage: python3 scripts/%s --config-name=lsmm +visualizer.policy_name=non_sweep/local_<id>/<step>\n" % __file__.split("/")[-1])

    
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
    ckpt_dir = str(CKPT_DIR)
    if visualizer_setup["deployment_name"] is None:
        raise ValueError("Set the output name with +visualizer.deployment_name=<name>")
    deployment_dir = f"{ROOT_DIR}/deployment_checkpoints/{visualizer_setup["deployment_name"]}"
    params = restore_ckpt(ckpt_dir,None,visualizer_setup)
    extract_and_save_actor(params,deployment_dir)



if __name__ == "__main__":
    main()