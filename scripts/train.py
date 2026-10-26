
from rememberwalk.utils.checkpointing import get_dir_and_run_for_checkpointing
from rememberwalk.utils.wandb_logging import filter_nested, EmptyLogger
from rememberwalk.utils.generate_setup import generate_setup
from rememberwalk.utils.paths import CKPT_DIR
from pathlib import Path
import optax
import time
import wandb
import sys
import importlib
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig




def save_run_config(ckpt_dir: str, config: dict):
    # The experiments rebuild the policy from this file, so it must be the exact config of the run
    OmegaConf.save(OmegaConf.create(config), Path(ckpt_dir) / "config.yaml")


def print_final_checkpoint(ckpt_dir: str, training_state):
    final_ckpt = (Path(ckpt_dir) / str(int(training_state.training_steps))).relative_to(CKPT_DIR)
    print(f"Final checkpoint (use this path in configs/experiments/paper.yaml): {final_ckpt}")


@hydra.main(version_base="1.3", config_path="../configs", config_name=None)
def main(cfg: DictConfig):
    if not cfg:
        raise ValueError("No config file specified.\n"
                        "You must define a configuration file to run this script.\n"
                        "Example usage: python3 scripts/train.py --config-name=lsmm\n")
    
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    upload_to_wandb = config["training"]["upload_to_wandb"]

    if config.get("training", {}).get("debug", False):
        print("\n DEBUG MODE DETECTED: Automatically scaling down training parameters...")
        config["training"]["nr_envs"] = 2
        config["training"]["batch_size"] = 8
        config["training"]["minibatch_size"] = 4
        config["training"]["num_updates_per_batch"] = 1
        config["eval"]["nr_eval_envs"] = 2
    
    #Load PPO implementaiton
    from rememberwalk.training.ppo import PPOTrainer

    setup = generate_setup(config)

    if upload_to_wandb:
        print("Updating for WANDB config")
        wandb.login()

        base_name = config.get("run_name", "experiment")
        ckpt_dir,run = get_dir_and_run_for_checkpointing()
        save_run_config(ckpt_dir, config)

        hydra_overrides = HydraConfig.get().overrides.task
        if hydra_overrides:
            # If overrides exist (e.g., during a W&B sweep), stitch them together
            clean_overrides = [o.replace('=', '-').replace('.', '_') for o in hydra_overrides]
            dynamic_part = "__".join(clean_overrides)
            final_run_name = f"{base_name}__{dynamic_part}_{run}"
        else:
            # If no overrides, just append the run number
            final_run_name = f"{base_name}_{run}"
        
        
        #Init wandb
        with wandb.init(project=config.get("wandb_project", "MazeNavigation"),name=f"{final_run_name}",config=config) as logger:
            env_setup = setup["env"]
            training_setup = setup["training"]
            environment = env_setup["mode"](env_setup)
            trainer = PPOTrainer(environment,training_setup,ckpt_dir,run,seed=training_setup["seed"])
            # Start training
            final_state = trainer.train(logger)
            wandb.finish()
        print_final_checkpoint(ckpt_dir, final_state)

    else:
        base_name = config.get("run_name", "experiment")
        ckpt_dir,run = get_dir_and_run_for_checkpointing()
        save_run_config(ckpt_dir, config)
        hydra_overrides = HydraConfig.get().overrides.task
        if hydra_overrides:
            # If overrides exist (e.g., during a W&B sweep), stitch them together
            clean_overrides = [o.replace('=', '-').replace('.', '_') for o in hydra_overrides]
            dynamic_part = "__".join(clean_overrides)
            final_run_name = f"{base_name}__{dynamic_part}_{run}"
        else:
            # If no overrides, just append the run number
            final_run_name = f"{base_name}_{run}"
        env_setup = setup["env"]
        training_setup = setup["training"]
        environment = env_setup["mode"](env_setup)
        trainer = PPOTrainer(environment,training_setup,ckpt_dir,run,seed=training_setup["seed"])
        # Start training
        logger=EmptyLogger()
        final_state = trainer.train(logger)
        print_final_checkpoint(ckpt_dir, final_state)



if __name__ == "__main__":
    main()