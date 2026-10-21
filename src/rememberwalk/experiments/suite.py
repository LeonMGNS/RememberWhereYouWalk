"""Experiment suite: which trained runs to compare and how to evaluate and render them.

A suite file (``configs/experiments/paper.yaml``) never repeats training hyperparameters.
Every run is rebuilt from the ``config.yaml`` that ``scripts/train.py`` writes next to its
checkpoints, so a policy is always evaluated with the configuration it was trained with.
Only the suite's ``overrides`` (rendering model, shared eval mazes, ...) are applied on top.
"""
import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from rememberwalk.utils.paths import CKPT_DIR, CONFIG_DIR, ROOT_DIR, MAZE_DIR

# Suffix of the persistent copy PPOTrainer writes when the curriculum is complete
SECURED_CKPT_SUFFIX = "_curr_complete"


@dataclass
class Approach:
    name: str
    color: tuple            # 0-255 RGB, used by every figure and the video
    runs: list              # checkpoint paths relative to CKPT_DIR, one per seed
    config: str = None      # training config name, only for runs without a saved config.yaml
    showcase_run: int = 0   # run used for the prediction figure and the live video

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    @property
    def color_unit(self) -> tuple:
        return tuple(c / 255.0 for c in self.color)


@dataclass
class Suite:
    name: str
    output_dir: Path
    overrides: dict
    curriculum_level: float
    terrain_curriculum_level: float
    nr_envs: int
    randomization_key: int
    approaches: list
    figures: dict
    showcase_env_index: int
    showcase_timesteps: list
    video: dict = field(default_factory=dict)

    def data_dir(self, approach: Approach, run_index: int) -> Path:
        return self.output_dir / "data" / approach.slug / f"run_{run_index + 1}"

    def output_path(self, relative: str) -> Path:
        return self.output_dir / relative


def load_suite(path) -> Suite:
    path = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    output_dir = Path(raw.get("output_dir", f"outputs/experiments/{path.stem}"))
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir

    approaches = []
    for entry in raw["approaches"]:
        approach = Approach(
            name=entry["name"],
            color=tuple(entry["color"]),
            runs=list(entry.get("runs") or []),
            config=entry.get("config"),
            showcase_run=int(entry.get("showcase_run", 0)),
        )
        if not approach.runs:
            print(f"** Skipping approach '{approach.name}': no runs listed in {path}")
            continue
        if not 0 <= approach.showcase_run < len(approach.runs):
            raise ValueError(f"showcase_run {approach.showcase_run} of '{approach.name}' is not one of its {len(approach.runs)} runs")
        approaches.append(approach)
    if not approaches:
        raise ValueError(f"No approach in {path} lists any runs")

    showcase = raw.get("showcase", {})
    rollout = raw.get("rollout", {})
    return Suite(
        name=raw.get("name", path.stem),
        output_dir=output_dir,
        overrides=raw.get("overrides", {}),
        curriculum_level=float(raw.get("curriculum_level", 1.0)),
        terrain_curriculum_level=float(raw.get("terrain_curriculum_level", 1.0)),
        nr_envs=int(rollout.get("nr_envs", 100)),
        randomization_key=int(rollout.get("randomization_key", 0)),
        approaches=approaches,
        figures=raw.get("figures", {}),
        showcase_env_index=int(showcase.get("env_index", 0)),
        showcase_timesteps=[int(t) for t in showcase.get("timesteps", [10, 1750, 3499])],
        video=raw.get("video", {}),
    )


def _saved_config_path(run_path: str):
    """config.yaml of the training run that produced ``run_path`` (<run dir>/<step>)."""
    run_dir = (CKPT_DIR / run_path).parent
    candidates = [run_dir / "config.yaml"]
    if run_dir.name.endswith(SECURED_CKPT_SUFFIX):
        candidates.append(run_dir.with_name(run_dir.name[:-len(SECURED_CKPT_SUFFIX)]) / "config.yaml")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _upgrade_legacy_keys(config):
    """Translate keys of configs saved by older code versions to the current names."""
    training = config.get("training", {})
    if "discount_factor" not in training and "min_discount_factor" in training:
        training["discount_factor"] = training["min_discount_factor"]
    training.pop("min_discount_factor", None)
    training.pop("max_discount_factor", None)
    return config


def training_config(approach: Approach, run_path: str) -> dict:
    """The training config of one run, as a plain dict."""
    if not (CKPT_DIR / run_path).exists():
        raise FileNotFoundError(f"Checkpoint {CKPT_DIR / run_path} of '{approach.name}' does not exist")
    saved = _saved_config_path(run_path)
    if saved is not None:
        source = saved
    elif approach.config is not None:
        source = CONFIG_DIR / f"{approach.config}.yaml"
        print(f"   {run_path}: no saved config.yaml, using {source.name}")
    else:
        raise FileNotFoundError(
            f"{run_path} has no config.yaml next to it. Add 'config: <training config name>' "
            f"to the approach '{approach.name}' in the suite file.")
    return _upgrade_legacy_keys(OmegaConf.to_container(OmegaConf.load(source), resolve=True))


def experiment_config(suite: Suite, approach: Approach, run_path: str, nr_envs: int = None) -> dict:
    """Training config of a run plus the suite overrides, slimmed down for evaluation."""
    config = OmegaConf.create(training_config(approach, run_path))
    for key, value in suite.overrides.items():
        OmegaConf.update(config, key, value, merge=True)
    config = OmegaConf.to_container(config, resolve=True)

    # Only the parameters are restored; the training batch is never used
    config["training"]["nr_envs"] = 1
    config["training"]["batch_size"] = 1
    config["training"]["minibatch_size"] = 1
    config["training"]["num_updates_per_batch"] = 1
    config["training"]["upload_to_wandb"] = False
    config["eval"]["nr_eval_envs"] = nr_envs if nr_envs is not None else suite.nr_envs

    eval_maze_file = MAZE_DIR / f"maze_{config['eval']['file_eval_maze_nr']}.pkl"
    if not eval_maze_file.is_file():
        raise FileNotFoundError(f"Eval maze file {eval_maze_file} is missing; generate it before running the experiments")
    return config


def environment_key(config: dict) -> str:
    """Runs whose configs only differ in seed or name can share one compiled environment."""
    config = copy.deepcopy(config)
    config.pop("run_name", None)
    config.get("training", {}).pop("seed", None)
    return json.dumps(config, sort_keys=True, default=str)
