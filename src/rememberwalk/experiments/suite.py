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
    imported: bool = False  # eval data (and live recording) copied in from elsewhere; runs are labels, not checkpoints

    @property
    def slug(self) -> str:
        return re.sub(r"[^a-z0-9]+", "_", self.name.lower()).strip("_")

    @property
    def color_unit(self) -> tuple:
        return tuple(c / 255.0 for c in self.color)


@dataclass
class Figure:
    kind: str               # bce | exploration | heat_maps | predictions
    file: str               # relative to the suite output_dir
    approaches: list        # Approach objects, in display order
    labels: list = None     # optional short display names, one per approach (default: approach names)
    colors: list = None     # bce only: optional RGB (0-1) per approach, overriding the approach colors
    linestyles: list = None # bce only: optional named dash pattern per approach (plots.style.LINESTYLES)


# Figures made when a suite file has no 'figures' section
DEFAULT_FIGURES = {
    "bce": "plots/bce_error.pdf",
    "exploration": "plots/exploration.pdf",
    "heat_maps": "plots/heat_maps.pdf",
    "predictions": "plots/maze_predictions.pdf",
}


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
    figures: list           # Figure objects
    showcase_env_index: int
    showcase_timesteps: list
    video: dict = field(default_factory=dict)
    video_approaches: list = field(default_factory=list)

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
            imported=bool(entry.get("imported", False)),
        )
        if not approach.runs:
            print(f"** Skipping approach '{approach.name}': no runs listed in {path}")
            continue
        if not 0 <= approach.showcase_run < len(approach.runs):
            raise ValueError(f"showcase_run {approach.showcase_run} of '{approach.name}' is not one of its {len(approach.runs)} runs")
        approaches.append(approach)
    if not approaches:
        raise ValueError(f"No approach in {path} lists any runs")

    by_name = {a.name: a for a in approaches}
    listed = {entry["name"] for entry in raw["approaches"]}

    def select(names, what):
        """Approaches named in a figure or the video; all approaches if names is None."""
        if names is None:
            return list(approaches)
        unknown = [n for n in names if n not in listed]
        if unknown:
            raise ValueError(f"{what} in {path} names unknown approaches {unknown}")
        return [by_name[n] for n in names if n in by_name]

    figures = []
    for name, entry in (raw.get("figures") or DEFAULT_FIGURES).items():
        if isinstance(entry, str):
            entry = {"file": entry}
        figure = Figure(kind=entry.get("kind", name), file=entry["file"],
                        approaches=select(entry.get("approaches"), f"Figure '{name}'"))
        if entry.get("labels") is not None:
            if entry.get("approaches") is None or len(entry["labels"]) != len(entry["approaches"]):
                raise ValueError(f"Figure '{name}' in {path}: give one label per listed approach")
            label_of = dict(zip(entry["approaches"], entry["labels"]))
            figure.labels = [label_of[a.name] for a in figure.approaches]
        for option in ("colors", "linestyles"):
            if entry.get(option) is None:
                continue
            if figure.kind != "bce":
                raise ValueError(f"Figure '{name}' in {path}: '{option}' is only supported for bce figures")
            if entry.get("approaches") is None or len(entry[option]) != len(entry["approaches"]):
                raise ValueError(f"Figure '{name}' in {path}: give one entry of '{option}' per listed approach")
            value_of = dict(zip(entry["approaches"], entry[option]))
            values = [value_of[a.name] for a in figure.approaches]
            if option == "colors":
                values = [tuple(c / 255.0 for c in color) for color in values]
            else:
                from rememberwalk.experiments.plots.style import LINESTYLES
                unknown = sorted(set(values) - set(LINESTYLES))
                if unknown:
                    raise ValueError(f"Figure '{name}' in {path}: unknown linestyles {unknown}, use {list(LINESTYLES)}")
            setattr(figure, option, values)
        if not figure.approaches:
            print(f"** Skipping figure '{name}': none of its approaches has runs")
            continue
        figures.append(figure)

    video = raw.get("video", {})
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
        figures=figures,
        showcase_env_index=int(showcase.get("env_index", 0)),
        showcase_timesteps=[int(t) for t in showcase.get("timesteps", [10, 1750, 3499])],
        video=video,
        video_approaches=select(video.get("approaches"), "The video"),
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
    if approach.imported:
        raise ValueError(f"'{approach.name}' is imported: its runs are data labels, not checkpoints")
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
