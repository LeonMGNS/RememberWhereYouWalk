"""Create all experiment data, figures and the live video of a suite with one command.

    python scripts/run_experiments.py                          # all stages of configs/experiments/paper.yaml
    python scripts/run_experiments.py --stages plots render    # re-make the figures from existing data
    python scripts/run_experiments.py --force                  # redo the rollouts even if their data exists

Stages, in this order:
    data    deterministic rollout of every run -> <output_dir>/data/<approach>/run_<i>/eval_data.pkl
    plots   BCE, exploration and heat-map figures (read the data, CPU only)
    render  MuJoCo maze-prediction figure (reads the data)
    video   live recording of every approach's showcase run, merged into one video

The render and video stages need an OpenGL context; on a headless machine set MUJOCO_GL=egl.
"""
import argparse
from pathlib import Path

from rememberwalk.experiments.suite import load_suite
from rememberwalk.utils.paths import CONFIG_DIR

STAGES = ["data", "plots", "render", "video"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=str(CONFIG_DIR / "experiments" / "paper.yaml"), help="suite file")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES, help="stages to run")
    parser.add_argument("--force", action="store_true", help="redo rollouts whose data already exists")
    args = parser.parse_args()

    suite = load_suite(args.suite)
    print(f"Suite '{suite.name}': {', '.join(a.name for a in suite.approaches)} -> {suite.output_dir}")

    def output(relative: str) -> Path:
        path = suite.output_path(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    if "data" in args.stages:
        from rememberwalk.experiments.data_creation import create_data
        create_data(suite, force=args.force)

    if "plots" in args.stages:
        from rememberwalk.experiments.plots.bce import make_bce_figure
        from rememberwalk.experiments.plots.exploration import make_exploration_figure
        from rememberwalk.experiments.plots.heat_maps import make_heat_map_figure
        make_bce_figure(suite, output(suite.figures.get("bce", "plots/bce_error.pdf")))
        make_exploration_figure(suite, output(suite.figures.get("exploration", "plots/exploration.pdf")))
        make_heat_map_figure(suite, output(suite.figures.get("heat_maps", "plots/heat_maps.pdf")))

    if "render" in args.stages:
        from rememberwalk.experiments.prediction_figure import make_prediction_figure
        make_prediction_figure(suite, output(suite.figures.get("predictions", "plots/maze_predictions.pdf")))

    if "video" in args.stages:
        from rememberwalk.experiments.live_video import record_live, assemble_video
        recordings = []
        for approach in suite.approaches:
            print(f"=== Live recording: {approach.name}")
            out_dir = suite.output_path(f"videos/{approach.slug}")
            record_live(suite, approach, out_dir)
            recordings.append(out_dir)
        assemble_video(recordings, output(suite.video.get("file", "videos/maze_predictions_live.mp4")))


if __name__ == "__main__":
    main()
