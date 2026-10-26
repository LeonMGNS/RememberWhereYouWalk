# Remember Where You Walk

Code to reproduce the results of the paper: train the policies, evaluate them and create the figures.

## Setup

```bash
conda create -n rememberwalk python=3.12
conda activate rememberwalk
pip install -e ".[cuda,vis,experiments]"   # drop "cuda" for CPU only
```

The maze files (`data/mazes/*.pkl`) are part of the repo.

## Training

Training logs to [Weights & Biases](https://wandb.ai), so log in once before the first run:

```bash
wandb login
```

Start a training with one of the configs in `configs/` (`lsmm`, `lsmm_additive`, `lstm`, `sru`, `fixed_window`, `transformer`):

```bash
python scripts/train.py --config-name=lsmm training.seed=0
```

The paper uses seeds 0, 1 and 2 per approach. For the memory-channel ablation, train `lsmm` with
`"architecture.memory.matrix_memory_shape=[50,50,C]" "architecture.memory.matrix_memory_update_window_shape=[5,5,C]"`
for C = 1, 2, 3, 5, 10.

Checkpoints and the run's `config.yaml` are written to `checkpoints/non_sweep/<run>/`. At the end, the
path of the final checkpoint is printed.

## Visualizer

Watch a trained policy in the interactive MuJoCo viewer. Use the training config of the policy and its
checkpoint path relative to `checkpoints/`:

```bash
python scripts/visualize.py --config-name=lsmm +visualizer.policy_name=non_sweep/<run>/<step>
```

- Optional: `+visualizer.curriculum_level=1.0` and `+visualizer.terrain_curriculum_level=1.0` (task and wall
  difficulty, default 1.0). Add `robot.config=mjx_navigation_plot` for the coloured maze of the paper.
- Press any key in the terminal to start. In the viewer, `Enter` starts a new episode and `S` pauses (resume in the terminal).

## Experiments and paper figures

Everything is configured in `configs/experiments/paper.yaml`:

- `approaches[*].runs`: the checkpoint paths of each approach, relative to `checkpoints/`, one per seed
  (paste the paths printed at the end of training).
- `approaches[*].config`: the training config to use for checkpoints without a saved `config.yaml`.
  Otherwise the saved config is used.
- `figures` / `video`: which approaches each figure and the video show.

Then create the evaluation data, all figures and the video with one command:

```bash
python scripts/run_experiments.py
```

The results are written to `outputs/experiments/paper/` (`data/`, `plots/`, `videos/`). Single steps can be
re-run, e.g. only the figures from existing data:

```bash
python scripts/run_experiments.py --stages plots render
```

Notes:
- On a machine without a display, set `MUJOCO_GL=egl` for the `render` and `video` stages.
- An approach marked `imported: true` (FixedWindow in the paper) is not re-evaluated. Copy its
  `eval_data.pkl` files to `outputs/experiments/paper/data/<approach>/run_<i>/` first.
