# Experiments transfer plan — RememberWhereYouWalk

Status: **implemented** (see §6). Decisions in §5 were made on 2026-10-21.

## 1. Goal
After training one or more policies, a user creates every paper figure and the live video with **one command**:

```bash
python scripts/run_experiments.py                    # all stages, default suite
python scripts/run_experiments.py --stages plots     # only re-plot from existing data
```

Constraints:
- Training and experiments are split logically.
- There are no two config files that describe the same thing and could disagree.
- The approach-specific settings stay correct, i.e. each policy is evaluated with the architecture it was trained with.

## 2. What exists today (old repo)

| Script | Calls | Needs |
|---|---|---|
| `run_data_creation.sh` | `data_creation.py --config-name=<approach>` ×4 | `experiments.policies`, `nr_experiment_envs`, `randomization_key`, `visualizer.curriculum_level`, eval mazes, `experiment_callback` |
| `run_video_creation.sh` | `record_live_predictions.py` ×4, then `assemble_live_video.py` | `experiments.live_policy/live_out/env_index/approach_index/save_screenshots`, `MazePainter` → **plot XML with `maze_tex`** |
| `plot_experiments.sh` | `plot_bce_error_experiment.py`, `plot_exploration_experiment.py`, `plot_exploration_heat_maps.py` (12 hard-coded `EvalX_1/policy_n` paths), `render_maze_predictions.py --config-name=maze_exploration_plot` | pickles from data creation; `prediction_runs`; plot XML |

Helper modules pulled in:
- `maze_prediction_plots.py`
- `plot_exploration_experiment_old.py` (only `load_exploration_data`)
- `MazeEvaluationTerrainGenerator.py` (still imports `SimpleNavigationEnv`)
- `utils/maze_visual_runtime.py` (already in the new repo)

**Problems found**
1. **Settings are mixed together in every training config.**
   - `run_name` is both the wandb name and the experiment output folder.
   - The `experiments:` and `visualizer:` sections sit inside each training config, and they are partly copy-pasted wrongly: `sru` lists the LSTM checkpoints, `lsmm_additive` uses `run_name: EvalMatrixMemory_1` and LSMM's paths.
   - `maze_exploration_plot.yaml` is a full copy of the LSMM config with only `robot.config` and `out_path` changed, which is exactly the kind of conflicting duplicate to avoid.
2. **The robot XML is chosen per training config.**
   - lstm, fixed_window and transformer train on `mjx_navigation_plot`.
   - lsmm trains on `height_map_vis`.
   - sru and lsmm_additive train on `mjx_navigation`.
   - **Verified:** the physics is identical between `mjx_navigation`, `mjx_navigation_plot` and `height_map_vis`. The plot XML only changes lighting and the `maze_tex` texture. `height_map_vis` adds 251 non-colliding mocap marker bodies (contype/conaffinity 0) that nothing in the new repo uses; they only add MJX cost. So the XML is purely a rendering choice and shouldn't live in the training config.
3. **Approaches are hard-coded in 4 places** (`APPROACH_NAMES`, `APPROACH_RGB`, `sns.color_palette`, `N_POLICIES = 3`, fixed order LSMM/LSTM/FW/TF).
   - SRU and LSMM-additive can't be plotted without code edits.
4. **Each approach's config picks its own eval mazes** (`file_eval_maze_nr`: lsmm_additive uses `only_desc_5000_eval`, the others `only_desc_100_eval`).
   - The figures rely on `env_index: 77` being the *same maze* for every approach, which only holds if all approaches use the same eval set and reset keys.
5. **The LSMM experiment callback in the new repo** has `cell_visits`, `predicted_distribution`, `true_distribution` and `pos` commented out.
   - The heat maps and prediction figure need them.
   - As it is, they would crash for LSMM.
6. **Paths:** absolute `/home/leon-magnus/...` paths, `dir_path/script/configs`, `sys.path` hacks, and imports between sibling scripts (`from plot_bce_error_experiment import ...`).

## 3. Proposed design

### 3.1 Two kinds of configuration, each with one job

```
configs/
  lstm.yaml  sru.yaml  lsmm.yaml  lsmm_additive.yaml  fixed_window.yaml  transformer.yaml  lsmm_deployment.yaml
      └─ TRAINING only: imports, env, robot, architecture, curriculum, terrain, reward, DR, training, eval
  experiments/
      paper.yaml        ← the experiment SUITE: which runs to compare and how to render them
      ablations.yaml    ← (optional) SRU + LSMM-additive vs LSMM
```

- **Training configs:** remove the `experiments:` and `visualizer:` sections entirely. `run_name` stays as the training/wandb name only. All of them use `robot.config: mjx_navigation`, the lean model; the 5×5 deployment config keeps its own XML.
- **Suite file:** the *only* place for experiment settings. It never repeats training hyperparameters. It refers to trained runs and supplies the experiment-only overrides.

### 3.2 A checkpoint carries its own training config (no mismatch possible)
- `train.py` additionally writes the resolved training config next to the checkpoints: `checkpoints/non_sweep/<run>/config.yaml`.
- The experiment runner loads the architecture and env config **from the run directory**. So a policy is always evaluated with the config it was trained with, and nothing has to be kept in sync by hand.
- For your existing paper checkpoints, which have no saved config, a suite entry can name the config instead (`config: lsmm`).

### 3.3 Suite file sketch (`configs/experiments/paper.yaml`)

```yaml
name: paper
output_dir: outputs/experiments/paper      # data/, videos/, plots/ below

overrides:                                 # applied to every run's training config
  robot.config: mjx_navigation_plot        # plot XML (maze_tex); physics identical
  eval.file_eval_maze_nr: only_desc_100_eval
  eval.nr_eval_mazes: 100
curriculum_level: 1.0
terrain_curriculum_level: 1.0

rollout:
  nr_envs: 100
  randomization_key: 0

approaches:                                # order = order in every figure
  - name: "LSMM (ours)"
    color: [1, 115, 178]
    runs:                                  # one entry per seed; any number
      - non_sweep/local_1784828266/2466899968
      - non_sweep/local_.../...
    config: lsmm                           # only needed for runs without a saved config.yaml
  - name: LSTM
    color: [222, 143, 5]
    runs: [...]
  - name: FixedWindow
    ...
  - name: Transformer
    ...

figures:
  bce: plots/bce_grid.pdf
  exploration: plots/exploration_plot.pdf
  heat_maps: plots/heat_map_plot.pdf
  predictions:
    file: plots/maze_predictions.pdf
    run_index: 0                           # which seed per approach is shown
    env_index: 77
    timesteps: [10, 1750, 3499]

live_video:
  run_index: 0
  env_index: 77
  screenshots: [10, 1750, 3499]
  file: videos/maze_predictions_live.mp4
```

What disappears:
- `approach_index`
- `live_out`
- `prediction_runs` with hard-coded `EvalX_1/policy_n`
- `maze_exploration_plot.yaml`
- the 12-path argument lists in the shell scripts

All of it is derived from `approaches`.

### 3.4 Code layout

```
src/rememberwalk/experiments/
  suite.py            load suite, resolve run dirs → (training config + overrides, checkpoint)
  rollout.py          ONE shared reset/eval loop (keys, eval-mode maze generator, callback→step)
                      used by both data creation and live recording, so env_index always matches
  data_creation.py    ← data_creation.py
  live_video.py       ← record_live_predictions.py + assemble_live_video.py
  prediction_figure.py← render_maze_predictions.py (+ MazeEvaluationTerrainGenerator)
  plots/
    style.py          names/colours/order come from the suite (no hard-coded APPROACH_*)
    bce.py            ← plot_bce_error_experiment.py
    exploration.py    ← plot_exploration_experiment.py
    heat_maps.py      ← plot_exploration_heat_maps.py (+ load_exploration_data)
    maze_render.py    ← shared bits of maze_prediction_plots.py
scripts/run_experiments.py   one CLI: --suite paper --stages data,figures,video --force
```

- `MazeEvaluationTerrainGenerator` moves to `terrain/`, with `SimpleNavigationEnv` renamed to `SimpleMazeEnv`.
- Package imports replace the `sys.path` hacks and sibling-script imports.
- All paths go through `utils/paths.py`.

### 3.5 Pipeline of `run_experiments.py`
1. **data:** for each approach × run, roll out deterministically on the shared eval mazes → `data/<approach>/<run_i>/eval_data.pkl`. Skipped if it already exists (data creation is the expensive part); `--force` regenerates.
2. **figures:** bce, exploration and heat maps from the pickles (CPU only). Then the MuJoCo prediction figure.
3. **video:** live recording per approach (`run_index`), then assembly into one mp4 plus screenshots.

### 3.6 Required fixes outside `experiments/`
- **Experiment callbacks:** all of them log the same full metric set (`bce_error`, `cell_visits`, `predicted_distribution`, `true_distribution`, `pos`, `step`), which includes re-enabling LSMM's. The training-time `deterministic_eval_callback` can stay light.
- **`train.py`:** writes `config.yaml` into the run dir, and prints the final checkpoint path at the end of training.
- **`visualize.py` / `export_policy.py`:** the policy and deployment name come from the CLI (`policy=...`), not from a `visualizer:` block in every training config. `generate_setup` gets defaults for the remaining viewer settings. The dead `used_gru`/`used_height_encoder`/`video` flags are dropped.
- **`pyproject.toml`:** add an `experiments` extra (`seaborn`, `matplotlib`, `opencv-python`).
- **Eval mazes:** the suite's eval set must exist, so this depends on the Phase F maze generator. The runner generates it if it's missing, or fails with a clear message.

### 3.7 Out of scope (unless you want them)
- `real_pkl` / `orientation`, the real-world deployment comparison (`real_world_maze_prediction.py`).
- `memory_out` (matrix-memory video).
- The wandb training-curve plot.
- All other `experiments/*.py` in the old repo.

## 4. Verification
- **Smoke suite:** 2 envs, short episode, 1 run per approach, run on CPU. It must produce every output file.
- **Parity:** on one existing paper checkpoint per approach, the new `eval_data.pkl` must match the old repo's output (same keys and env, so equal up to float noise). The figures are then compared side by side.

## 5. Decisions
1. **One suite**, `configs/experiments/paper.yaml`, holding the 4 paper approaches plus the ablations (SRU, LSMM additive). There is no separate `ablations.yaml`.
2. **Explicit checkpoint paths**, relative to `checkpoints/`.
3. **`train.py` writes `config.yaml`** into the run folder.
4. **All training configs use `mjx_navigation`** and `nr_mazes: 300_000`. `lsmm_deployment` keeps its 5×5 XML.
5. **Fixed seeds:** 3 runs per approach.

## 6. Implementation
| Item | Where |
|---|---|
| Suite file | `configs/experiments/paper.yaml` |
| One-command runner (`--stages data plots render video`, `--force`) | `scripts/run_experiments.py` |
| Suite loading, saved-config lookup (also for `*_curr_complete` copies), legacy key upgrade, overrides | `experiments/suite.py` |
| Shared env/reset/eval-dict/policy loading | `experiments/rollout.py` |
| Rollouts → `eval_data.pkl` (+ `run.json`; reuses one compiled env for the seeds of an approach) | `experiments/data_creation.py` |
| BCE / exploration / heat-map figures (pooled variants used in the paper) | `experiments/plots/{bce,exploration,heat_maps}.py`, style in `plots/style.py` |
| MuJoCo prediction figure | `experiments/prediction_figure.py`, `plots/maze_render.py` |
| Live recording + assembly into one video (any number of approaches) | `experiments/live_video.py` |
| Eval terrain generator | `terrain/MazeEvaluationTerrainGenerator.py` |
| `config.yaml` + final checkpoint path printed | `scripts/train.py` |
| LSMM experiment callback logs the full metric set | `callback/MatrixMemoryCallback.py` |
| `experiments:` / `visualizer:` removed from training configs; viewer and export take `+visualizer.policy_name=<path relative to checkpoints/>` | `configs/*.yaml`, `utils/generate_setup.py`, `scripts/visualize.py`, `scripts/export_policy.py` |
| `experiments` extra (opencv, matplotlib, seaborn) | `pyproject.toml` |

**Not ported:**
- The non-pooled plot variants (per-policy grid, per-policy bars, 3-maps-per-cluster heat maps).
- The `--demo` CLIs.
- The `eval_data_*.pkl` fast paths from `filter_eval_data.py`.

**Fixed while porting:**
- In `assemble_live_video.py`, the frame border colours had red and blue swapped. The colour was reversed to BGR but drawn into an RGB frame.
- The 4-approach plot layouts break with 6 approaches. With more than 4 approaches:
  - the BCE plot gets extra height for each additional legend row;
  - the exploration bars get rotated labels;
  - the heat maps use 3 columns with a short KL label.

  With up to 4 approaches the paper layout is unchanged.

## 7. Verification (CPU smoke run, 2026-10-21)
- **Setup:**
  - One tiny training run per config through `scripts/train.py`. Each wrote its `config.yaml` and printed its final checkpoint path.
  - A smoke suite: 6 approaches, 4 envs, 60-step episodes.
  - LSMM listed with 2 runs. LSTM given as a legacy `experiment_checkpoints/.../finished` path without a `config.yaml`, so it falls back to `config: lstm`.
  - An approach without runs, which is skipped.
- **Result:** all stages ran (`data plots render video`), and a second `data` run skipped the existing files. All six `eval_data.pkl` files contain `bce_error`, `cell_visits`, `predicted_distribution`, `true_distribution` and `pos`.
- **Not done:** a parity check against the old repo's `eval_data.pkl` for the paper checkpoints. That needs those checkpoints and the `only_desc_100_eval` maze file (Phase F).
