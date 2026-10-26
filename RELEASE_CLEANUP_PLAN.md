# Release cleanup plan: `src/rememberwalk`

Status (2026-10-23): **P0 and P1 done (P1 committed). P2 `actor_critic` done, full suite 44/44 bit-exact. Next: P2 `callback`.** P2 is not committed yet.

**Resume here:**
1. Commit the current state (P0 tests + P1 deletions). Optionally tag the last commit before P1 as `pre-cleanup`.
2. Run the tests with the `visual_loco_mjx` conda env (it has `pytest` + `ruff`):
   - `python -m pytest tests -m "not slow" -q` after every change (~45 s);
   - `python -m pytest tests -q` after every directory (~8.5 min).
   Both run on CPU. Goldens live in `tests/golden/` (git-ignored, machine-specific). Re-record them with `--update-golden` only on known-good code.
3. Continue with §5 P2 items, directory by directory, then P3 (format/annotate), P4 (config/contract renames), P5 (bug decisions B1, B2, B8, B9, B10).
Scope: `src/rememberwalk/` (≈12.1k lines, 24 directories). Directories in order: `actor_critic` first, then the rest alphabetically.

---

## 0. Ground rules (apply to every phase)

1. **Existing checkpoints must keep loading.** The paper runs are restored with orbax. They are also rebuilt from the `config.yaml` saved next to each checkpoint (`experiments/suite.py`). Therefore:
   - Do not rename `Params` / `TrainingState` fields (`policy_parameters`, `f_parameters`, `transformer_parameters`, ...).
   - Do not change flax layer structure or auto-names (`Dense_0`, `LayerNorm_0`, ...) of any network.
   - Do not move or rename modules referenced by import strings in configs (`actor_critic.MatrixMemory.ActorCritic`, `callback.*`, `reward.*`, ...). If a rename is wanted, do it in Phase 4 with an alias module or a mapping in `generate_setup._resolve_import`.
2. **No silent behaviour change.** Keep PRNG key indices (`init_keys[n]`), `jax.random.split` calls and numerical constants. Bugs found below are listed separately (§3). Each one gets an explicit keep or fix decision.
3. Every phase ends with the verification in §4.

## 1. Formatting and annotation conventions (Phase 3 target)

| Topic | Convention |
|---|---|
| Formatter / linter | `ruff format` + `ruff check` (already a dev dependency). Line length 110, configured in `pyproject.toml` (`[tool.ruff]`). Imports sorted (stdlib / third party / `rememberwalk`). |
| Module header | One docstring per file: what the module does, and which config key selects it (e.g. `imports.reward_handler`). |
| Docstrings | Google style (`Args:`, `Returns:`) on every public function and class. Shapes are written explicitly, e.g. `observation: (batch, actor_observation_size)`. |
| Type hints | All public signatures. Fix wrong ones (e.g. `get_done(...) -> SimpleMazeEnv` should return `MazeState`). Remove the copy-pasted `Tuple[jnp.ndarray\|jnp.ndarray\|jnp.ndarray]`. |
| Comments | Explain *why*, not *what*. Remove dead commented-out code, chat-style comments ("excellent placement") and stale references ("replaces the Transformer Encoder"). Section comments (`# --- Height encoder ---`) only in long methods. |
| Naming | snake_case for locals. Fix typos in identifiers that are not part of the config or checkpoint contract (`sefl`, `actor_output_ize`, `propriocpetion`, `gorund`, `setp`). Typos in config keys go to Phase 4. |
| Prints | Library code logs nothing by default. `print` stays only in the user-facing progress of `ppo.train` and the experiment scripts. Debug prints (e.g. 22 in `maze_visual_runtime.py`) are removed or guarded by `verbose`. |

## 2. Phases

| Phase | Content | Behaviour change? |
|---|---|---|
| **P0 Safety net** | Git tag `pre-cleanup`. Golden outputs (§4). Add the ruff config. | no |
| **P1 Delete dead code** | Unused files, functions, attributes, imports and commented-out code (per-directory lists below). | no |
| **P2 Deduplicate** | Shared network heads, a shared actor-critic base, shared loss/BCE helpers, merged near-identical files. Must keep the param trees identical. | no (verified) |
| **P3 Format and annotate** | Conventions from §1, per directory. | no |
| **P4 Contract changes (optional)** | Config-key typos, file/class renames, removing obsolete state fields. Each needs a config migration or alias. | yes, config only |
| **P5 Bug decisions** | Items in §3: either documented as-is (paper results) or fixed in a separate commit. | yes, if fixed |

---

## 3. Bugs and oddities found (decide before P5, do not fix silently)

| # | Where | Finding | Note |
|---|---|---|---|
| B1 | `actor_critic/MLP.py:HeightEncoder` | The 2nd dense layer is applied to `data`, not `height_latent`. The 1st layer is computed and then discarded. | Fixing it changes the param shapes and breaks every checkpoint. Recommend **document only**. |
| B2 | `actor_critic/Transformer.py` | `CrossEntropyHead` is initialised with `transformer_encoded_size` but applied to `transformer_memory_output_features` inputs. It only works because both are equal in the configs. | Use the same variable in both places (no-op for current configs). |
| B3 | `callback/MatrixMemoryCallback.probabilistic_eval_callback` / `visualizer_callback` | `jnp.where(state.done, ...)` without `[:, None, None, None]`, so a batch shape mismatch is possible. | Moot if the probabilistic eval is removed (P1). |
| B4 | `callback/MazeExplorationTransformerCallback` | `probabilistic_eval_callback` builds the history entry twice (2nd without `noisy_true_pos`) and doesn't write it back. `visualizer_callback` unpacks 5 return values from a 4-value function. | Removed by the P1/P2 callback rewrite. |
| B5 | `done/SmallMazeDeploymentDoneFunction` | `lidar_contact`, `x/y_height_field_bound` are computed but unused. `joint_vel_done` is only a metric. | P1 removes the unused ones (`lidar_contact` costs compute every step). |
| B6 | `env/BaseMJXEnv.reset` | Uses `self.parameters` (not set in the base class), the typo key `intial_state`, and returns `None`. Never called. | P1 delete. |
| B7 | `robots/GO2.py` | Branches for `mjx_plot`, `mjx_hfield`, `mjx_hfield_vis`, `mjx_hfield_low` have no matching XML in `assets/go2/`. | P1 delete. |
| B9 | `training/ppo.py:run_deterministic_evaluation` | "Just finished" uses a hard-coded `n_steps >= 3499` instead of `episode_length`. | Only correct for `episode_length: 3500`. Fix in P5 (changes eval metrics for other lengths, e.g. the tests' 20-step episodes). |
| B10 | `terrain/MazeStaticTerrainGenerator`, `SmallMazeTerrainGenerator` | The `use_file_maze: false` branch calls `Maze.generate_terrain()`, which doesn't exist. | All configs load from file. P2: drop the branch and the flag. |
| B8 | `reset/MazeResetFunction.reset_variables` | Draws `use_pred_loc` (a PRNG draw plus a split whose result is discarded) for a field nothing seems to use. | Removing it changes the RNG stream, so P4/P5 only. |

---

## 4. Verification (after every phase)

Implemented in `tests/` (P0, see the log in §6). After every change: `pytest tests -m "not slow"`. After every directory: `pytest tests`.

1. **Param-tree check:** for every config in `configs/*.yaml`, build `ActorCritic` and compare `jax.tree_util.tree_structure` and leaf shapes of `params` against a dump from `pre-cleanup`. Also restore one real checkpoint per approach with `utils/script_utils.restore_ckpt`.
2. **Golden rollout:** with a fixed seed, run 200 env steps plus one deterministic eval on 8 envs per config. Store the metrics, and require bitwise-equal (or `allclose(atol=0)`) results after each phase. This could live in `tests/test_golden.py`; `tests/` is currently empty.
3. **Smoke train:** `scripts/train.py` for 1–2 iterations per config (small `nr_envs`). Run `scripts/run_experiments.py` on one approach.
4. `ruff check` is clean (from P3 on).

---

## 5. Per-directory plan

Line counts are current. "P1", "P2", ... mark the phase each item belongs to.

### 5.1 `actor_critic/` (2 488 lines, 9 files)

What's there: `BaseActorCritic` (log_prob/entropy), `MLP.py` (shared `Actor`, `Critic`, `HeightEncoder` plus an old MLP-only `ActorCritic`), and the approaches `MatrixMemory` (LSMM), `MatrixMemoryAdditive`, `LSTM`, `SRU`, `Transformer`, `FixedWindow`.

**Remove (P1)**
- `BaseActorCritic`: the empty `__init__` and the stub methods without `self` (`get_deterministic_action()`, `evaluate_actions()`, ...). Replace them with a documented abstract interface (`abc.ABC` + `@abstractmethod`) or drop them.
- `MLP.ActorCritic`, `MLP.Params`, `MLP.TrainingState`, `MLP.Transition`: no config uses them. `training/ppo.py:7` only imports them for type hints. Replace those hints with a `Protocol` or the base class.
- `MatrixMemory`/`MatrixMemoryAdditive`: unused attributes `cross_attention_n_channels`, `cross_attention_num_heads`, `cross_attention_qkv_features`, `maximum_loc_delta_pred`, `window_shape`, `sqrt_height_sample_size`, `height_field_resolution`, `actor_critic_dict`; unused dict entry `pred_pos`; unused imports `np`, `Tuple`, `partial`.
- `Transformer`: `memory_buffer_size`, `actor_output_ize` (typo, unused).
- All files: `sqrt_height_sample_size` (unused everywhere).
- Remove chat-style and stale comments ("excellent placement", "replaces the Transformer Encoder", "setp").

**Deduplicate (P2)**
- `LinearVelocityHead` and `CrossEntropyHead` are copy-pasted in 6 files, so move them to one module (`actor_critic/heads.py`, or keep `MLP.py` and rename it to `networks.py` with an alias). The flax param names stay the same because the heads are top-level `init` calls.
- `F` and `G` in `MatrixMemory` are identical classes, so use one `WindowUpdateMLP` instantiated twice. The `F_input`/`G_input` concatenation is computed twice with identical inputs, so compute it once.
- `MatrixMemoryAdditive` is 95 % identical to `MatrixMemory`. Make it a subclass that overrides `Params`, `__init__` (no `F`) and the one update line. Its config strings stay valid.
- `FixedWindow` shares everything with `Transformer` except the encoder. Same for `SRU` and `LSTM`, except the core. Extract a `HistoryActorCritic` / `RecurrentActorCritic` base, or at least shared helpers for actor/critic/head init, `create_optimizer_and_training_state` (identical in all 6) and `evaluate_actions` (identical in all 6).
- Keep the `init_keys[i]` indices exactly as they are now, per approach.

**Format and annotate (P3)**
- Class docstrings describing the architecture (LSMM: memory shape, write window, update rule `M ← (M ⊙ F + G) / (max + 0.02)`; the paper's equation reference).
- Unify the constructor style: `MatrixMemory` takes `actor_critic_dict`, the others take `config`. Use one style (config dict with explicit `.get` defaults).
- Shapes in docstrings for `sample_action`, `get_deterministic_action` and `log_prob_from_observation`.

### 5.2 `callback/` (458 lines, 3 files)

What's there: per-approach `experiment_callback`, `deterministic_eval_callback`, `probabilistic_eval_callback`, `visualizer_callback`, plus a local `compute_bce_loss`/`compute_bce_reward`.

- **P1:** Remove `probabilistic_eval_callback` in all 3 files together with `PPOTrainer.run_probablilistic_evaluation` and the `use_deterministic_eval` switch. Every config uses `use_deterministic_eval: true`, and the probabilistic variants contain bugs B3/B4. Remove the commented-out metric lines, the unused `import mujoco` and `import jax`, and the `print(jnp.linalg.norm(...))` in `MatrixMemoryCallback.visualizer_callback`.
- **P2:** Move `compute_bce_loss`/`compute_bce_reward` to one shared module. They are duplicated in callbacks, collect_trajectories, loss_functions and `utils/script_utils`. `experiment_callback` and `deterministic_eval_callback` differ only in which metrics they collect, so use one function with a `full_metrics: bool` (static) flag.
- **P3:** Real docstrings. The current ones are copy-pasted and wrong: they mention a `key` argument, and `Tuple[...]` return types.

### 5.3 `collision/` (100 lines)

- **P1:** Check whether `get_contact`, `get_collision_lookup` and `get_collision` are all needed. Only `get_collision` is used by `update_lasttouch`. `contact_fn` is used only by the lidar contact in `done`, which is removed in B5.
- **P3:** Docstrings are already good. Remove the `TerrainState`/`TerrainMjxEnv` references, which point to types that no longer exist.

### 5.4 `command/` (65 lines)

- **P1:** Remove the unused `change_vel_prob`, `zero_vel_prob` and the first unused `goal_position = state.goal_position`.
- **P3:** The file name (`VanishingGoalPositionCommandCreator`) and the class name (`StaticMazeCommandCreator`) don't match, and the docstring talks about box/walking terrain and waypoints. Rewrite the docstring to describe the actual behaviour: the goal heading in the body frame, zero when `no_goal_position`. Renaming the file is P4.

### 5.5 `control/` (59 lines)

- **P1:** Remove the unused `extrinsic_p_gain_factor`, `extrinsic_d_gain_factor`, `motor_strength_factor` and `extrinsic_position_offset` attributes, which are hard-coded to 13 joints.
- **P3:** Fine otherwise.

### 5.6 `curriculum/` (87 lines)

- **P2:** The goal and no-goal branches repeat the same 4 conditions with different thresholds. Factor them into `_thresholds(no_goal)` → (inc, dec), and name the magic numbers (0.8/0.4/0.6/0.3, 0.9/0.6 of episode length, −0.25, 0.95) as module constants.
- **P3:** Fix the return type and the docstring (it mentions waypoints and travelled distance).

### 5.7 `done/` (55 lines)

- **P1:** B5 (unused bounds and `lidar_contact`), plus the commented-out `inside_wall`/`tolerances` lines.
- **P3:** Fix the docstring (it lists lidar and waypoint conditions that aren't used) and the return type.
- **P4:** Metric key typo `done_below_gorund` (changes the wandb name).

### 5.8 `env/` (197 lines)

- **P1:** Delete `BaseMJXEnv.reset` (B6). Remove the unused env attributes: `mode`, `save_observation_history`, `observation_history_length`, `episode_length_in_seconds`, `critic_observation_size`, `height_map_max_delay`, `nr_height_map_points`, `critic_nr_height_map_points`, `max_allowed_loc_error`, `initial_mjx_model`, `MazeCellBasedRand`. Verify each with grep and the golden test. Also remove the stray `#` comment in `step` and the wrong comment order ("Reset the state variables if needed" before the PD control).
- **P2:** Consider merging `BaseMJXEnv` into `SimpleMazeEnv`. It is the only subclass and the base only provides `step`.
- **P3:** Describe the step pipeline in the docstring (reset → DR → PD → physics → reward → done → command → curriculum → reset → obs).

### 5.9 `experiments/` (1 730 lines, newest code)

- Already reasonably documented, so only light work.
- **P1:** Check the `print`s (≈30): keep user progress, drop debug output. Remove 4 commented-code lines (plots).
- **P3:** Run ruff; add docstrings to plot functions where missing.

### 5.10 `observation/` (918 lines)

- **P1 (biggest single win):** `TerrainObs` has 19 of 30 methods unreachable from configs or code. That is the whole lidar ray-casting pipeline (`generate_lidar_rays*`, `transform_lidar_rays*`, `get_lidar_points*`, `height_map_from_lidar`, `create_height_map`, `apply_polar_occlusion`, `lidar_transformations*`) plus `get_height_map`, `get_gr_height_map`, `get_gr_height_map_new`, `get_noisy_height_map`, `apply_height_map_noise`, `get_height_from_xpos_interpolated`, `get_idx_from_hfield_coords` and `get_feet_in_edge_box`. That's about 400 lines. Afterwards, drop the `__init__` attributes that only fed these methods, and check whether the `lidar:` config section is still read.
- **P2:** `MazeDeploymentObservationFunctionGRHeightMap` and `SmallMazeDeployment...` differ only in a "hold height map/position every 2nd step" block plus one key split. Merge them with a config parameter (`observation_update_every_n_steps`, default 1). Keep the extra `split` only when n > 1 so the RNG stream stays identical.
- **P3:** Fix the `update_orientation(sefl, ...)` typo; add docstrings with frames and units.

### 5.11 `randomization/` (705 lines)

- **P1:** Remove `MazeCellDR.randomize_propriocpetion_per_grid_cell` and `reset_randomize_propriocpetion_per_grid_cell` (unused). If `env.MazeCellBasedRand` is really unused, the whole `MazeCellDR` and its config import can go (P4 for the config line). Remove 7 commented-code lines.
- **P3:** Normalise the file names in P4 only (`Pertubation_Randomization`, `Init_state_randomization`, mixed case). P3 itself covers docstrings for each DR term with ranges and units.

### 5.12 `reset/` (47 lines)

- **P1:** Nothing safe beyond formatting. Obsolete state resets (`lidar_horizontal_pos`, `travelled_distance`, `nr_goals_reached`, `max_loc_error`, `avg_loc_error`, `use_pred_loc`) belong to P4 together with the state fields (B8).

### 5.13 `reward/` (770 lines, 2 files)

- **P1:** `binary_cross_entropy_computation` is unused in both files. Remove 4 commented-code lines and the empty "Joint velocity limit" stub, if present.
- **P2:** `MazeSmallRewardFunction` and `MazeSmallDeploymentRewardFunction` differ in only 2 constants: the close-to-wall distance (0.35/0.15 vs 0.2/0.1) and the half cell width (1.0 vs 0.5). Use one class with these as config parameters, with defaults equal to the current training values. Derive the half cell width from `space_maze_cell / 2` if that matches both configs. Keep the deployment module as a thin alias until P4.
- **P3:** One docstring line per reward term (what it rewards, scale key in config).

### 5.14 `robots/` (296 lines)

- **P1:** Delete `BaseRobot.scale_action` (unused) and the GO2 branches without XML (B7). Check whether `assets/go2/go2_height_map_vis.xml` is still needed; only old Hydra outputs reference `height_map_vis`.
- **P2:** Merge `BaseRobot` into `GO2` if it adds nothing (it only declares attributes).
- **P3:** Docstrings for the joint/geom id fields and units (gains, limits).

### 5.15 `states/` (133 lines)

- **P2:** `MazeStateObservationHistory` is `MazeState` plus one field. Use `flax.struct.dataclass` inheritance. Check that all construction is keyword-based (it is in `generate_setup.py`).
- **P3:** A one-line comment per field (shape, unit, who writes it).
- **P4:** Remove the obsolete fields (see `reset/`).

### 5.16 `terrain/` (1 238 lines)

- **P1:** Remove `get_maze_nr_from_global_hfield_idx*` (unused in all 3 generators) and `get_valid_global_goal_position`. Remove the commented code. In `wilson_algorithm.py`:
  - The module-level parameter block (`maze_rows = 5` … `terrain_parameters = {...}`) runs on every import. Move it into a `scripts/generate_mazes.py` CLI.
  - Move the offline generation methods (`generate_only_maze_description`, `generate_eval_only_maze_description`, `get_hand_crafted_maze`, `print_maze*`) there as well.
  - Remove `convert_maze_to_wall_tensor_new`.
  - The static reachability check flagged more `Maze` methods; check each by hand because `generate_terrain` is reached dynamically.
- **P2:** `MazeStaticTerrainGenerator`, `SmallMazeTerrainGenerator` and `MazeEvaluationTerrainGenerator` share wall generation and coordinate helpers. Extract a base class. The small-maze one adds random wall thickness and an open start cell, so it keeps its override.
- **P3:** Docstrings with the coordinate conventions (hfield index vs world vs maze cell).

### 5.17 `training/` (1 342 lines)

- **P1:** In `ppo.py`: remove the commented metric initialisation (lines 123–128) and `run_probablilistic_evaluation` (see callbacks). Remove `compute_localization_error_vector` (unused) from `collect_trajectories_lstm.py`.
- **P2:**
  - `transformer_loss.py` and `fixed_window_loss.py` are identical except for the attribute name of the sequence length. Use one `history_loss.py` with `actor_critic.sequence_length` (keep both module paths as aliases until P4).
  - Move `interpolate_entropy_cost`, `compute_bce_loss`, `compute_bce_reward` and the GAE/advantage code (duplicated in 4 loss files and 3 collectors) to `training/common.py`.
  - The 3 collectors share about 70 % of their code; consider one collector with an approach-specific `step_memory` hook.
- **P3:** Docstrings with tensor shapes (unroll length × envs × …).

### 5.18 `utils/` (1 507 lines)

- **P1:** In `script_utils.py`: remove `to_gpu`, `edges_to_wall_tensor`, `unflatten_edge_vector`, `get_maze_figure`, `get_maze_figure_matrix_memory` (about 170 lines of matplotlib), `Logger.save_log` and `get_next_log_number`, and `compute_bce_loss` (dup). Check whether `Logger` and `update_maze_figure` are still used at all. In `Viewer.py`: remove `update_mujoco` and `update_height_map`. In `maze_visual_runtime.py`: remove `paint_mask` and `diagnose`/`sweep_orients` if only used for calibration during development, plus 22 debug prints.
- **P2:** Split `script_utils.py` by purpose (`checkpointing.py`: restore/save/export; `logging.py`). `get_highest_maze_index` belongs with maze generation.
- **P3:** `generate_setup.py` (526 lines): split it into sections (`_build_robot`, `_build_env`, `_build_state`, ...), each with a docstring. Replace the "Warning: could not resolve import" print with raising, because a silent `None` hides config errors.
- **P4:** Config-key typos used in `generate_setup.py`: `terrain_obseravtion_hanlder`, `observation_randomization_hanlder`, `pertubation_randomization_handler`, `intial_state`. Fixing them needs a key-migration step for old saved `config.yaml` files.

---

## 6. Progress log

| Directory | P1 delete | P2 dedup | P3 format | P4 contract | Verified | Notes |
|---|---|---|---|---|---|---|
| actor_critic | ✅ | ✅ | ☐ | ☐ | ✅ P2 | B1, B2 open |
| callback | ✅ | ☐ | ☐ | – | ✅ P1 | B3, B4 resolved (probabilistic eval removed) |
| collision | ✅ (nothing to remove) | – | ☐ | – | ✅ P1 | all 3 functions are live |
| command | ✅ | – | ☐ | ☐ | ✅ P1 | |
| control | ✅ | – | ☐ | – | ✅ P1 | |
| curriculum | ✅ | ☐ | ☐ | – | ✅ P1 | |
| done | ✅ | – | ☐ | ☐ | ✅ P1 | B5 resolved |
| env | ✅ | ☐ | ☐ | – | ✅ P1 | B6 resolved |
| experiments | ✅ (nothing to remove) | – | ☐ | – | ✅ P1 | prints are user progress, kept |
| observation | ✅ | ☐ | ☐ | – | ✅ P1 | TerrainObservation 701 → 228 lines |
| randomization | ✅ | – | ☐ | ☐ | ✅ P1 | MazeCellDR deleted |
| reset | – | – | ☐ | ☐ | ✅ P1 | B8 open (P4) |
| reward | ✅ | ☐ | ☐ | ☐ | ✅ P1 | the 2 files now differ in 2 lines only |
| robots | ✅ | ☐ | ☐ | – | ✅ P1 | B7 resolved |
| states | – | ☐ | ☐ | ☐ | ✅ P1 | |
| terrain | ✅ | ☐ | ☐ | – | ✅ P1 | B10 open (P2) |
| training | ✅ | ☐ | ☐ | ☐ | ✅ P1 | B9 open |
| utils | ✅ | ☐ | ☐ | ☐ | ✅ P1 | script_utils 587 → 197 lines |

Log entries (newest first):
- 2026-10-23: **P2 `actor_critic` done.** 2 348 → 1 309 lines; full suite 44/44 bit-exact (8 min 19 s); `ruff check --select F` clean.
  - New `heads.py` (`LinearVelocityHead`, `CrossEntropyHead`). `BaseActorCritic` gained `init_shared_networks` (keys passed per approach, indices unchanged), `create_optimizer`, `evaluate_actions`, `encode_height`, `predict_linear_velocity`, `predict_distribution`, `actor_distribution`, `sample_from_distribution`.
  - New `history.py`: `HistoryActorCritic` base plus the shared `TrainingState`/`Transition` for Transformer and FixedWindow (re-exported from both modules). Both now expose `sequence_length`, `history_input_size`, `memory_output_features`; the old attribute names are kept for the losses. FixedWindow's own (unused, differently ordered) `Transition` is gone.
  - MatrixMemory: `F`/`G` are one `WindowUpdateMLP`. `MatrixMemoryAdditive` is a subclass overriding `init_update_networks`, `make_params_and_labels`, `apply_window_update`; it re-exports LSMM's `TrainingState`/`Transition`.
  - SRU subclasses LSTM's `ActorCritic` (overrides `core_sizes`, `make_core`) and re-exports LSTM's `Params`/`TrainingState`/`Transition`.
  - **Lesson:** computing the F/G input concatenation once (as planned) broke bit-exactness of the LSMM training epoch (~1e-7, gradient summation order). F and G now each build their own copy via `make_update_input()`. Expect the same effect wherever a shared sub-expression feeds two differentiated branches.
- 2026-10-22: **P1 (delete dead code) complete for all directories.**
  - `src/rememberwalk` went from 12 125 to 10 430 lines: 49 files changed, +80 / −1 790 lines, including configs, `pyproject.toml` and one import in `scripts/visualize.py`.
  - **Final verification:** full suite **44/44 passed, bit-exact** against the pre-cleanup CPU goldens (8 min 36 s). `scripts/train.py`, `visualize.py`, `export_policy.py`, `run_experiments.py` and all experiment modules import cleanly. `ruff check src --select F` is clean except 15 unused locals in the offline `wilson_algorithm.py` (left for P3).
  - **Intermediate runs (all green):**
    - fast suite (25 tests, ~45–50 s) after every directory;
    - full suite 44/44 after `actor_critic` + `callback`;
    - full suite 44/44 after `observation`;
    - full suite 44/44 at the end.
  - **What was removed, per directory:**
    - `actor_critic`:
      - The old MLP-only `ActorCritic`/`Params`/`TrainingState`/`Transition` in `MLP.py`; only `Actor`, `Critic` and `HeightEncoder` are left. `ppo.py` no longer imports them; its 4 type hints were dropped.
      - The empty `__init__` and stub methods in `BaseActorCritic`.
      - Unused attributes: `cross_attention_*`, `maximum_loc_delta_pred`, `window_shape`, `height_field_resolution`, `actor_critic_dict`, `sqrt_height_sample_size` (all 6 approaches), and the Transformer's `actor_output_ize` and `memory_buffer_size`.
      - The unused `pred_pos` entries in the LSMM eval/visualizer dicts.
      - The unused imports `numpy`, `Tuple` and `partial`, plus one chat-style comment.
    - `callback`:
      - `probabilistic_eval_callback` in all 3 files.
      - Together with it: `PPOTrainer.run_probablilistic_evaluation` and the `use_deterministic_eval` switch (evaluation is now always deterministic), the matching `generate_setup` entries, and the `probabilistic_eval_callback` / `use_deterministic_eval` lines in all 7 configs.
      - Commented-out metric lines, unused imports, and unused locals in `visualizer_callback`.
    - `command`: unused `change_vel_prob`, `zero_vel_prob` and a dead `goal_position` assignment.
    - `control`: unused `extrinsic_*` / `motor_strength_factor` attributes and the `jnp` import.
    - `curriculum`: unused `jax` import.
    - `done`: unused height-field bounds and `lidar_contact` (B5), commented code, unused import.
    - `env`:
      - `BaseMJXEnv.reset` (B6).
      - 11 unused `SimpleMazeEnv` attributes: `mode`, `save_observation_history`, `observation_history_length`, `episode_length_in_seconds`, `critic_observation_size`, `height_map_max_delay`, `nr_height_map_points`, `critic_nr_height_map_points`, `max_allowed_loc_error`, `initial_mjx_model`, `MazeCellBasedRand`.
      - The matching unused keys in the `generate_setup` env dict.
    - `observation`:
      - 19 unreachable `TerrainObs` methods: the whole lidar ray-casting pipeline, the old height-map variants, `get_feet_in_edge_box` and the interpolation helpers.
      - The 35 `__init__` attributes that only fed them, including the `jaxkd` kd-tree. `jaxkd` was removed from `pyproject.toml` dependencies.
    - `randomization`:
      - `MazeCellBasedRandomization.py` deleted (constructed but never used). Its `generate_setup` lines and the `maze_cell_based_rand_handler` config key in all configs were removed too.
      - An unused noise draw (it uses an already-split key, so the RNG stream is unchanged), commented code, unused import.
    - `reward` (both files):
      - unused `binary_cross_entropy_computation`;
      - unused locals (`sqrt_curr_coeff`, `out_of_bounce_mask`, `clipped_ids`, `zero_visit_mask`, `current_neighbor_ids`, `not_visited_mask`) and `cell_neighbor_offsets`;
      - the duplicated `"terrain_curr_coeff"` metrics key (same value), commented code, unused import.
    - `robots`: GO2 branches for non-existent XMLs (`mjx_plot`, `mjx_hfield*`, `mjx_hfield_low`; B7), `BaseRobot.scale_action`, unused imports.
    - `terrain`:
      - `get_maze_nr_from_global_hfield_idx*` (all 3 generators) and `get_valid_global_goal_position`; commented code.
      - `wilson_algorithm.py`: `convert_maze_to_wall_tensor_new` and unused imports removed. The module-level parameter block now sits under `if __name__ == "__main__":`, so it no longer runs on import.
    - `training`: `compute_localization_error_vector`, commented metric init in `ppo.py`, unused locals and imports.
    - `utils`:
      - `script_utils.py`: `to_gpu`, `compute_bce_loss` (dup), `unflatten_edge_vector`, `edges_to_wall_tensor`, `get_maze_figure*`, `update_maze_figure`, the `Logger` class and `get_highest_maze_index`.
      - `Viewer.update_mujoco` / `update_height_map`.
      - `MazePainter.paint_mask` / `diagnose` / `calibrate`, `sweep_orients`, `CAL_COLORS` (one-off calibration tools; orientation is settled).
      - An unused local in `generate_setup`.
  - **Compatibility:** checkpoints still restore (test_checkpoints). Saved `config.yaml` files that still contain removed keys (`probabilistic_eval_callback`, `use_deterministic_eval`, `maze_cell_based_rand_handler`) are fine, because `generate_setup` simply ignores them.
  - **New findings during P1 (not changed, decide later):**
    - **B9:** `PPOTrainer.run_deterministic_evaluation` hard-codes `n_steps >= 3499` for the "just finished" test instead of using `episode_length`.
    - **B10:** `MazeStaticGenerator`/`SmallMazeTerrainGenerator` call `self.maze_creator.generate_terrain()` when `use_file_maze` is false, but `Maze` has no such method. Every config uses `use_file_maze: true`. In P2, always load mazes from file and drop the branch and its `use_file_maze` flag.
    - **Config leftovers (P4):** the `lidar:` keys (`horizontal_fov`, `vertical_fov`, `horizontal_step`, `vertical_step`, `min_depth`, `max_depth`, `sampling_resolution`, `*_noise`, `spinning_step_size`, `height_shadow_noise`) and `height_map.height_map_gaussian_std` / `knn_lidar_height_map` are no longer read.
    - **State field leftovers (P4):** `proprio_cell_values` is no longer written by anything (with MazeCellDR gone).
    - **`scripts/visualize.py` (out of `src` scope):** 10 unused imports and 2 unused locals.
  - **Test-harness changes since P0:**
    - The tests now run on **CPU by default** (`JAX_PLATFORMS=cpu` in `conftest.py`) with an even smaller workload: 2 envs, 20-step episodes, unroll 4. The goldens were re-recorded on CPU on the unchanged code before P1 started, and confirmed with a compare run.
    - Tests are sorted by config, and only one built config is kept in memory, which fixes the GPU OOM seen in the first GPU run.
    - Timing: fast suite ~45 s, full suite ~8.5 min.
- 2026-10-22: GPU golden run (before switching to CPU): recording passed 19/19 slow tests. The compare run passed 18/19; `test_training_epoch[fixed_window]` hit a CUDA out-of-memory error from caching all configs (fixed by the one-config cache). The user chose to switch to CPU instead of re-running.
- 2026-10-22: **P0 safety net added** (`tests/`, ruff config in `pyproject.toml`).
  - `tests/harness.py` builds each of the 7 configs shrunk to 4 envs, 40-step episodes, an unroll of 8 and 16 mazes, with seed 0 and curriculum 0.5.
  - `test_param_tree.py` covers the initial params signature (names, shapes, dtypes, sum, abs-sum), the full TrainingState layout and the optimizer labels.
  - `test_checkpoints.py` restores the 4 paper checkpoints (`experiment_checkpoints/*/policy_1`), checks the param signature and runs a 40-step eval with the restored policy.
  - `test_golden_rollout.py` covers a deterministic eval and one jitted training epoch (rollout plus PPO update) per config, plus a repeatability self-check.
  - GPU kernels are nondeterministic by default (~1e-4 drift from step 1). `conftest.py` sets `--xla_gpu_deterministic_ops=true`, which makes runs bit-identical, so goldens are compared exactly by default (`GOLDEN_RTOL`/`GOLDEN_ATOL` loosen this).
  - Goldens are recorded with jax 0.11.2 on an RTX 5090 (env `visual_loco_mjx`). They are hardware and version specific, so re-record them with `--update-golden` on the pre-cleanup commit when testing on another machine.
  - Usage: `pytest tests -m "not slow"` (~15 s) and `pytest tests -m slow` (~30 min).
- 2026-10-22: Plan created from static analysis (import/reference graph over `src`, `scripts`, `configs`; reachability of class methods). No code changed.
