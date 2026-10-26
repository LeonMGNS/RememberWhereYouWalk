# PPO review — RememberWhereYouWalk

## Context
This is a review of the PPO implementation in the release repo: `training/ppo.py`, the 4 loss functions and the 3 collect-trajectory functions, checked against the actor critics and the env reset logic. It was first written as a report only. The bugs have since been fixed in this repo; the status table below tracks every item. Checkpoints trained with the old code are **not** reproducible bit-for-bit with the fixed code.

## Fix status

| # | Item | Status | Where |
|---|------|--------|-------|
| 1 | LSTM/SRU bootstrap obs | ✅ fixed | `lstm_loss.py`: `critic_next_observation[-1]` |
| 2 | Transformer/FixedWindow history | ➖ intentional, unchanged | — |
| 3 | Timeout bootstraps from next episode | ✅ fixed | `lstm_loss.py`, `matrix_memory_loss.py`: `deltas *= truncation_mask` |
| 4 | KL stop freezes step counter | ✅ fixed | `ppo.py` `minibatch_sgd_step`: steps counted after the `cond` |
| 5 | Final checkpoint = untrained params | ✅ fixed | `ppo.py` `train()`: unpacks the carry, saves and returns the trained state; `save_ckpt` skips a step that is already saved |
| 6 | Velocity target in world frame | ✅ fixed | new `collect_trajectories_functions/linear_velocity_target.py`, used by all 3 collect fns |
| 7 | RL gradients into velocity head (Transformer/FixedWindow) | ✅ fixed | `transformer_loss.py`, `fixed_window_loss.py`: `stop_gradient(pred_linear_vel)` |
| m | Same minibatch permutation every epoch | ↩ reverted, kept as is | A per-epoch permutation copies the whole batch (≈3 GB for LSMM) on every epoch, and the minibatches are large anyway. Shuffled once per batch in `optimize`, as before |
| m | `run_probablilistic_evaluation` broken | ⏭ ignored | probabilistic eval will be removed |
| m | Hard-coded `3499` | ✅ fixed (deterministic eval) | `ppo.py`: `episode_length - 1` |
| m | `.squeeze()` drops env axis | ✅ fixed | `.squeeze()` removed from both transformer losses. The encoder output is already `(T, B, enc)`, so it was a no-op except when B = 1 |
| m | Transformer losses ignore loss scaling factors | ✅ fixed | applies `policy_/value_loss_scaling_factor` |
| m | min/max discount factor | ✅ replaced | single `training.discount_factor` in all configs, `generate_setup.py`, `ppo.py` and the losses; the unused `interpolate_param` was removed from the losses |

Config mapping (all 7 configs): batch 100k, 5000 envs → **unroll_length = 20**, 4 minibatches, 5 epochs, kl_div_thr 0.15, deterministic eval.
- LSTM, SRU → `lstm_loss` + `collect_trajectories_lstm`
- LSMM, LSMM-deployment, LSMM-additive → `matrix_memory_loss` + `collect_trajectories_MatrixMemory`
- Transformer, FixedWindow → `transformer_loss` / `fixed_window_loss` + `collect_trajectories_MazeExplorationTransformer` (seq_len 10)

Env facts used: `step()` auto-resets physics/obs inside the step when `done`, so `critic_obs` on a done step is the *next episode's* first obs. `n_steps`/`summed_reward` are reset lazily at the start of the next step. `truncation = done & n_steps >= episode_length`.

## Findings (most severe first)

1. **LSTM/SRU bootstrap uses the wrong observation** — `loss_functions/lstm_loss.py:60`
   `evaluate_actions(params, critic_observation[-1], critic_next_height_observations[-1])` pairs the *current* obs of the last step with the *next* height map. `critic_next_observation` is unpacked (line 24) but never used. matrix_memory_loss does it correctly (line 51). Effect: biased value target for the last step of each 20-step rollout, which propagates through GAE. Affects lstm.yaml and sru.yaml, so the LSTM vs SRU vs LSMM comparison isn't apples to apples.

2. ~~Transformer/FixedWindow history mismatch~~ → **not a bug: intentional design trade-off** — `transformer_loss.py:45`, `fixed_window_loss.py`
   Storing the full observation history for every step is too memory-heavy, so the history is rebuilt per batch in the loss, starting from zeros. Known side effects worth keeping in mind when reading the logs:
   - With unroll 20 and seq_len 10, steps 0–8 of each rollout are evaluated with a shorter history than the one the action was sampled with.
   - So the ratio is not exactly 1 before the first update.
   - `approx_kl` for these two configs is somewhat higher than for the recurrent ones. That matters for the KL early-stop (#4).

   If this ever needs tightening: storing only the history at t=0 of each rollout (one per env, not one per step) would cost about 5000 × 10 × ~130 floats ≈ 26 MB.

3. **Timeouts bootstrap from the next episode (LSTM, SRU, LSMM)** — `lstm_loss.py:68`, `matrix_memory_loss.py:59`
   At a truncated step, `next_value = baseline[t+1]` is V(first state of the *new* episode), because of the auto-reset. The transformer losses mask this with `deltas *= trunc_mask` (Brax style); these two losses don't. Impact is small (≈1 step per 3500), but it is a wrong TD target.

4. **KL early-stop also freezes the step counter** — `ppo.py:203,208`
   `training_steps` is only incremented inside `new_training_state`. Once `kl_div_stop` fires, the rest of the batch (all remaining minibatches and epochs) adds no steps, so `training_steps` under-counts real env steps. Knock-on effects: the entropy schedule, the logging and checkpoint cadence, the eval step axis, and the `while trained_steps < training_steps` stopping condition, which runs past the configured budget.

5. **Final checkpoint saves the untrained params** — `ppo.py:136-138`
   `carry = (env_state, training_state, key)` rebinds to the *initial* objects. `save_ckpt(training_state, training_state.training_steps)` then saves step-0 params, and `train()` returns them too. The last real params after training are only saved if a periodic save happened to coincide.

6. **Linear-velocity head target is in the world frame** — `collect_trajectories_lstm.py:46`, `_MatrixMemory.py:43`, `_MazeExplorationTransformer.py:49`
   - **The target:** `[qvel[0], qvel[1], qvel[5]]`. For a MuJoCo free joint, qvel[0:3] is the **world**-frame linear velocity and qvel[3:6] is the **body**-frame angular velocity, so the target mixes frames. vx/vy are in the world frame; the yaw rate (qvel[5]) is fine.
   - **The head's input:** only the 45-d actor obs (target heading, no-goal flag, joint pos/vel, last action, body angular velocity, projected gravity). That contains no absolute yaw and no linear velocity.
   - **The env elsewhere already uses the body frame:** the observation function rotates linear velocity with `quat_inv.apply(...)` (`MazeDeploymentObservationFunctionGRHeightMap.py:21`).
   - **Result:** the magnitude of vx/vy is learnable, but not the direction in the world frame. That direction depends on the robot's current yaw, which the head can't see. The MSE-optimal prediction shrinks toward the average over headings, which is roughly the starting heading early in an episode (initial yaw noise is 0) and close to 0 later.
   - **Impact:** all memory variants receive this as an input feature (e.g. MatrixMemory's G/F input). Every method shares it, so the comparison stays fair, but the head isn't the body-velocity estimator its name suggests.
   - **Fix (applied):** the target is now the body-frame `quat_inv.apply(qvel[0:3])[:2]` plus qvel[5], via `linear_velocity_target()`.
   - **Why the head couldn't learn it:** the absolute yaw is only in `noisy_true_pos`, which goes to the memory modules, not to the head. The initial yaw alone doesn't help a memoryless MLP, which needs the *current* yaw.
   - **Check:** if `linear_velocity_error_log` plateaus near the typical walking speed, this is the cause.

7. **Transformer/FixedWindow let RL gradients into the velocity head** — `transformer_loss.py:78`
   `pred_linear_vel` enters the history without `stop_gradient`. LSTM/LSMM wrap it in `stop_gradient` (lstm_loss.py:47, matrix_memory_loss.py:41). This is an inconsistency between the ablation arms.

## Minor
- `ppo.py:224` — one permutation is drawn per batch and reused for all 5 epochs, so the minibatches are identical every epoch.
- `ppo.py:261-289` — `run_probablilistic_evaluation` is broken: it's called with `(eval_dict, key)` but its signature is `(parameters, key)`, it calls reset with only 1 arg, and it mutates `self.eval_dict`. It's dead code, because every config uses deterministic eval.
- `ppo.py:254,284` — `3499` is hard-coded instead of `episode_length - 1`.
- `transformer_loss.py:74` (and fixed_window) — `.squeeze()` with no axis drops the env axis when a minibatch holds 1 env, so `debug: true` (2 envs, 2 minibatches) breaks for these two models.
- `transformer_loss.py:122,124` — ignores `policy_loss_scaling_factor` / `value_loss_scaling_factor` (both 1.0 in the configs, so there is no effect today).
- `max_discount_factor` is read but never used; only `min_discount_factor` applies.

## Verified correct
- GAE recursion and λ-return value target.
- Per-minibatch advantage normalization; clipped surrogate; entropy schedule.
- Recurrent-state alignment between collection and loss for LSTM/SRU (zero reset after done) and LSMM (0.5 reset after done; the heads use the pre-reset updated memory, same as `sample_action`).
- Transformer history semantics: "history before the current obs", reset after done.
- The minibatch shuffle uses the same permutation for every Transition leaf.
- The env's deferred reset of `n_steps`/`summed_reward` makes the episode-length and reward logging correct.

## Verification
CPU smoke run of all 7 configs with the old `visual_locomotion_mjx` package blocked from importing.
- **Run settings:** 4 envs, unroll 4, 2 minibatches of 2 envs, 2 epochs, 2 training iterations.
- **Script:** scratchpad `check_ppo.py`; not part of the repo.

| Config | first-minibatch `approx_kl` | steps after 1 iteration | KL-stop forced: steps / params | `train()` returns / latest ckpt |
|---|---|---|---|---|
| lstm | 0 | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| sru | 0 | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| lsmm | 0 | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| lsmm_additive | 0 | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| lsmm_deployment | 0 | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| fixed_window | 1.6e-4 (expected, #2) | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |
| transformer | 4.6e-5 (expected, #2) | 16 ✓ | 32 ✓ / unchanged ✓ | 32 / 32 ✓ |

- An exact 0 KL for the recurrent models confirms that collection and loss agree before the first update.
- `linear_velocity_target`: with yaw +90° and world velocity (1, 0, 0), it returns body (0, −1) plus the yaw rate, as expected.
- **Not covered:** a full-scale GPU run and eval. Eval still needs the eval maze files (Phase F).