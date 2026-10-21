"""Live maze-prediction video: every approach's showcase run, side by side.

record_live() rolls out one approach exactly like data_creation (same eval mazes,
same reset keys, same callback -> step order) but as a Python loop, so at every
step the panel for the showcase env can be rendered: the predicted maze
(thresholded, painted in the approach colour), the GO2 at its TRUE current
configuration (full qpos: base pose AND joint angles), and the trajectory so far.
Because the rollout matches data_creation, env_index is the same maze for every
approach. At the showcase timesteps the full-resolution frame is also saved as a
screenshot.

assemble_video() then merges the per-approach recordings into one 1920x1080 video:
[Groundtruth | approach panels], titled, with a running timestamp.

For bitwise-reproducible reruns set XLA_FLAGS="--xla_gpu_deterministic_ops=true".

Outputs per approach in <output_dir>/videos/<approach>/: panel.mp4 (square, padded
with the last frame to full episode length so all recordings align), groundtruth.png,
groundtruth_hires.png, screenshots/t*.png, screenshots/timesteps.json, meta.json.
"""
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import cv2
import mujoco

from rememberwalk.experiments.rollout import build_environment, build_eval_dict, load_policy, reset_environments
from rememberwalk.experiments.suite import experiment_config
from rememberwalk.experiments.plots.maze_render import infer_grid, sigmoid
from rememberwalk.experiments.prediction_figure import (
    description_to_hfield, make_renderer, draw_trajectory,
    WALL_THRESHOLD, FLOOR_RGB, TRUE_RGB, RENDER_PX,
)
from rememberwalk.terrain.MazeEvaluationTerrainGenerator import MazeEvaluationTerrainGenerator
from rememberwalk.utils.maze_visual_runtime import MazePainter

PANEL_PX = 512          # square panel resolution recorded per VIDEO frame
SHOT_PX = RENDER_PX     # screenshot resolution = figure resolution (1280)
FRAME_STRIDE = 1        # record every n-th sim step
CONTROL_FREQ_HZ = 50


def render_live_panel(model, renderer, painter, hfield_flat, wall_rgb,
                      qpos, path_xy, step, total_steps):
    """One panel: given maze geometry/colour, robot at its FULL current qpos
    (base + all joints), trajectory so far."""
    model.hfield_data[:] = hfield_flat
    mujoco.mjr_uploadHField(model, renderer._mjr_context, 0)
    painter.wall_rgb = tuple(wall_rgb)
    painter.floor_rgb = tuple(FLOOR_RGB)
    painter.paint(model, renderer=renderer, force=True)

    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos)          # actual configuration, verbatim
    mujoco.mj_forward(model, data)
    mujoco.mj_camlight(model, data)
    renderer.update_scene(data, camera=0)
    img = np.ascontiguousarray(renderer.render())
    if step > 0:
        img = draw_trajectory(img, path_xy, upto=step, total_steps=total_steps)
    return img


def record_live(suite, approach, out_dir):
    """Record the live panel of one approach's showcase run into out_dir."""
    out_dir = str(out_dir)
    run_path = approach.runs[approach.showcase_run]
    env_index = suite.showcase_env_index
    stride = int(suite.video.get("frame_stride", FRAME_STRIDE))
    nr_experiment_envs = suite.nr_envs
    shots_cfg = list(suite.showcase_timesteps)
    os.makedirs(out_dir, exist_ok=True)
    shot_dir = os.path.join(out_dir, "screenshots")
    if shots_cfg:
        os.makedirs(shot_dir, exist_ok=True)
    if not 0 <= env_index < nr_experiment_envs:
        raise ValueError(f"env_index {env_index} outside the "
                         f"{nr_experiment_envs}-env batch")

    # ---- same environment, reset keys and eval dict as data_creation ------
    config = experiment_config(suite, approach, run_path)
    environment, setup = build_environment(config)
    env_setup = setup["env"]
    training_setup = setup["training"]
    actor_critic = training_setup["actor_critic"]
    T = int(environment.episode_length)
    shots = sorted({min(max(s, 0), T - 1) for s in shots_cfg})
    for want in shots_cfg:
        got = min(max(want, 0), T - 1)
        if want != got:
            print(f"   screenshot step {want} clipped to {got} (T={T})")

    state, env_keys = reset_environments(environment, nr_experiment_envs, suite.randomization_key,
                                         suite.curriculum_level, suite.terrain_curriculum_level)
    eval_dict = build_eval_dict(training_setup, state, nr_experiment_envs, environment.action_size)
    print(f"Restoring policy {run_path}")
    eval_dict["parameters"] = load_policy(setup, run_path)

    callback = training_setup["experiment_callback"]
    callback_jit = jax.jit(callback, static_argnums=(0, 2))
    step_vmapped = jax.jit(jax.vmap(environment.step))

    # ---- groundtruth + wall generator -------------------------------------
    true_desc = np.asarray(state.maze_description[env_index], float).reshape(-1)
    R, C = infer_grid(true_desc.size)
    print(f"maze {R}x{C}, T={T}, env {env_index}/{nr_experiment_envs}, "
          f"stride {stride}, screenshots {shots}")

    # Only used for generate_maze_walls inside description_to_hfield -- same
    # trick as render_maze_predictions.py, which is known-good.
    wall_generator = MazeEvaluationTerrainGenerator(
        env_setup["terrain_parameters"],
        jnp.asarray(true_desc.reshape(R + 1, C + 1, 2)))

    # ONE renderer for everything. Two renderers = two GL contexts, and
    # mjr_uploadHField / the painter's texture upload only reach the context
    # that is current -- the screenshot context then renders with a stale
    # wall texture (walls added since its last use stay floor-grey). So:
    # render at figure resolution, downscale for the video.
    render_px = SHOT_PX if shots else PANEL_PX
    renderer = make_renderer(environment.model, render_px)
    painter = MazePainter(environment.model)

    def to_video(img):
        if img.shape[0] == PANEL_PX and img.shape[1] == PANEL_PX:
            return img
        return cv2.resize(img, (PANEL_PX, PANEL_PX),
                          interpolation=cv2.INTER_AREA)

    hf_true = description_to_hfield(wall_generator, true_desc, R, C)
    park = np.zeros(environment.model.nq)
    park[2] = -50.0
    park[3] = 1.0
    gt = render_live_panel(environment.model, renderer, painter, hf_true,
                           TRUE_RGB, park, np.zeros((1, 2)), 0, T)
    cv2.imwrite(f"{out_dir}/groundtruth.png", to_video(gt)[..., ::-1])
    cv2.imwrite(f"{out_dir}/groundtruth_hires.png", gt[..., ::-1])

    # ---- rollout: data_creation's eval_step as a Python loop --------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(f"{out_dir}/panel.mp4", fourcc,
                         CONTROL_FREQ_HZ / stride, (PANEL_PX, PANEL_PX))
    path = np.zeros((T, 2), np.float32)
    wall_rgb = approach.color
    key = env_keys[-1]
    frames = 0
    done_step = None
    last_img = None
    last_hf = None            # last (hfield, qpos, step) for early-done padding
    last_qpos = None
    last_t = 0
    shot_set = set(shots)

    for t in range(T):
        # callback on the CURRENT state -> action + metrics (incl. the
        # prediction), carrying memory/hidden state inside eval_dict, exactly
        # as inside data_creation's scan body.
        eval_dict, metrics = callback_jit(actor_critic, eval_dict, environment)
        cur_state = eval_dict["state"]

        qpos = np.asarray(cur_state.pipeline_state.qpos[env_index])
        path[t] = qpos[:2]

        need_video = (t % stride == 0)
        need_shot = t in shot_set
        if need_video or need_shot:
            p = np.asarray(metrics["predicted_distribution"][env_index], float)
            if p.min() < 0 or p.max() > 1:
                p = sigmoid(p)
            hf = description_to_hfield(
                wall_generator, (p > WALL_THRESHOLD).astype(float), R, C)
            last_hf, last_qpos, last_t = hf, qpos, t
            img = render_live_panel(environment.model, renderer, painter,
                                    hf, wall_rgb, qpos, path, t, T)
            if need_video:
                last_img = to_video(img)
                vw.write(last_img[..., ::-1])
                frames += 1
            if need_shot:
                cv2.imwrite(f"{shot_dir}/t{t:05d}.png", img[..., ::-1])
                print(f"  screenshot t={t} -> {shot_dir}/t{t:05d}.png")

        state_keys = jax.random.split(key, nr_experiment_envs)
        next_state, state_keys = step_vmapped(cur_state, eval_dict["action"],
                                              state_keys)
        eval_dict["state"] = next_state
        key = state_keys[-1]

        if t % 500 == 0:
            print(f"  step {t}/{T}")
        if bool(np.asarray(next_state.done[env_index])):
            done_step = t
            print(f"  env {env_index} done at step {t}; freezing last frame")
            break

    # pad the video to full length so all four approaches stay frame-aligned
    total_frames = (T + stride - 1) // stride
    while frames < total_frames and last_img is not None:
        vw.write(last_img[..., ::-1])
        frames += 1

    # screenshots requested past an early termination: frozen at the final
    # prediction/pose, trajectory ending at done_step (same normalisation)
    if last_hf is not None:
        for s in shots:
            if done_step is not None and s > done_step:
                shot = render_live_panel(environment.model, renderer,
                                         painter, last_hf, wall_rgb,
                                         last_qpos, path, last_t, T)
                cv2.imwrite(f"{shot_dir}/t{s:05d}.png", shot[..., ::-1])
                print(f"  screenshot t={s} FROZEN at done step {done_step}")

    vw.release()
    renderer.close()

    meta = {"name": approach.name, "color": list(approach.color),
            "checkpoint": run_path, "env_index": env_index,
            "T": T, "stride": stride, "frames": frames,
            "done_step": done_step,
            "fps": CONTROL_FREQ_HZ / stride,
            "control_freq_hz": CONTROL_FREQ_HZ,
            "panel_px": PANEL_PX, "shot_px": render_px,
            "save_screenshots": shots}
    with open(f"{out_dir}/meta.json", "w") as f:
        json.dump(meta, f)
    if shots:
        with open(f"{shot_dir}/timesteps.json", "w") as f:
            json.dump({"timesteps": shots, "T": T,
                       "control_freq_hz": CONTROL_FREQ_HZ}, f)
    print(f"wrote {out_dir}/panel.mp4 ({frames} frames), groundtruth[_hires]"
          f".png, {len(shots)} screenshots, meta.json")


# --------------------------------------------------------------------------
# Assembly of the per-approach recordings into one video
# --------------------------------------------------------------------------
W, H = 1920, 1080
BG = 255
TITLE_BAND = 52
TIME_BAND = 46
GAP = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX


def fit(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def put_center(frame, text, cx, cy, scale, color=(20, 20, 20), thick=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    cv2.putText(frame, text, (int(cx - tw / 2), int(cy + th / 2)),
                FONT, scale, color, thick, cv2.LINE_AA)


def compose(gt, panels, names, colors, t_sec, panel_px):
    n = 1 + len(panels)
    frame = np.full((H, W, 3), BG, np.uint8)
    x0 = (W - (n * panel_px + (n - 1) * GAP)) // 2
    y0 = (H - (TITLE_BAND + panel_px + TIME_BAND)) // 2 + TITLE_BAND

    tiles = [("Groundtruth", (40, 40, 40), gt)] + [
        (nm, c, p) for nm, c, p in zip(names, colors, panels)]
    for k, (name, color, img) in enumerate(tiles):
        x = x0 + k * (panel_px + GAP)
        frame[y0:y0 + panel_px, x:x + panel_px] = fit(img, panel_px)
        put_center(frame, name, x + panel_px / 2, y0 - TITLE_BAND / 2, 0.72)
        cv2.rectangle(frame, (x - 2, y0 - 2),
                      (x + panel_px + 1, y0 + panel_px + 1),
                      tuple(int(v) for v in color), 2)   # frame is RGB until it is written
    put_center(frame, f"t = {t_sec:6.2f} s", W / 2,
               y0 + panel_px + TIME_BAND / 2, 0.8)
    return frame


def assemble_video(dirs, out_path):
    """Merge the recordings in dirs (display order) into one video at out_path."""
    dirs = [str(d) for d in dirs]
    metas = []
    for d in dirs:
        with open(os.path.join(d, "meta.json")) as f:
            metas.append(json.load(f))
    names = [m["name"] for m in metas]
    colors = [tuple(m["color"]) for m in metas]
    fps = metas[0]["fps"]
    if len({m["fps"] for m in metas}) > 1:
        raise SystemExit(f"recordings differ in fps: {[m['fps'] for m in metas]}")
    if len({m["env_index"] for m in metas}) > 1:
        raise SystemExit("recordings are from different env_index values: "
                         f"{[m['env_index'] for m in metas]}")
    if len({m["T"] for m in metas}) > 1:
        raise SystemExit(f"recordings differ in T: {[m['T'] for m in metas]}")

    gts = [cv2.imread(os.path.join(d, "groundtruth.png"))[..., ::-1]
           for d in dirs]
    for k in range(1, len(dirs)):
        if gts[k].shape != gts[0].shape:
            raise SystemExit(
                f"groundtruth of {dirs[k]} has shape {gts[k].shape}, "
                f"expected {gts[0].shape} -- recordings incompatible")
        diff = float(np.abs(gts[k].astype(int) - gts[0].astype(int)).mean())
        if diff > 3.0:
            raise SystemExit(
                f"groundtruth of {dirs[k]} differs from "
                f"{dirs[0]} (mean abs diff {diff:.1f}) -- the "
                f"rollouts are not simulating the same maze")

    caps = [cv2.VideoCapture(os.path.join(d, "panel.mp4")) for d in dirs]
    n_frames = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps)

    n = 1 + len(dirs)
    panel_px = min((W - (n - 1) * GAP - 40) // n,
                   H - TITLE_BAND - TIME_BAND - 40)
    os.makedirs(os.path.dirname(str(out_path)) or ".", exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         fps, (W, H))
    for i in range(n_frames):
        panels = []
        for c in caps:
            ok, bgr = c.read()
            if not ok:
                break
            panels.append(bgr[..., ::-1])
        if len(panels) < len(dirs):
            break
        frame = compose(gts[0], panels, names, colors, i / fps, panel_px)
        vw.write(frame[..., ::-1])
        if i % 500 == 0:
            print(f"  frame {i}/{n_frames}")
    vw.release()
    for c in caps:
        c.release()
    print(f"wrote {out_path}: {n_frames} frames, {W}x{H} @ {fps:g} fps "
          f"({n_frames / fps:.1f} s)")
