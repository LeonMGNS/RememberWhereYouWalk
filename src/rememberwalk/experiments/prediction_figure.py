"""Maze-prediction figure with real MuJoCo panels.

Every panel is a MuJoCo render. The groundtruth row shows the TRUE maze. Each
approach row shows that approach's PREDICTED maze, thresholded into walls and
painted in the approach's colour via MazePainter, with the GO2 at its recorded
(x, y, yaw) and the trajectory so far. Columns are timesteps.

Data comes from the eval_data.pkl of each approach's showcase run; nothing is
re-simulated. The rendering model must contain the maze_tex texture
(go2_mjx_navigation_plot.xml, set in the suite overrides) -- MazePainter refuses otherwise.
"""
import numpy as np
import jax
import jax.numpy as jnp
import mujoco

from rememberwalk.experiments.suite import experiment_config
from rememberwalk.experiments.plots.maze_render import (
    load_prediction_data, infer_grid, sigmoid,
    plot_render_grid, select_environment,
)
from rememberwalk.terrain.MazeEvaluationTerrainGenerator import MazeEvaluationTerrainGenerator
from rememberwalk.utils.generate_setup import generate_setup
from rememberwalk.utils.maze_visual_runtime import MazePainter

TRUE_RGB = (40, 40, 40)          # near-black walls for the true maze
FLOOR_RGB = (180, 180, 180)     # brighter floor; walls and dark line pop against it

RENDER_PX = 1280                 # square; camera 0 is the fixed top-down ortho.
                                 # Must fit the offscreen buffer set by
                                 # <visual><global offwidth offheight> in the
                                 # XML; make_renderer() falls back if not.
WALL_THRESHOLD = 0.4             # p > threshold -> wall, else no wall
CONTROL_FREQ_HZ = 50             # sim steps per second, for the time axis
WORLD_EXTENT = 5.0               # hfield half-size: world spans [-5, 5] m
# Trajectory gradient: plain dark -> light. Truncated short of pure white so
# the newest segment stays visible on the (now brighter) floor.
import matplotlib.pyplot as _plt

TRAJ_CMAP = _plt.get_cmap("copper")


def world_to_px(x, y, H, W, extent=WORLD_EXTENT):
    """World (x, y) -> pixel (col, row) for camera 0.

    The fixed top-down camera has xyaxes="0 1 0 -1 0 0" and an orthographic
    fovy equal to the maze extent, so image-right is +world-y and image-down is
    +world-x, with [-extent, extent] filling the frame exactly.
    """
    col = (y + extent) / (2 * extent) * (W - 1)
    row = (x + extent) / (2 * extent) * (H - 1)
    return int(round(col)), int(round(row))


def draw_trajectory(img, path_xy, upto, total_steps, extent=WORLD_EXTENT,
                    stride=4, thickness=None):
    """Overlay the path travelled so far, colour-graded by GLOBAL time.

    The gradient is normalised by total_steps, not by the drawn length, so a
    given colour means the same wall-clock time in every row -- the t=35s
    panel's line simply stops partway through the colormap.
    """
    import cv2
    cmap = TRAJ_CMAP
    H, W = img.shape[:2]
    if thickness is None:
        thickness = max(3, round(3 * W / 640))   # keep line weight at high res
    upto = int(min(upto, len(path_xy) - 1))
    idx = np.arange(0, upto + 1, stride)
    if idx.size == 0 or idx[-1] != upto:
        idx = np.append(idx, upto)
    pts = [world_to_px(path_xy[i, 0], path_xy[i, 1], H, W, extent) for i in idx]
    img = np.ascontiguousarray(img)
    for a in range(len(pts) - 1):
        frac = idx[a + 1] / max(total_steps - 1, 1)
        r, g, b, _ = cmap(float(frac))
        cv2.line(img, pts[a], pts[a + 1],
                 (int(255 * r), int(255 * g), int(255 * b)),
                 thickness, lineType=cv2.LINE_AA)
    return img


def yaw_to_quat(yaw):
    """MuJoCo order (w, x, y, z) for a rotation about z."""
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def description_to_hfield(generator, desc_flat, R, C):
    """The recipe from shift_on_initial_terrain -- build walls, transpose,
    flatten -- with one crucial difference: the clip to [0.005, 0.995] is
    applied ONLY to nonzero entries. A blanket clip turns excluded walls' 0
    into 0.005, which (a) grows a micro-ridge of geometry for every candidate
    wall and (b) makes the certainty ratio 0.005/0.005 = 1 on those cells, so
    the whole lattice rendered at full saturation. Zeros must stay zero."""
    desc = jnp.asarray(desc_flat, jnp.float32).reshape(R + 1, C + 1, 2)
    desc = jnp.where(desc > 0, jnp.clip(desc, 0.005, 0.995), 0.0)
    hf = np.asarray(generator.generate_maze_walls(desc))
    return hf.T.reshape(-1).astype(np.float32)


def make_renderer(model, px=RENDER_PX):
    """Renderer at the requested size, degrading if the offscreen buffer set
    in the XML (<visual><global offwidth/offheight>) is smaller."""
    for size in (px, 1024, 800, 640):
        try:
            rend = mujoco.Renderer(model, size, size)
            if size != px:
                print(f"** offscreen buffer too small for {px}px, using "
                      f"{size}px. Raise offwidth/offheight in the XML for "
                      f"full print resolution.")
            return rend
        except Exception as e:
            last = e
    raise last


def render_panel(model, renderer, painter, hfield_flat, wall_rgb,
                 pose=None, keyframe="home", color_values=None):
    """One MuJoCo render: given heightfield and wall colour; robot posed at
    (x, y, yaw) if pose is given, hidden below the floor otherwise.

    color_values (optional, per hfield cell in hfield_data order, 0..1)
    switches from flat wall colour to certainty shading: each cell is
    interpolated between the floor colour and wall_rgb by its value.
    """
    model.hfield_data[:] = hfield_flat
    # Same call order as the working render() in the maze video script.
    mujoco.mjr_uploadHField(model, renderer._mjr_context, 0)

    painter.floor_rgb = tuple(FLOOR_RGB)
    if color_values is not None:
        painter.paint_values(model, color_values, wall_rgb=tuple(wall_rgb),
                             renderer=renderer)
    else:
        painter.wall_rgb = tuple(wall_rgb)
        painter.paint(model, renderer=renderer, force=True)

    data = mujoco.MjData(model)
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
    mujoco.mj_resetDataKeyframe(model, data, max(kid, 0))
    if pose is not None:
        x, y, yaw = float(pose[0]), float(pose[1]), float(pose[2])
        data.qpos[0:2] = (x, y)
        data.qpos[3:7] = yaw_to_quat(yaw)
    else:
        data.qpos[2] = -50.0          # park the robot far below the floor
    mujoco.mj_forward(model, data)

    mujoco.mj_camlight(model, data)
    renderer.update_scene(data, camera=0)
    return renderer.render()


def make_prediction_figure(suite, out_path, seed=0):
    approaches = suite.approaches
    names = [a.name for a in approaches]

    # ---- data -------------------------------------------------------------
    data = {a.name: load_prediction_data(suite.data_dir(a, a.showcase_run)) for a in approaches}
    env_index, T = select_environment(data, seed=seed, env_index=suite.showcase_env_index)
    steps = [min(int(t), T - 1) for t in suite.showcase_timesteps]
    R, C = infer_grid(data[names[0]]["predicted"].shape[-1])
    print(f"maze grid {R}x{C}, timesteps {steps}, env {env_index}")

    def true_at(t):
        tr = data[names[0]]["true"]
        return np.asarray((tr[t] if tr.ndim == 3 else tr)[env_index], float)

    def pred_probs(name, t):
        p = np.asarray(data[name]["predicted"][t, env_index], float)
        if p.min() < 0 or p.max() > 1:
            p = sigmoid(p)
        return p

    def pred_binary(name, t):
        """p > WALL_THRESHOLD -> wall, else no wall."""
        return (pred_probs(name, t) > WALL_THRESHOLD).astype(float)

    # ---- environment: built ONCE with the true maze -----------------------
    # The maze geometry is the same for every approach, so the first one's config is used
    first = approaches[0]
    config = experiment_config(suite, first, first.runs[first.showcase_run], nr_envs=1)
    setup = generate_setup(config)
    env_setup = setup["env"]
    generator = MazeEvaluationTerrainGenerator(
        env_setup["terrain_parameters"],
        jnp.asarray(true_at(steps[0]).reshape(R + 1, C + 1, 2)))
    env_setup["imports"]["TerrainGenerator"] = generator
    env_setup["initial_curr_coeff"] = suite.curriculum_level
    env_setup["initial_terrain_curr_coeff"] = suite.terrain_curriculum_level

    env = env_setup["mode"](env_setup)
    key = jax.random.PRNGKey(0)
    env.reset(suite.curriculum_level, suite.terrain_curriculum_level, key)   # populate model buffers

    renderer = make_renderer(env.model, RENDER_PX)
    painter = MazePainter(env.model)                 # flip_rows default

    # ---- render all panels ------------------------------------------------
    images = []
    for t in steps:
        # True panel: maze only. The robot and its trajectory live in the
        # prediction panels, because each approach followed its own path.
        row = [render_panel(env.model, renderer, painter,
                            description_to_hfield(generator, true_at(t), R, C),
                            TRUE_RGB, pose=None)]
        for approach in approaches:
            name = approach.name
            pose = np.asarray(data[name]["pos"][t, env_index])
            img = render_panel(env.model, renderer, painter,
                               description_to_hfield(
                                   generator, pred_binary(name, t), R, C),
                               approach.color, pose=pose)
            path = np.asarray(data[name]["pos"][:, env_index, :2])
            img = draw_trajectory(img, path, upto=t, total_steps=T)
            row.append(img)
        images.append(row)
        print(f"rendered row t={t}")
    renderer.close()

    plot_render_grid(images, names, steps, str(out_path),
                     control_freq_hz=CONTROL_FREQ_HZ, traj_cmap=TRAJ_CMAP,time_on_x=True)
    print(f"wrote {out_path}")
