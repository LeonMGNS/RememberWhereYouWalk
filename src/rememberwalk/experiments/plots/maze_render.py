"""Panel assembly for the MuJoCo-rendered maze-prediction figure and the data it reads.

Rows of the figure are [Groundtruth | one per approach], columns are timesteps
(or transposed); every panel is a MuJoCo render made by prediction_figure.py.
"""
import gc
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from rememberwalk.experiments.plots.style import RC, TEXTWIDTH_IN, FIGWIDTH_IN, FS_SCRIPT, png_path


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def load_prediction_data(data_dir):
    """Load one run's episode_lengths, predicted/true distributions and poses."""
    print(f"Loading {data_dir}")
    with open(f"{data_dir}/eval_data.pkl", "rb") as f:
        data = pickle.load(f)

    metrics = data["metrics"]

    def grab(key):
        if key not in metrics:
            raise KeyError(
                f"{key!r} not in metrics of {data_dir}. "
                f"Available: {sorted(metrics)}"
            )
        return np.asarray(metrics[key])

    out = {
        "episode_lengths": np.asarray(data["episode_lengths"]),
        "predicted": grab("predicted_distribution"),
        "true": grab("true_distribution"),
        "pos": grab("pos"),
    }
    del data
    gc.collect()
    return out


def infer_grid(D):
    """R, C from a flattened (R+1)*(C+1)*2 description, assuming square."""
    n = np.sqrt(D / 2)
    if not np.isclose(n, round(n)):
        raise ValueError(
            f"cannot infer a square maze from D={D}; pass R and C explicitly"
        )
    n = int(round(n))
    return n - 1, n - 1


def _autocrop_bbox(img, thresh=12):
    """Bounding box of the non-black content of one render."""
    lum = np.asarray(img)[..., :3].max(axis=2)
    rows = np.flatnonzero((lum > thresh).any(axis=1))
    cols = np.flatnonzero((lum > thresh).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return 0, img.shape[0], 0, img.shape[1]
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1


def plot_render_grid(images, names, timesteps, out_path,
                     figwidth_in=None, autocrop=True, pad_px=6,
                     control_freq_hz=None,
                     traj_cmap="plasma", traj_label="Trajectory time",
                     show_traj_legend=True,
                     wall_prob_colors=None,     # list of 0-255 RGB per approach
                     wall_prob_floor=(150, 150, 150),
                     scale=0.85,
                     time_on_x=False):          # transpose: timesteps as columns,
                                                # Groundtruth + approaches as rows               # panel block width as a share of
                                                # the page; <1 shrinks the images
                                                # (and thus the height), leaving
                                                # centred side margins
    """images : list of rows, each a list of HxWx3 arrays, columns being
    [True | one per approach] in the same order as names.

    Spans the full IEEE text width (figure*) by default; every text element is
    footnote size. All panels are cropped with the same bbox (taken from the
    first image -- the camera is fixed, so the maze occupies the same pixels in
    every render).
    """
    # input is always images[time][Groundtruth + approaches]
    n_time, n_series = len(images), 1 + len(names)
    assert all(len(row) == n_series for row in images), \
        f"each row needs {n_series} images (Groundtruth + {len(names)} approaches)"
    if time_on_x:
        nrows, ncols = n_series, n_time
        if figwidth_in is None:
            figwidth_in = FIGWIDTH_IN      # 3 time columns fit a single column
    else:
        nrows, ncols = n_time, n_series

    if autocrop:
        r0, r1, c0, c1 = _autocrop_bbox(images[0][0])
        r0 = max(r0 - pad_px, 0); c0 = max(c0 - pad_px, 0)
        r1 += pad_px; c1 += pad_px
        images = [[np.asarray(im)[r0:r1, c0:c1] for im in row] for row in images]

    h, w = images[0][0].shape[:2]
    fw = figwidth_in or TEXTWIDTH_IN
    panel_w = fw * scale / ncols
    fh = panel_w * (h / w) * nrows + 0.26      # + title strip

    with mpl.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols, figsize=(fw, fh),
                                 layout="constrained")
        margin = (1.0 - scale) / 2.0
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01,
                                    wspace=0.012, hspace=0.015,
                                    rect=(margin, 0, scale, 1))
        axes = np.atleast_2d(axes)
        def t_label(i):
            return (f"t = {timesteps[i] / control_freq_hz:.1f} s"
                    if control_freq_hz else f"t = {timesteps[i]}")

        series_labels = ["Groundtruth"] + list(names)
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i, j]
                # interpolation="none": embed the ORIGINAL pixels into vector
                # output. Any other value resamples the image to the save dpi
                # first -- the PDF default is the figure dpi of 100, which
                # collapses a 1280px render to ~110px per panel.
                src = images[j][i] if time_on_x else images[i][j]
                ax.imshow(src, interpolation="none")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
        if time_on_x:
            for i, lab in enumerate(series_labels):
                axes[i, 0].set_ylabel(lab, fontsize=FS_SCRIPT)
            for j in range(ncols):
                axes[0, j].set_title(t_label(j), fontsize=FS_SCRIPT, pad=2.0)
        else:
            for i in range(nrows):
                axes[i, 0].set_ylabel(t_label(i), fontsize=FS_SCRIPT)
            for j, lab in enumerate(series_labels):
                axes[0, j].set_title(lab, fontsize=FS_SCRIPT, pad=2.0)

        if show_traj_legend:
            # Horizontal colorbar under the grid explaining the trajectory
            # gradient. Ends are labelled in seconds when the control
            # frequency is known, in steps otherwise. The mappable is built
            # from the same normalisation draw_trajectory uses (0 .. T_end),
            # so the bar's ends are exactly the line's start/end colours.
            import matplotlib.cm as _cm
            t_end = max(timesteps)
            if control_freq_hz:
                vmax, unit = t_end / control_freq_hz, "s"
            else:
                vmax, unit = t_end, "steps"
            cm_obj = (plt.get_cmap(traj_cmap)
                      if isinstance(traj_cmap, str) else traj_cmap)
            sm = _cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin=0.0, vmax=vmax),
                cmap=cm_obj)
            cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                                orientation="horizontal",
                                fraction=0.035, pad=0.012, shrink=0.42,
                                aspect=45, anchor=(0.5, 1.0))
            cbar.set_ticks([0.0, vmax])
            cbar.set_ticklabels([f"t = 0 {unit}", f"t = {vmax:.0f} {unit}"])
            cbar.set_label(traj_label, fontsize=FS_SCRIPT, labelpad=1.5)
            cbar.ax.tick_params(labelsize=FS_SCRIPT, length=1.5, width=0.4)
            cbar.outline.set_linewidth(0.4)

            if wall_prob_colors:
                # Saturation ramp below the time bar: one thin row per
                # approach, each running floor colour -> full approach colour,
                # i.e. exactly the lerp the panels use for wall certainty.
                # Positioned manually relative to the settled colorbar, so it
                # sits clear of the time bar's tick labels.
                fig.canvas.draw()
                cpos = cbar.ax.get_position()
                gap = cpos.height * 2.6          # clears the time-bar ticks
                rh = cpos.height * len(wall_prob_colors) * 0.55
                rax = fig.add_axes([cpos.x0, cpos.y0 - gap - rh - 0.055,
                                    cpos.width, rh])
                t = np.linspace(0.0, 1.0, 256)
                floor = np.asarray(wall_prob_floor, float)
                ramp = np.stack([
                    floor[None, :] + t[:, None] * (np.asarray(c, float) - floor)
                    for c in wall_prob_colors
                ]) / 255.0
                rax.imshow(ramp, aspect="auto", extent=[0, 1, 0, 1],
                           interpolation="bilinear")
                rax.set_yticks([])
                rax.set_xticks([0.0, 0.5, 1.0])
                rax.set_xticklabels(["0", "0.5", "1"])
                rax.tick_params(labelsize=FS_SCRIPT, length=1.5, width=0.4,
                                pad=1.0)
                rax.set_xlabel("Predicted wall probability",
                               fontsize=FS_SCRIPT, labelpad=1.5)
                for sp in rax.spines.values():
                    sp.set_linewidth(0.4)

        fig.savefig(out_path, dpi=400)
        fig.savefig(png_path(out_path), dpi=400)
        plt.close(fig)
    return out_path


def select_environment(data, seed=0, env_index=None):
    """One env that ran full length in every approach (shared reset assumed)."""
    names = list(data.keys())
    T = min(d["predicted"].shape[0] for d in data.values())
    full = np.ones(np.asarray(data[names[0]]["episode_lengths"]).shape[0], bool)
    for n in names:
        full &= (np.asarray(data[n]["episode_lengths"]) >= T)
    cand = np.flatnonzero(full)
    if env_index is None:
        if cand.size == 0:
            raise SystemExit("no environment ran full length in ALL approaches; "
                             "pass env_index explicitly")
        env_index = int(np.random.default_rng(seed).choice(cand))
    print(f"using environment {env_index} ({cand.size} full-length candidates)")
    return env_index, T
