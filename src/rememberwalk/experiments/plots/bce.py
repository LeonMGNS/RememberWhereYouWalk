"""Maze reconstruction error (BCE) over time: one curve per approach, pooled over its runs."""

import gc
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
from matplotlib.transforms import Bbox

from rememberwalk.experiments.plots.style import (
    CONTROL_DT,
    FIGWIDTH_IN,
    FS_FOOTNOTE,
    LINESTYLES,
    RC,
    png_path,
)


def _thousands(x, _pos):
    if x == 0:
        return "0"
    return f"{x:g}"


def load_bce_data(data_dir):
    """Load (episode_lengths, bce_error) for a single run.

    bce_error is (T, n_envs); episode_lengths is (n_envs,). Loads the full dict,
    extracts the two arrays and drops the rest so it can be collected before the
    next run is loaded.
    """
    print(f"Loading {data_dir}")
    with open(f"{data_dir}/eval_data.pkl", "rb") as f:
        data = pickle.load(f)

    episode_lengths = np.asarray(data["episode_lengths"])
    bce_error = np.asarray(data["metrics"]["bce_error"])

    del data
    gc.collect()

    return episode_lengths, bce_error


def full_episode_only(episode_lengths, bce_error, label=""):
    """Keep only envs whose episode ran the full length of the simulation.

    Returns (T, n_kept). Reports how many envs survived -- that count belongs
    in the figure caption, since the std band is taken over exactly these envs.
    """
    T = bce_error.shape[0]
    keep = np.asarray(episode_lengths) == T
    n_kept = int(keep.sum())
    if n_kept == 0:
        # The rollout writes 0 for every step after termination; counting those as a
        # perfect reconstruction would pull the mean down, so they are masked out.
        print(f"** WARNING: no full-length episodes for {label}, keeping all envs up to their termination")
        alive = np.arange(T)[:, None] < np.asarray(episode_lengths)[None, :]
        return np.where(alive, bce_error, np.nan)
    if n_kept < keep.size:
        print(f"   {label}: {n_kept}/{keep.size} envs ran the full {T} steps")
    return bce_error[:, keep]


def _smooth(y, ker, pad):
    """Moving average with edge padding, same length as y."""
    return np.convolve(np.pad(y, pad, mode="edge"), ker, mode="same")[pad : -pad or None]


def plot_bce_pooled(
    runs,
    out_path,
    colors,
    linestyles=None,
    steps=None,
    ylim=None,
    band="std",  # "std" | "sem" | "iqr"
    band_alpha=0.16,
    smooth=0,
    figwidth_in=None,
    figheight_in=None,
    title="",
):
    """One panel, one curve per approach, computed over the POOLED envs.

    runs   : dict {approach_name: (T, n_envs_pooled)} -- all runs of an approach
             concatenated along the env axis, so mean and band are taken over runs
             and environments jointly. A list/tuple of per-run (T, n_i) arrays is
             also accepted and concatenated here.
    colors : one RGB (0-1) per approach, in the order of runs.
    linestyles : optional dash pattern per approach (default: all solid).
    """
    runs = {
        k: (
            np.concatenate([np.atleast_2d(np.asarray(a, float)) for a in v], axis=1)
            if isinstance(v, (list, tuple))
            else np.asarray(v, float)
        )
        for k, v in runs.items()
    }

    # Dash patterns need a thinner stroke to stay readable where curves overlap
    linewidth = 1.3 if linestyles else 1.75
    lo_seen, hi_seen = [], []
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(
            figsize=(figwidth_in or FIGWIDTH_IN, figheight_in or 1.7), layout="constrained"
        )
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)

        for k, (name, arr) in enumerate(runs.items()):
            arr = np.asarray(arr, dtype=float)
            mean = np.nanmean(arr, axis=1)
            if band == "std":
                sdev = np.nanstd(arr, axis=1)
                lo, hi = mean - sdev, mean + sdev
            elif band == "sem":
                n_alive = np.maximum((~np.isnan(arr)).sum(axis=1), 1)
                sdev = np.nanstd(arr, axis=1) / np.sqrt(n_alive)
                lo, hi = mean - sdev, mean + sdev
            elif band == "iqr":
                lo = np.nanpercentile(arr, 25, axis=1)
                hi = np.nanpercentile(arr, 75, axis=1)
            else:
                raise ValueError(band)
            lo = np.maximum(lo, 0.0)  # BCE is non-negative

            if smooth > 1:
                ker = np.ones(smooth) / smooth
                pad = smooth // 2
                mean, lo, hi = (_smooth(y, ker, pad) for y in (mean, lo, hi))

            x = np.arange(mean.size) if steps is None else np.asarray(steps) * CONTROL_DT
            color = colors[k]
            ax.fill_between(x, lo, hi, color=color, alpha=band_alpha, linewidth=0)
            ax.plot(
                x,
                mean,
                color=color,
                label=name,
                linewidth=linewidth,
                linestyle=linestyles[k] if linestyles else "-",
            )
            lo_seen.append(float(np.nanmin(lo)))
            hi_seen.append(float(np.nanmax(hi)))

        ax.set_xlim(x[0], x[-1])
        if ylim is None:
            lo_d, hi_d = min(lo_seen), max(hi_seen)
            padv = 0.04 * (hi_d - lo_d)
            ylim = (lo_d - padv, hi_d + padv)
        ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(FuncFormatter(_thousands))
        ax.xaxis.set_major_locator(MultipleLocator(10))
        if title:
            fig.suptitle(title, fontsize=FS_FOOTNOTE)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, which="major", color="0.85", linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("Maze reconst. err.", fontsize=8)
        # Figure-level legend below the axes, CrossQ-paper style: horizontal
        # grid, thick colour swatches, no frame. Proxy handles so thickening
        # the swatches leaves the plotted curves untouched.
        handles = [
            Line2D(
                [],
                [],
                color=colors[k],
                linewidth=linewidth,
                solid_capstyle="butt",
                linestyle=linestyles[k] if linestyles else "-",
            )
            for k in range(len(runs))
        ]
        fig.legend(
            handles,
            list(runs.keys()),
            loc="outside lower center",
            ncol=2,
            fontsize=8,
            frameon=False,
            columnspacing=1.6,
            handlelength=3.0 if linestyles else 1.6,  # long enough to show a full dash period
            handletextpad=0.5,
            labelspacing=0.35,
        )

        # --- centre the axes box on a fixed-width page --------------------
        # The y-label and tick labels extend only leftwards, so the default
        # tight crop leaves the axes right of centre once LaTeX centres the
        # figure. Instead: measure the left overhead, shrink the layout so an
        # equal margin is left free on the right, and save the FULL page
        # (explicit Bbox overrides savefig.bbox="tight" from RC). The saved
        # width is then exactly the requested figure width, axes centred.
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        ab = ax.get_window_extent(fig.canvas.get_renderer()).transformed(inv)
        W, H = fig.get_size_inches()
        extra = ab.x0 - (W - ab.x1)  # left overhead minus right overhead
        if extra > 0.005:
            fig.get_layout_engine().set(rect=(0, 0, max(1 - extra / W, 0.5), 1))
        page = Bbox([[0, 0], [W, H]])
        fig.savefig(out_path, bbox_inches=page)
        fig.savefig(png_path(out_path), dpi=400, bbox_inches=page)
        plt.close(fig)
    return out_path


def make_bce_figure(suite, out_path, approaches=None, labels=None, colors=None, linestyles=None):
    """Writes the map reconstruction error (BCE) over time, pooled over runs, of ``approaches`` (default:
    all approaches of the suite) to ``out_path``.

    ``labels`` are optional display names, one per approach. ``colors`` (0-1 RGB) and ``linestyles``
    (names of ``style.LINESTYLES``) override the approach colours and the solid line.
    """
    approaches = approaches or suite.approaches
    labels = labels or [a.name for a in approaches]
    # One run at a time: only the filtered (T, n_kept) arrays stay alive.
    runs = {}
    for approach, label in zip(approaches, labels, strict=True):
        runs[label] = [
            full_episode_only(
                *load_bce_data(suite.data_dir(approach, i)), label=f"{approach.name} run {i + 1}"
            )
            for i in range(len(approach.runs))
        ]

    lengths = {a.shape[0] for v in runs.values() for a in v}
    T = min(lengths)
    if len(lengths) > 1:
        print(f"** runs differ in length {sorted(lengths)}, truncating to T={T}")
        runs = {k: [a[:T] for a in v] for k, v in runs.items()}

    legend_rows = int(np.ceil(len(runs) / 2))
    plot_bce_pooled(
        runs,
        out_path,
        colors=colors or [a.color_unit for a in approaches],
        linestyles=[LINESTYLES[s] for s in linestyles] if linestyles else None,
        # many overlapping bands would merge into one blob, even more so in a single colour
        band_alpha=0.16 if len(runs) <= 4 else (0.035 if linestyles else 0.07),
        figheight_in=1.7 + 0.17 * max(legend_rows - 2, 0),
        steps=np.arange(T),
        ylim=None,  # set e.g. (0.10, 0.45) to pin it across figures
        band="std",
        smooth=51,
    )
    print(f"wrote {out_path}")
