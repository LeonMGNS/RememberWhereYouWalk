"""Percentage of the maze explored: one bar per approach, pooled over its runs.

Memory note: cell_visits is (T, envs, cells_x, cells_y). Each run is reduced to a
single (n_envs,) array of per-env explored-percentages at load time and everything
else is dropped.
"""
import gc
import pickle
import textwrap

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from rememberwalk.experiments.plots.style import RC, FIGWIDTH_IN, FS_SCRIPT, png_path


def load_exploration_data(data_dir):
    """Load (episode_lengths, cell_visits) for a single run."""
    print(f"Loading {data_dir}")
    with open(f"{data_dir}/eval_data.pkl", "rb") as f:
        data = pickle.load(f)

    episode_lengths = np.asarray(data["episode_lengths"])
    cell_visits = np.asarray(data["metrics"]["cell_visits"])

    del data
    gc.collect()

    return episode_lengths, cell_visits


def percent_explored(episode_lengths, cell_visits, full_episodes_only=True,
                     label=""):
    """Reduce one run to a (n_envs,) array of percent-of-maze-explored.

    Counts are cumulative, so each env's total is read at its own final step
    rather than at the global last step. That is robust whether the buffer
    freezes or zeroes after an episode ends -- reading cell_visits[-1] blindly
    would silently report 0% for early-terminated envs if it zeroes.
    """
    cell_visits = np.asarray(cell_visits)
    episode_lengths = np.asarray(episode_lengths)
    T, n_envs = cell_visits.shape[0], cell_visits.shape[1]
    last = np.clip(episode_lengths, 1, T) - 1             # (n_envs,)
    final = cell_visits[last, np.arange(n_envs)]          # (n_envs, cx, cy)
    final = final.reshape(n_envs, -1)

    pct = 100.0 * np.mean(final >= 1, axis=-1)                # (n_envs,)

    if full_episodes_only:
        keep = episode_lengths == T
        n_kept = int(keep.sum())
        if n_kept == 0:
            print(f"** WARNING: no full-length episodes for {label}, keeping all envs")
        else:
            if n_kept < keep.size:
                print(f"   {label}: {n_kept}/{keep.size} envs ran the full {T} steps")
            pct = pct[keep]

    return pct


def plot_exploration_bars_pooled(
    runs,
    out_path,
    colors,
    title="",
    ylim=(0, 105),
    figwidth_in=None,
    figheight_in=1.3,
    bar_width=0.62,
    err="std",            # "std" | "sem" -- std shows spread; SEM shrinks with
                          # sqrt(n) and is nearly invisible at n ~ 3000
    annotate=True,
    wrap=13,
):
    """One bar per approach over the POOLED envs.

    runs   : dict {approach_name: (n_envs_pooled,)} of percent-explored values.
             A list/tuple of per-run arrays is also accepted and is concatenated here.
    colors : one RGB (0-1) per approach, in the order of runs.
    """
    runs = {k: (np.concatenate([np.asarray(a, float).ravel() for a in v])
                if isinstance(v, (list, tuple)) else np.asarray(v, float).ravel())
            for k, v in runs.items()}
    names = list(runs.keys())
    with mpl.rc_context(RC):
        fig, ax = plt.subplots(figsize=(figwidth_in or FIGWIDTH_IN,
                                        figheight_in), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01)

        xs = np.arange(len(names))
        for a, (name, pct) in enumerate(runs.items()):
            pct = np.asarray(pct, float)
            mean = float(pct.mean())
            spread = float(pct.std())
            if err == "sem":
                spread /= np.sqrt(max(pct.size, 1))
            ax.bar(xs[a], mean, width=bar_width,
                   color=colors[a],
                   edgecolor="0.15", linewidth=0.6, zorder=2)
            ax.errorbar(xs[a], mean, yerr=spread, fmt="none", ecolor="0.15",
                        elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3)
            if annotate:
                ax.text(xs[a], mean + spread + 1.5, f"{mean:.1f}",
                        ha="center", va="bottom", fontsize=FS_SCRIPT, zorder=4)

        ax.set_xticks(xs)
        if len(names) > 4:
            ax.set_xticklabels(names, rotation=30, ha="right", rotation_mode="anchor")
        else:
            ax.set_xticklabels([textwrap.fill(n, wrap) for n in names])
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(MultipleLocator(25))
        ax.set_ylabel("Explored cells (%)", fontsize=FS_SCRIPT)
        ax.grid(axis="y", color="0.85", linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if title:
            fig.suptitle(title, fontsize=FS_SCRIPT)

        fig.savefig(out_path)
        fig.savefig(png_path(out_path), dpi=400)
        plt.close(fig)
    return out_path


def make_exploration_figure(suite, out_path):
    # Reduce each run to (n_envs,) immediately; the big array never outlives
    # the loop iteration that loaded it.
    runs = {}
    for approach in suite.approaches:
        runs[approach.name] = []
        for i in range(len(approach.runs)):
            label = f"{approach.name} run {i + 1}"
            ep, cv = load_exploration_data(suite.data_dir(approach, i))
            # full_episodes_only=False: include early-terminated envs. Their
            # percentage is read at their own final step (cumulative counts), so a
            # run that died at t=500 contributes what it actually explored.
            runs[approach.name].append(percent_explored(ep, cv, full_episodes_only=False, label=label))
            n_full = int((np.asarray(ep) == cv.shape[0]).sum())
            print(f"   {label}: all {cv.shape[1]} envs included ({n_full} ran the full episode)")
            del cv

    plot_exploration_bars_pooled(runs, out_path, colors=[a.color_unit for a in suite.approaches])
    print(f"wrote {out_path}")
