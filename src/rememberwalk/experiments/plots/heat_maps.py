"""Average time spent per maze cell: one pooled heat map per approach.

All maps share one normalisation and one colourbar, so equal colour means equal
time everywhere in the figure. Each panel title carries the KL divergence of the
visit distribution to uniform (lower = more uniform exploration).

Memory: cell_visits is (T, envs, cx, cy). Each run is reduced to its per-env final
visit maps at load time, so the big array never outlives its iteration.
"""
import textwrap

import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.transforms import Bbox

from rememberwalk.experiments.plots.style import RC, FIGWIDTH_IN, FS_SCRIPT, CONTROL_DT, png_path
from rememberwalk.experiments.plots.exploration import load_exploration_data

# Sequential map for the visit counts. mako_r is perceptually uniform, runs
# light -> dark (so low counts read as empty), and its blue-green end echoes
# the #0173b2 the BCE figure uses for the proposed method. rocket_r is the
# warm equivalent if you prefer to stay close to YlOrBr.
CMAP_NAME = "mako_r"
CMAP = sns.color_palette(CMAP_NAME, as_cmap=True)


def final_visit_counts(episode_lengths, cell_visits, full_episodes_only=True,
                       label=""):
    """Per-env final cumulative visit maps, (n_kept, cx, cy). Each env is read at
    its own final step, so several runs can be pooled before averaging."""
    cell_visits = np.asarray(cell_visits)
    episode_lengths = np.asarray(episode_lengths)
    T, n_envs = cell_visits.shape[0], cell_visits.shape[1]
    last = np.clip(episode_lengths, 1, T) - 1
    final = cell_visits[last, np.arange(n_envs)]
    if full_episodes_only:
        keep = episode_lengths == T
        n_kept = int(keep.sum())
        if n_kept == 0:
            print(f"** WARNING: no full-length episodes for {label}, keeping all envs")
        else:
            if n_kept < keep.size:
                print(f"   {label}: {n_kept}/{keep.size} envs ran the full {T} steps")
            final = final[keep]
    return np.asarray(final, dtype=np.float64)


def kl_to_uniform(mean_map):
    """KL(p || u): empirical visit distribution vs uniform over the cells.

    p is the mean map normalised to sum 1; u = 1/n_cells. Equal to
    log(n_cells) - H(p), zero iff visits are perfectly uniform, and finite even
    when some cells were never visited (0 log 0 = 0). The absolute per-cell
    count (e.g. 3500/25) cancels in the normalisation, so only the SHAPE of
    the distribution matters. Lower = more uniform = better exploration.
    """
    m = np.asarray(mean_map, float)
    total = m.sum()
    if total <= 0:
        return float("nan")
    p = (m / total).ravel()
    n = p.size
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] * n)))


def plot_visit_heatmaps_pooled(
    runs,
    out_path,
    title="",
    figwidth_in=None,      # None -> single column (3.38in)
    figheight_in=None,     # None -> keeps the cells square
    ncols=2,               # 2 -> 2x2; set 4 for a single row
    cmap=None,
    norm="linear",
    vmax=None,
    show_kl=True,
    short_kl=False,        # "D_KL = x" instead of "D_KL(p || u) = x", for narrow panels
    cbar_frac=0.13,        # colourbar share of the figure width
    wrap=18,
    cbar_label="Average time spent per cell (s)",
):
    """One pooled map per approach, KL-to-uniform in each title.

    runs : dict {approach_name: (cx, cy) mean map over the pooled envs}.
    """
    runs = {k: np.asarray(v, float) for k, v in runs.items()}
    if cmap is None:
        cmap = CMAP
    elif isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)

    names = list(runs.keys())
    maps = [np.asarray(runs[n], float) for n in names]
    hi = float(vmax if vmax is not None else max(m.max() for m in maps))
    if norm == "log":
        lo = min(float(m[m > 0].min()) for m in maps if (m > 0).any())
        cnorm = mcolors.LogNorm(vmin=max(lo, 1e-6), vmax=hi)
    else:
        cnorm = mcolors.Normalize(vmin=0.0, vmax=hi)

    width = figwidth_in or FIGWIDTH_IN
    nrows = int(np.ceil(len(names) / ncols))
    if figheight_in is not None:
        height = figheight_in
    else:
        # square cells: body width minus the colourbar, split over ncols, plus
        # a two-line title strip per row. Scaled slightly below full size so
        # the figure sits lower on the page; the maps shrink to keep their
        # square aspect and the centring below balances the freed width.
        map_w = width * (1.0 - cbar_frac) / ncols
        height = map_w * nrows * 0.86 + nrows * 0.20 + 0.04

    with mpl.rc_context(RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(width, height), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01,
                                    wspace=0.02, hspace=0.04)
        axes = np.atleast_1d(axes).ravel()

        im = None
        for ax, name, m in zip(axes, names, maps):
            im = ax.imshow(m.T, cmap=cmap, norm=cnorm, origin="lower",
                           interpolation="nearest", aspect="equal")
            label = textwrap.fill(name, wrap)
            if show_kl:
                # p = pooled visit distribution, u = uniform over the cells.
                # Detail belongs in the caption; the panel just needs the symbol.
                if short_kl:
                    label += f"\n$D_{{\\mathrm{{KL}}}} = {kl_to_uniform(m):.3f}$"
                else:
                    label += ("\n$D_{\\mathrm{KL}}(p\\,\\|\\,u) = "
                              f"{kl_to_uniform(m):.3f}$")
            ax.set_title(label, fontsize=FS_SCRIPT, pad=2.0)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
                sp.set_color("0.6")

        for ax in axes[len(names):]:      # unused cells in a partial grid
            ax.axis("off")

        cbar = fig.colorbar(im, ax=list(axes[:len(names)]),
                            fraction=cbar_frac * 0.42, pad=0.015, shrink=0.9)
        cbar.set_label(cbar_label, fontsize=FS_SCRIPT, labelpad=2.0)
        cbar.ax.tick_params(labelsize=FS_SCRIPT - 1, width=0.4, length=1.5)
        cbar.outline.set_linewidth(0.4)

        if title:
            fig.suptitle(title, fontsize=FS_SCRIPT)

        # --- centre the map block on a fixed-width page -------------------
        # The colourbar (and its label) extend only to the right of the maps,
        # so a tight crop leaves the maps left of centre once LaTeX centres
        # the figure. Measure the imbalance around the union of the map axes
        # and shift/shrink the layout so both margins match, then save the
        # FULL page (explicit Bbox overrides savefig.bbox="tight").
        fig.canvas.draw()
        inv = fig.dpi_scale_trans.inverted()
        W, H = fig.get_size_inches()

        def _imbalance():
            fig.canvas.draw()
            r = fig.canvas.get_renderer()
            b = [ax.get_window_extent(r).transformed(inv)
                 for ax in axes[:len(names)]]
            return (W - max(x.x1 for x in b)) - min(x.x0 for x in b)

        # Shifting the layout region also re-flows the colourbar, so a single
        # shot under-corrects; iterate the shift until both margins match.
        x0 = 0.0
        for _ in range(6):
            e = _imbalance()                # right overhang minus left
            if abs(e) < 0.01:
                break
            x0 = min(max(x0 + e / (2 * W), 0.0), 0.4)
            fig.get_layout_engine().set(rect=(x0, 0, 1 - x0, 1))
        page = Bbox([[0, 0], [W, H]])
        fig.savefig(out_path, bbox_inches=page)
        fig.savefig(png_path(out_path), dpi=400,
                    bbox_inches=page)
        plt.close(fig)
    return out_path


def make_heat_map_figure(suite, out_path):
    pooled = {}
    for approach in suite.approaches:
        finals = []
        for i in range(len(approach.runs)):
            ep, cv = load_exploration_data(suite.data_dir(approach, i))
            finals.append(final_visit_counts(ep, cv, label=f"{approach.name} run {i + 1}"))
            del cv
        stacked = np.concatenate(finals, axis=0)      # (n_pooled, cx, cy)
        print(f"   {approach.name}: pooled {stacked.shape[0]} envs over {len(finals)} runs")
        # visits -> seconds (50 Hz control); KL is scale-invariant, unchanged
        pooled[approach.name] = stacked.mean(axis=0) * CONTROL_DT

    if len(pooled) <= 4:
        plot_visit_heatmaps_pooled(pooled, out_path, ncols=2, figheight_in=2.0)   # the 2x2 paper figure
    else:
        plot_visit_heatmaps_pooled(pooled, out_path, ncols=3, wrap=12, short_kl=True)   # height keeps the cells square
    print(f"wrote {out_path}")
