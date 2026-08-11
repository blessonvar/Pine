"""ONE visual language for every figure in the dissertation.

Nothing else in hpp/plots sets a colour, a font size or a figure size: they call
use_style(), new_fig(), finalize() and save(). That is what makes 20 figures across
4 models look like they came from the same paper and it is enforced, not aspirational:
`grep set_title hpp/plots/*.py` must never match anything.

Conventions:
  * a method has exactly one colour+marker everywhere (METHODS), so a reader learns the
    legend once
  * every figure is standalone and single-panel no 1xN subplot rows
  * no on-image titles: captions live in the dissertation LaTeX via \\caption, not baked
    into the PNG. finalize() still draws a small grey subtitle ("how to read it"), it is
    the title that's gone
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

# ink
INK = "#1F2933"
MUTED = "#7B8794"
GRIDC = "#DDE2E7"

PALETTE = {
    "hpp": "#1D6FB8",
    "reap": "#D85A30",
    "slimmoe": "#1D9E75",
    "baseline": "#6B7280",
    "accent": "#7A4FB5",
    "original": "#D81B60",
    "dropped": "#C9CDD4",
}

METHODS = {
    "hpp": dict(label="HPP", color=PALETTE["hpp"], marker="o", ls="-"),
    "reap": dict(label="REAP", color=PALETTE["reap"], marker="s", ls="-"),
    "slimmoe": dict(label="SlimMoE", color=PALETTE["slimmoe"], marker="^", ls="-"),
}

# canonical sizes: pick one, never invent a new one inline
FIG = (5.8, 4.0)        # the default single-panel figure
WIDE = (9.6, 3.4)       # per-layer bars
SQUARE = (7.2, 5.8)     # heatmaps
STACK = (6.4, 5.8)      # two stacked panels sharing an x axis

SUB_SIZE = 8.5           # subtitle font size
SUB_PAD = 5              # points from the axes top to the subtitle's bottom

RC = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.labelcolor": INK,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "grid.color": GRIDC,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
    "lines.markersize": 5.5,
    "lines.markeredgewidth": 1.1,
    "lines.markeredgecolor": "white",
}


def use_style():
    sns.set_theme(style="whitegrid", context="paper", rc=RC)
    plt.rcParams.update(RC)


def new_fig(figsize=FIG):
    use_style()
    return plt.subplots(figsize=figsize)


def finalize(ax, xlabel=None, ylabel=None, subtitle=None, legend=True, pct_x=False,
             pct_y=False, grid_axis="y"):
    """No on-image titles: captions live in the dissertation LaTeX. subtitle is the one
    exception, a small grey "how to read it" line. grid_axis=None for heatmaps."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if subtitle:
        # offset in POINTS, not axes fraction -> same gap above the axes regardless of
        # axes height
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction",
                    xytext=(0, SUB_PAD), textcoords="offset points",
                    fontsize=SUB_SIZE, color=MUTED, va="bottom", ha="left",
                    annotation_clip=False)
    if pct_x:
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    if pct_y:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    if grid_axis:                       # None -> heatmaps, which must not draw a grid
        ax.grid(True, axis=grid_axis)
        ax.set_axisbelow(True)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
    return ax


def save(fig, cfg, name):
    d = Path(cfg.fig_dir)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.png")
    return d / f"{name}.png"


def method_style(cfg, method, alpha=None, beta=None):
    """One colour per method. HPP ablations (alpha/beta != default) keep HPP's colour but
    go dashed + translucent, so the default curve is always the readable one."""
    st = dict(METHODS.get(method, dict(label=method.upper(), color=INK, marker="o", ls="-")))
    if method == "hpp":
        import pandas as pd
        is_default = ((alpha is None or pd.isna(alpha) or alpha == cfg.alpha)
                      and (beta is None or pd.isna(beta) or beta == cfg.beta))
        if not is_default:
            st.update(ls="--", alpha=0.75, label=f"HPP ($\\alpha$={alpha:g}, $\\beta$={beta:g})")
    return st