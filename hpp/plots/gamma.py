"""The gamma calibration figures TWO standalone plots, one per link of the argument.

gamma is not tuned, it is read off the model, and the derivation is a chain of two
questions that share ONE quantity: how many experts survive per layer.

  Step 1 (cliff)  how much of the primary metric survives at each expert count?
                     [GPU, measured] m* = the FEWEST experts still above the collapse
                     threshold the edge of the cliff, not just the safest point tested.
  Step 2 (curve)  which gamma produces that expert count?    [offline, floors]
                     gamma* = read off at m*.

Both plots put survivors on the x axis and mark m*, and both print the final gamma* value
directly on the figure, so a reader only ever needs to look at ONE of the two to get the
answer the other is there to show the answer is not arbitrary. Kept as two standalone
figures (not stacked) on purpose: they were a single stacked figure once and the lower
title collided with the upper legend at readable panel heights.

Legends are for MULTI-series charts. Each of these has exactly one data series, so every
reference line (threshold, m*, gamma*) is labelled directly on the plot instead a legend
box here would just be repeating the annotations already on the canvas.
"""
from __future__ import annotations

from hpp.plots.style import FIG, INK, MUTED, PALETTE, finalize, save, use_style

# plain-language axis label for the metrics this project actually calibrates against;
# falls back to the raw metric name for anything else
_METRIC_LABEL = {"gsm8k": "GSM8K accuracy", "perplexity": "WikiText-2 perplexity",
                 "math500": "MATH-500 accuracy"}


def plot_gamma_cliff(cfg, drops, m_star, gamma_star, sparsity):
    """Step 1: what the model actually tolerates. Measured on GPU, one probe per point."""
    import matplotlib.pyplot as plt

    use_style()
    metric = cfg.calib_primary_metric
    lower = metric == "perplexity"
    d = drops.sort_values("survivors")
    v = d[metric].astype(float)
    best = v.min() if lower else v.max()
    thr = best / cfg.calib_collapse_frac if lower else best * cfg.calib_collapse_frac

    fig, ax = plt.subplots(figsize=FIG)
    ylo, yhi = float(v.min()), float(v.max())
    pad = max((yhi - ylo) * 0.35, abs(best) * 0.05, 1e-9)
    if lower:
        ax.axhspan(thr, yhi + pad, color=PALETTE["reap"], alpha=0.06, zorder=0)
        thr_xy, thr_va = (d["survivors"].iloc[0], thr), "bottom"
    else:
        ax.axhspan(ylo - pad, thr, color=PALETTE["reap"], alpha=0.06, zorder=0)
        thr_xy, thr_va = (d["survivors"].iloc[0], thr), "top"
    ax.set_ylim(ylo - pad, yhi + pad)
    ax.axhline(thr, ls="--", lw=1.3, color=PALETTE["reap"], zorder=2)
    ax.annotate(f"collapse threshold ({cfg.calib_collapse_frac:.0%} of best)", xy=thr_xy,
                xytext=(4, 4 if thr_va == "bottom" else -4), textcoords="offset points",
                fontsize=8.5, color=PALETTE["reap"], ha="left", va=thr_va)

    ax.plot(d["survivors"], v, "-o", color=PALETTE["accent"], zorder=3)
    for _, r in d.iterrows():
        ax.annotate(f"{r[metric]:.3g}", (r["survivors"], r[metric]),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=8, color=MUTED)

    ax.axvline(m_star, ls="-", lw=1.3, color=INK, alpha=0.4, zorder=1)
    ax.annotate(f"keep {m_star}/layer\n$\\to$ $\\gamma$ = {gamma_star:.3f}",
                xy=(m_star, yhi + pad), xytext=(6, -4), textcoords="offset points",
                fontsize=9, fontweight="semibold", color=INK, ha="left", va="top")

    finalize(ax, xlabel="experts kept per layer (survivors)",
             ylabel=_METRIC_LABEL.get(metric, metric),
             subtitle=f"whole-expert drop probe at s={sparsity:.0%} · "
                      f"{'lower' if lower else 'higher'} is better",
             legend=False)
    save(fig, cfg, f"gamma_cliff_{cfg.model_tag}_s{int(round(sparsity * 100))}")
    plt.show()
    return fig


def plot_gamma_curve(cfg, curve, m_star, gamma_star, sparsity):
    """Step 2: which gamma reproduces m*. Offline: floors only, no GPU."""
    import matplotlib.pyplot as plt

    use_style()
    c = curve.sort_values("avg_survivors")
    fig, ax = plt.subplots(figsize=FIG)
    ax.plot(c["avg_survivors"], c["gamma"], "-", color=PALETTE["hpp"], zorder=3)
    ax.axhline(gamma_star, ls=":", lw=1.1, color=MUTED, zorder=1)
    ax.axvline(m_star, ls="-", lw=1.3, color=INK, alpha=0.4, zorder=1)
    ax.annotate(f"m* = {m_star}", xy=(m_star, ax.get_ylim()[0]), xytext=(-6, 6),
                textcoords="offset points", fontsize=8.5, color=MUTED, ha="right", va="bottom")
    ax.scatter([m_star], [gamma_star], s=70, color=INK, zorder=5)
    ax.annotate(f"$\\gamma^*$ = {gamma_star:.3f}", (m_star, gamma_star),
                textcoords="offset points", xytext=(10, -12), fontsize=10,
                fontweight="semibold", color=INK)

    finalize(ax, xlabel="experts kept per layer (survivors)",
             ylabel="$\\gamma$ (energy retained per expert)",
             subtitle=f"s={sparsity:.0%} · higher $\\gamma$ $\\to$ larger floors "
                      f"$\\to$ fewer survivors",
             legend=False)
    save(fig, cfg, f"gamma_curve_{cfg.model_tag}_s{int(round(sparsity * 100))}")
    plt.show()
    return fig


def plot_gamma_calibration(cfg, curve, drops, m_star, gamma_star, sparsity):
    plot_gamma_cliff(cfg, drops, m_star, gamma_star, sparsity)
    plot_gamma_curve(cfg, curve, m_star, gamma_star, sparsity)