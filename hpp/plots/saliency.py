"""Diagnostics: what the three saliency signals actually look like, and what a given
run decided to keep."""
from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns

from hpp.plots.style import INK, MUTED, FIG, PALETTE, SQUARE, WIDE, finalize, save, use_style


def _tick_step(n, target=10):
    """Label every row/column on a small heatmap; thin out to ~target labels on a large one,
    so 24-64 layers/experts don't render as an unreadable wall of digits."""
    return max(1, round(n / target))


def plot_layer_saliency(cfg, scores):
    import matplotlib.pyplot as plt

    use_style()
    imp = np.asarray(scores["layer_imp"], float)
    total = imp.sum()
    if total <= 0:
        print("  [WARN] layer_imp is all zeros -- using uniform weights")
        norm = np.ones_like(imp) / len(imp)
    else:
        norm = imp / total
    df = pd.DataFrame({"layer_idx": np.arange(len(imp)),
                       "layer_id": [cfg.idx2lid[i] for i in range(len(imp))],
                       "layer_importance": imp, "normalized": norm})
    df.to_csv(cfg.log_dir / f"layer_saliency_{cfg.model_tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=WIDE)
    ax.bar(df["layer_id"], df["normalized"] * 100, color=PALETTE["hpp"],
           edgecolor="white", linewidth=0.6)
    ax.axhline(100 / len(imp), ls="--", lw=1.2, color=MUTED, label="uniform ($\\alpha$=0)")
    finalize(ax, xlabel="decoder layer (depth $\\rightarrow$)",
             ylabel="share of total influence (%)",
             subtitle="$1-\\cos(x,\\, x+\\mathrm{MoE(LN}(x)))$: how far the block moves the residual")
    save(fig, cfg, f"layer_saliency_{cfg.model_tag}")
    plt.show()
    return df


def plot_expert_saliency(cfg, scores):
    """The single expert-saliency heatmap. HPP and REAP share this exact ranking; the
    figure therefore shows what BOTH methods believe, and any difference in the results
    is attributable to allocation alone."""
    import matplotlib.pyplot as plt

    use_style()
    I = np.asarray(scores["expert_imp"], float)
    share = I / np.maximum(I.sum(axis=1, keepdims=True), 1e-12)
    rows = [{"layer_idx": l, "layer_id": cfg.idx2lid[l], "expert": e,
             "importance": I[l, e], "normalized_in_layer": share[l, e]}
            for l in range(I.shape[0]) for e in range(I.shape[1])]
    pd.DataFrame(rows).to_csv(cfg.log_dir / f"expert_saliency_{cfg.model_tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=SQUARE)
    sns.heatmap(share * 100, cmap="viridis", ax=ax, annot=False,
                xticklabels=_tick_step(share.shape[1]), yticklabels=_tick_step(share.shape[0]),
                cbar_kws={"label": "share of layer saliency (%)"})
    finalize(ax, xlabel="expert id", ylabel="layer index (depth $\\rightarrow$)",
             subtitle="$E[g\\cdot\\|f\\|\\,|\\,\\mathrm{active}]$ — drives the drop order and the BETA widths",
             legend=False, grid_axis=None)
    save(fig, cfg, f"expert_saliency_{cfg.model_tag}")
    plt.show()
    return pd.DataFrame(rows)


def plot_kept_heatmap(cfg, keep, tag):
    """Remaining capacity per (layer, expert). This is the picture of what HPP does that
    REAP cannot: a continuum of widths rather than a binary keep/drop mask."""
    import matplotlib.pyplot as plt

    use_style()
    M = np.zeros((cfg.num_moe_layers, cfg.num_routed))
    for (li, e), k in keep.items():
        M[li, e] = 100.0 * k.numel() / cfg.moe_inter
    dead = int((M == 0).sum())

    fig, ax = plt.subplots(figsize=SQUARE)
    sns.heatmap(M, cmap="RdYlGn", vmin=0, vmax=100, ax=ax, annot=False,
                xticklabels=_tick_step(M.shape[1]), yticklabels=_tick_step(M.shape[0]),
                cbar_kws={"label": "% of expert width kept"})
    finalize(ax, xlabel="expert id", ylabel="layer index (depth $\\rightarrow$)",
             subtitle=f"dark red = expert dropped entirely ({dead} of "
                      f"{cfg.num_moe_layers * cfg.num_routed})",
             legend=False, grid_axis=None)
    save(fig, cfg, f"kept_heatmap_{tag}")
    plt.show()
    return M


def plot_floors(cfg, floors, gamma):
    import matplotlib.pyplot as plt

    use_style()
    df = pd.DataFrame([{"layer_idx": li, "expert": e, "floor_channels": floors[(li, e)],
                        "floor_pct": 100.0 * floors[(li, e)] / cfg.moe_inter}
                       for (li, e) in floors])
    df.to_csv(cfg.log_dir / f"floors_{cfg.model_tag}_g{gamma:.3f}.csv", index=False)

    fig, ax = plt.subplots(figsize=FIG)
    # ONE shared bin grid for both series: separate bins() calls each auto-range over their
    # OWN subset, so a handful of near-identical drop candidates gets bins a few PIXELS wide
    # correctly counted, invisibly thin. bins here is 2.5-points wide, always.
    bins = np.linspace(0, 100, 41)
    surv = df[df["floor_pct"] >= 10]["floor_pct"]
    drop = df[df["floor_pct"] < 10]["floor_pct"]
    if not surv.empty:
        ax.hist(surv, bins=bins, color=PALETTE["hpp"], edgecolor="white", linewidth=0.5,
                alpha=0.9, label=f"needs real width ({len(surv)} experts)")
    if not drop.empty:
        ax.hist(drop, bins=bins, color=PALETTE["reap"], edgecolor="white", linewidth=0.5,
                label=f"drop candidates ({len(drop)} experts)")
    mean_v = df["floor_pct"].mean()
    ax.axvline(mean_v, color=INK, ls="--", lw=1.3, label=f"mean = {mean_v:.1f}% of D")
    ax.set_xlim(0, 100)
    finalize(ax, xlabel="channels needed to retain $\\gamma$ of the expert's own energy (% of D)",
             ylabel="experts",
             subtitle="an expert whose energy is concentrated demands little -> it becomes droppable")
    save(fig, cfg, f"floors_{cfg.model_tag}_g{gamma:.3f}")
    plt.show()
    return df