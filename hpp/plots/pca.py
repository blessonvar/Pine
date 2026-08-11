"""Functional-subspace figure (REAP Fig. 1 adapted to Original / REAP / HPP).

Each expert's "functional fingerprint" is its mean output vector over the calibration
tokens (dim = hidden). A single 2-D PCA is fitted on the ORIGINAL fingerprints per layer
so all three figures share coordinates (same xlim/ylim); each figure then highlights
which experts that method keeps. The reported PC1+PC2 cumulative variance mirrors REAP's
Table A4.

One standalone figure per method, like everywhere else in this package. No on-image
titles: see finalize() in hpp/plots/style.py.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hpp.allocator import select_hpp
from hpp.baselines import reap_keep
from hpp.plots.style import FIG, METHODS, PALETTE, finalize, new_fig


def _pca_2d(X):
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xc = X - X.mean(0, keepdims=True)
    if Xc.shape[0] < 2 or not np.isfinite(Xc).all():
        return np.zeros((X.shape[0], 2)), float("nan")
    try:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        vals, vecs = np.linalg.eigh(Xc @ Xc.T)
        order = np.argsort(vals)[::-1]
        coords = vecs[:, order[:2]] * np.sqrt(np.maximum(vals[order[:2]], 0.0))
        tot = max(float(np.maximum(vals, 0).sum()), 1e-12)
        return coords, float(np.maximum(vals[order[:2]], 0).sum() / tot)
    coords = Xc @ Vt[:2].T
    var = (S ** 2) / max(Xc.shape[0] - 1, 1)
    evr = var / max(var.sum(), 1e-12)
    return coords, float(evr[:2].sum())


def pca_functional_subspace(cfg, scores, sparsity=0.50, layers=None,
                            alpha=None, beta=None, gamma=None):
    layers = layers or (cfg.moe_layer_ids[0], cfg.moe_layer_ids[-1])

    rk, _, _ = reap_keep(cfg, scores, sparsity)
    reap_alive = {li: {e for e in range(cfg.num_routed) if rk[(li, e)].numel() > 0}
                  for li in range(cfg.num_moe_layers)}
    hk, _, _, _ = select_hpp(cfg, scores, sparsity, alpha, beta, gamma)
    hpp_alive = {li: {e for e in range(cfg.num_routed) if hk[(li, e)].numel() > 0}
                 for li in range(cfg.num_moe_layers)}

    rows = []
    for lid in layers:
        li = cfg.lid2idx[lid]
        X = np.stack([scores["fingerprint"][(li, e)] for e in range(cfg.num_routed)], 0)
        coords, evr_all = _pca_2d(X)
        sr = np.array(sorted(reap_alive[li]))
        sh = np.array(sorted(hpp_alive[li]))
        _, evr_r = _pca_2d(X[sr]) if sr.size > 2 else (None, np.nan)
        _, evr_h = _pca_2d(X[sh]) if sh.size > 2 else (None, np.nan)
        rows.append({"layer_id": lid, "evr_original": round(evr_all, 4),
                     "evr_reap": round(evr_r, 4), "evr_hpp": round(evr_h, 4),
                     "reap_survivors": int(sr.size), "hpp_survivors": int(sh.size)})

        # shared coordinate frame across the three standalone figures, so they stay
        # visually comparable even though each is its own plot
        pad_x = 0.06 * max(np.ptp(coords[:, 0]), 1e-9)
        pad_y = 0.06 * max(np.ptp(coords[:, 1]), 1e-9)
        xlim = (coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ylim = (coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)

        variants = [
            dict(name="original",
                 subtitle="PCA reference frame: every expert's fingerprint, before any pruning.",
                 surv=np.arange(cfg.num_routed), color=PALETTE["original"], marker="D",
                 kept_label=f"experts (n={cfg.num_routed})", show_dropped=False),
            dict(name="reap",
                 subtitle=f"grey = dropped. REAP keeps {sr.size}/{cfg.num_routed} whole experts.",
                 surv=sr, color=PALETTE["reap"], marker=METHODS["reap"]["marker"],
                 kept_label=f"kept (n={sr.size})", show_dropped=True),
            dict(name="hpp",
                 subtitle=f"grey = dropped. HPP keeps {sh.size}/{cfg.num_routed} whole experts.",
                 surv=sh, color=PALETTE["hpp"], marker=METHODS["hpp"]["marker"],
                 kept_label=f"kept (n={sh.size})", show_dropped=True),
        ]

        cfg.pca_dir.mkdir(parents=True, exist_ok=True)
        for v in variants:
            fig, ax = new_fig(FIG)
            if v["show_dropped"]:
                ax.scatter(coords[:, 0], coords[:, 1], s=16, c=PALETTE["dropped"],
                           alpha=0.7, linewidths=0, label="dropped")
            ax.scatter(coords[v["surv"], 0], coords[v["surv"], 1], s=32, c=v["color"],
                       marker=v["marker"], alpha=0.95, linewidths=0.6,
                       edgecolors="white", label=v["kept_label"])
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            finalize(ax, xlabel="PC1", ylabel="PC2", subtitle=v["subtitle"], legend=True,
                     grid_axis="both")
            fig.savefig(cfg.pca_dir / f"pca_subspace_{cfg.model_tag}_L{lid}_"
                                      f"s{int(sparsity * 100)}_{v['name']}.png",
                        bbox_inches="tight")
            plt.show()

    df = pd.DataFrame(rows)
    df.to_csv(cfg.pca_dir / f"pca_variance_{cfg.model_tag}_s{int(sparsity * 100)}.csv",
              index=False)
    print("\n  PC1+PC2 cumulative variance explained (cf. REAP Table A4):")
    print(df.to_string(index=False))
    return df