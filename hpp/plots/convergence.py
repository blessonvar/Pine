"""Saliency convergence: is the calibration set big enough?

All three panels plot TOTAL VARIATION distance between successive saliency estimates:
TV = 0.5 * L1, x100 -> "% of saliency mass that moved since the previous checkpoint".
TV is bounded in [0,1] and means the same thing regardless of how many bins the
distribution has (24 layers vs 32 experts vs 2880 channels), so ONE threshold line is
fair across all three raw L1 is mechanically larger for more bins and would make the
channel panel look permanently unconverged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hpp.plots.style import FIG, MUTED, PALETTE, finalize, save, use_style
from hpp.scores import expert_saliency_at_N, layer_importance_at_N
from hpp.utils import save_pkl

# subtitle: "how to read it"; logy: channel panel needs a log y axis since raw L1 is
# mechanically larger for more bins (see module docstring), so its curve spans orders
# of magnitude
META = {
    "layer": ("drives the ALPHA budget split", False),
    "expert": ("drives survivor choice and BETA widths", False),
    "channel": ("drives the floors and channel ranking", True),
}
PANELS = ("layer", "expert", "channel")


def _tv(y):
    return np.asarray(y, dtype=float) * 100.0 / 2.0


def _nice_log_ticks(lo, hi):
    cand = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 200])
    return cand[(cand >= lo) & (cand <= hi)]


def plot_one(cfg, name, d):
    import matplotlib.pyplot as plt

    use_style()
    subtitle, logy = META[name]
    Ns = np.asarray(d["Ns"], float)
    y = _tv(d["l1"])
    m = np.isfinite(y)
    x_f, y_f = Ns[m], y[m]
    thr = cfg.conv_threshold * 100.0

    fig, ax = plt.subplots(figsize=FIG)
    if logy and y_f.size and (y_f > 0).all():
        ax.set_yscale("log")
        lo, hi = min(y_f.min(), thr) * 0.6, y_f.max() * 1.6
        ax.set_ylim(lo, hi)
        ticks = _nice_log_ticks(lo, hi)
        if ticks.size:
            ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(plt.FuncFormatter(lambda v, _: ""))
    else:
        ax.set_ylim(0, max(y_f.max() if y_f.size else 1.0, thr) * 1.25)

    y0 = ax.get_ylim()[0]
    ax.fill_between(x_f, y_f, y0, color=PALETTE["hpp"], alpha=0.10, zorder=1)
    ax.plot(x_f, y_f, "-o", color=PALETTE["hpp"], zorder=3)
    ax.axhline(thr, ls="--", lw=1.3, color=PALETTE["reap"], zorder=2,
               label=f"converged below {thr:g}%")

    below = np.where(y_f < thr)[0]
    if below.size:
        xc = x_f[below[0]]
        ax.axvline(xc, ls=":", lw=1.1, color=MUTED, zorder=2)
        ax.annotate(f"converged @ {int(xc)}", xy=(xc, ax.get_ylim()[1]),
                    xytext=(5, -10), textcoords="offset points",
                    fontsize=8.5, color=MUTED, va="top")
    if y_f.size:
        ax.annotate(f"{y_f[-1]:.2f}", xy=(x_f[-1], y_f[-1]), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="semibold", color=PALETTE["hpp"])

    finalize(ax, xlabel="calibration windows", ylabel="saliency mass shifted (%)",
             subtitle=subtitle)
    save(fig, cfg, f"convergence_{name}_{cfg.model_tag}")
    plt.show()
    return fig


def convergence_study(cfg, prof, plot=True):
    Ns = [n for n in cfg.conv_checkpoints if n <= prof["hist_cnt"].shape[0]]
    snaps = prof["cscore_snaps"]
    snap_Ns = [n for n in cfg.conv_checkpoints if n in snaps]
    out = {}

    def share_l1(vals, axis):
        l1 = [np.nan]
        for i in range(1, len(vals)):
            a, b = vals[i - 1], vals[i]
            if axis is not None:
                a = a / (a.sum(axis, keepdims=True) + 1e-12)
                b = b / (b.sum(axis, keepdims=True) + 1e-12)
                l1.append(float(np.abs(a - b).sum(axis).mean()))
            else:
                a = a / (a.sum() + 1e-12)
                b = b / (b.sum() + 1e-12)
                l1.append(float(np.abs(a - b).sum()))
        return l1

    out["expert"] = {"Ns": Ns, "l1": share_l1([expert_saliency_at_N(prof, N) for N in Ns], axis=1)}
    out["layer"] = {"Ns": Ns, "l1": share_l1([layer_importance_at_N(prof, N) for N in Ns], axis=None)}

    ch = [np.nan]
    for i in range(1, len(snap_Ns)):
        prev, cur = snaps[snap_Ns[i - 1]], snaps[snap_Ns[i]]
        diffs = []
        for k in cur:
            a = prev[k].astype(np.float64)
            b = cur[k].astype(np.float64)
            a = a / (a.sum() + 1e-12)
            b = b / (b.sum() + 1e-12)
            diffs.append(np.abs(a - b).sum())
        ch.append(float(np.mean(diffs)))
    out["channel"] = {"Ns": snap_Ns, "l1": ch}

    cfg.conv_dir.mkdir(parents=True, exist_ok=True)
    save_pkl(out, cfg.conv_dir / "convergence.pkl")
    for name, d in out.items():
        vals = [v if not np.isfinite(v) else v * 100.0 / 2.0 for v in d["l1"]]
        pd.DataFrame({"calib_windows": d["Ns"], "tv_pct_mass_shifted": vals}).to_csv(
            cfg.conv_dir / f"convergence_{name}.csv", index=False)
    if plot:
        for name in PANELS:
            if name in out:
                plot_one(cfg, name, out[name])
    return out