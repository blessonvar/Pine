"""Headline figures: one standalone chart per metric, x = ACTUAL sparsity.

Requested sparsity is never plotted. REAP quantises to whole experts and HPP's widths
quantise to multiples of round_to, so the number a method actually achieved is the only
honest x. On the 25/50/75 grid REAP happens to land exactly, but HPP does not always.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hpp.plots.style import FIG, MUTED, PALETTE, finalize, method_style, save, use_style
from hpp.results import load_results

QUALITY = {
    "perplexity": ("WikiText-2 perplexity", "lower is better"),
    "math_avg": ("Math — mean(GSM8K, MATH-500)", "higher is better"),
    "mc_avg": ("Multiple choice — 8-suite mean", "higher is better"),
    "code_avg": ("Code — Eval+ = mean(HumanEval+, MBPP+)", "higher is better"),
}
EFFICIENCY = {
    "peak_vram_gb": ("Peak VRAM", "GB, lower is better"),
    "avg_vram_gb": ("Time-averaged VRAM", "GB, lower is better"),
    "ttft_ms": ("Time to first token", "ms, lower is better"),
    "decode_tps": ("Decode throughput", "tokens/s, higher is better"),
}
METRICS = {**QUALITY, **EFFICIENCY}


def plot_metric(cfg, df, col, profile, methods=("hpp", "reap", "slimmoe")):
    import matplotlib.pyplot as plt

    if df.empty or col not in df.columns or not df[col].notna().any():
        return None
    use_style()
    title, subtitle = METRICS.get(col, (col, ""))
    fig, ax = plt.subplots(figsize=FIG)

    bl = df[df.method == "baseline"]
    if not bl.empty and pd.notna(bl.iloc[0].get(col)):
        ax.axhline(bl.iloc[0][col], ls="--", lw=1.6, color=PALETTE["baseline"],
                   zorder=1, label="unpruned baseline")

    for method in methods:
        sub = df[(df.method == method)].dropna(subset=[col])
        if sub.empty:
            continue
        if method == "hpp":
            for (a, b), grp in sub.groupby(["alpha", "beta"], dropna=False):
                grp = grp.sort_values("actual_sparsity")
                st = method_style(cfg, "hpp", a, b)
                ax.plot(grp["actual_sparsity"] * 100, grp[col], marker=st["marker"],
                        ls=st["ls"], color=st["color"], label=st["label"],
                        alpha=st.get("alpha", 1.0), zorder=3)
        else:
            grp = sub.sort_values("actual_sparsity")
            st = method_style(cfg, method)
            ax.plot(grp["actual_sparsity"] * 100, grp[col], marker=st["marker"],
                    ls=st["ls"], color=st["color"], label=st["label"], zorder=3)

    finalize(ax, xlabel="actual expert sparsity", ylabel=title.split(" — ")[0],
             subtitle=subtitle, pct_x=True)
    save(fig, cfg, f"{col}_{cfg.model_tag}_{profile}")
    plt.show()
    return fig


def plot_quality(cfg, profile="full_sweep"):
    df = load_results(cfg, profile)
    return [plot_metric(cfg, df, c, profile) for c in QUALITY]


def plot_efficiency(cfg, profile="full_sweep"):
    df = load_results(cfg, profile)
    return [plot_metric(cfg, df, c, profile) for c in EFFICIENCY]


def plot_all(cfg, profile="full_sweep"):
    plot_quality(cfg, profile)
    plot_efficiency(cfg, profile)


def compare_precisions(cfg, profile="full_sweep", cols=("perplexity", "math_avg", "code_avg",
                                                        "mc_avg", "avg_vram_gb")):
    """Overlay the nf4 and bf16 result trees. Needs both runs to exist."""
    import matplotlib.pyplot as plt
    from dataclasses import replace

    use_style()
    trees = {}
    for q in (False, True):
        c = replace(cfg, quantize=q)
        d = load_results(c, profile)
        if not d.empty:
            trees[c.prec_tag] = d
    if len(trees) < 2:
        print(f"  need both precisions; found {list(trees)}")
        return None

    ls = {"bf16": "-", "nf4": "--"}
    figs = []
    for col in cols:
        if not all(col in d.columns and d[col].notna().any() for d in trees.values()):
            continue
        fig, ax = plt.subplots(figsize=FIG)
        for prec, d in trees.items():
            for method in ("hpp", "reap"):
                sub = d[(d.method == method)].dropna(subset=[col]).sort_values("actual_sparsity")
                if sub.empty:
                    continue
                st = method_style(cfg, method)
                ax.plot(sub["actual_sparsity"] * 100, sub[col], marker=st["marker"],
                        ls=ls[prec], color=st["color"], alpha=1.0 if prec == "bf16" else 0.6,
                        label=f"{st['label']} [{prec}]")
        title, subtitle = METRICS.get(col, (col, ""))
        finalize(ax, xlabel="actual expert sparsity", ylabel=title.split(" — ")[0],
                 subtitle=subtitle, pct_x=True)
        save(fig, cfg, f"precision_{col}_{cfg.model_tag}_{profile}")
        plt.show()
        figs.append(fig)

    rows = []
    for method in ("baseline", "hpp", "reap"):
        a, b = trees.get("nf4"), trees.get("bf16")
        a = a[a.method == method]
        b = b[b.method == method]
        if a.empty or b.empty:
            continue
        keys = ["sparsity"] + (["alpha", "beta"] if method == "hpp" else [])
        m = a.merge(b, on=keys, suffixes=("_nf4", "_bf16"))
        for _, r in m.iterrows():
            row = {"method": method, "sparsity": r["sparsity"]}
            for col in cols:
                x, y = r.get(f"{col}_nf4"), r.get(f"{col}_bf16")
                if pd.notna(x) and pd.notna(y):
                    row[f"d_{col}"] = round(float(y) - float(x), 4)
            rows.append(row)
    delta = pd.DataFrame(rows)
    if not delta.empty:
        print("\n  bf16 - nf4 (positive = bf16 better, except perplexity):")
        print(delta.to_string(index=False))
        delta.to_csv(cfg.res_dir / f"precision_delta_{cfg.model_tag}_{profile}.csv", index=False)
    return delta


def plot_task_bars(cfg, profile="full_sweep", cols=("math_avg", "mc_avg", "code_avg")):
    """HPP vs REAP per category, grouped by sparsity, baseline as a per-task reference."""
    import matplotlib.pyplot as plt

    df = load_results(cfg, profile)
    if df.empty:
        return None
    use_style()
    cols = [c for c in cols if c in df.columns and df[c].notna().any()]
    if not cols:
        return None
    names = [QUALITY[c][0].split(" — ")[0] for c in cols]
    base = df[df.method == "baseline"]
    base = base.iloc[0] if not base.empty else None

    order = [(m, s) for s in cfg.sparsities for m in ("hpp", "reap")]
    shade = dict(zip(cfg.sparsities, np.linspace(0.95, 0.5, len(cfg.sparsities))))
    x = np.arange(len(cols))
    width = 0.8 / max(len(order), 1)

    fig, ax = plt.subplots(figsize=(2.4 * len(cols) + 3.0, 4.2))
    for j, (m, s) in enumerate(order):
        sub = df[(df.method == m) & np.isclose(df.sparsity, s)]
        r = sub.iloc[0] if not sub.empty else None
        vals = [float(r[c]) if (r is not None and pd.notna(r.get(c))) else np.nan for c in cols]
        pos = x - 0.4 + width * (j + 0.5)
        ax.bar(pos, vals, width * 0.9, color=PALETTE[m], alpha=shade[s],
               edgecolor="white", linewidth=0.7, label=f"{m.upper()} {s:.0%}")
        for px, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(px, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                        color=MUTED)
    if base is not None:
        for i, c in enumerate(cols):
            if pd.notna(base.get(c)):
                ax.plot([x[i] - 0.4, x[i] + 0.4], [base[c]] * 2, ls="--", lw=1.5,
                        color=PALETTE["baseline"], label="baseline" if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    finalize(ax, ylabel="accuracy", subtitle="one-shot, no repair")
    ax.legend(ncol=2, loc="best")
    save(fig, cfg, f"task_comparison_{cfg.model_tag}_{profile}")
    plt.show()
    return fig