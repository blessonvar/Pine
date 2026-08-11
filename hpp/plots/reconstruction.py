"""LoRA repair curves: HPP vs REAP, one standalone figure per quantity."""
from __future__ import annotations


from hpp.plots.style import FIG, PALETTE, finalize, save, use_style


def _plot_curve(cfg, dfs, ycol, subtitle, sparsity, name):
    import matplotlib.pyplot as plt

    use_style()
    fig, ax = plt.subplots(figsize=FIG)
    for method, df in dfs.items():
        d = df[["step", ycol]].dropna()
        if d.empty:
            continue
        color = PALETTE.get(method, "#444")
        if ycol == "train_loss":
            ax.plot(d["step"], d[ycol], color=color, lw=1.3, alpha=0.85,
                    label=method.upper())
        else:
            ax.plot(d["step"], d[ycol], "-o", color=color, ms=4, label=method.upper())
    finalize(ax, xlabel="LoRA step", ylabel="loss", subtitle=f"{subtitle}, s={sparsity:.0%}")
    save(fig, cfg, name)
    plt.show()
    return fig


def plot_reconstruction(cfg, dfs, sparsity):
    """Two separate figures: the training curve and the held-out curve answer different
    questions and should not share an axis."""
    _plot_curve(cfg, dfs, "train_loss",
                "C4, identical seed / data order / schedule for both methods", sparsity,
                f"reconstruction_train_{cfg.model_tag}_s{int(sparsity * 100)}")
    _plot_curve(cfg, dfs, "eval_loss",
                "WikiText-2, same corpus as the reported perplexity", sparsity,
                f"reconstruction_eval_{cfg.model_tag}_s{int(sparsity * 100)}")