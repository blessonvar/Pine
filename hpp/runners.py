"""Runners. run_sweep does nothing except call run_hpp / run_reap.

That is deliberate: in the old notebooks run_sweep inlined its own pruning and evaluation,
so a sweep produced the CSV row but skipped the allocation csv, the pruned-index npz and
the kept heatmap that the single-run path emitted. Every method therefore now goes through
exactly one function, and a run yields the same artefacts regardless of how it was invoked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from hpp.allocator import select_hpp
from hpp.baselines import reap_keep, slimmoe_keep
from hpp.config import HPPConfig
from hpp.evaluation import evaluate
from hpp.plots.saliency import plot_kept_heatmap
from hpp.results import csv_path, is_done, log_result
from hpp.utils import free, run_tag


def _log_selection(cfg, keep, prune, info, tag):
    """Artefacts every pruned run must leave behind, whatever called it."""
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    if info is not None:
        info.to_csv(cfg.log_dir / f"allocation_{tag}.csv", index=False)
    if prune:
        np.savez_compressed(
            cfg.log_dir / f"pruned_idx_{tag}.npz",
            **{f"l{li}_e{e}": idx.numpy().astype(np.int16) for (li, e), idx in prune.items()})
    plot_kept_heatmap(cfg, keep, tag)


def replot_kept_heatmaps(cfg: HPPConfig, scores, profile="full_sweep", gamma=None):
    """Regenerate every kept-heatmap PNG for a sweep already logged in `profile`'s CSV,
    with no GPU and no re-evaluation. keep is deterministic given scores + the row's own
    sparsity/alpha/beta/gamma, so select_hpp/reap_keep/slimmoe_keep reproduce exactly the
    same keep dict run_sweep computed originally -- this just replots it (e.g. after a
    style change), it does not verify the original run was correct."""
    gamma = cfg.gamma if gamma is None else gamma
    path = csv_path(cfg, profile)
    if not path.exists():
        print(f"  no results at {path}")
        return
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        method = r["method"]
        if method == "baseline":
            continue
        sparsity = float(r["sparsity"])
        if method == "hpp":
            alpha, beta = float(r["alpha"]), float(r["beta"])
            keep, _, _, _ = select_hpp(cfg, scores, sparsity, alpha, beta, gamma)
            tag = run_tag("hpp", sparsity, alpha, beta, gamma)
        elif method == "reap":
            keep, _, _ = reap_keep(cfg, scores, sparsity)
            tag = run_tag("reap", sparsity)
        elif method == "slimmoe":
            keep, _, _ = slimmoe_keep(cfg, scores, sparsity)
            tag = run_tag("slimmoe", sparsity)
        else:
            continue
        plot_kept_heatmap(cfg, keep, tag)


def run_baseline(cfg: HPPConfig, adapter, profile="full_sweep"):
    if is_done(cfg, profile, "baseline", 0.0):
        print("  [skip] baseline")
        return
    print(f"\n=== BASELINE [{cfg.prec_tag}] ===")
    tok, model = adapter.load()
    metrics = evaluate(cfg, adapter, model, tok, profile, "baseline")
    log_result(cfg, profile, "baseline", 0.0, 0.0, metrics)
    del model, tok
    free()


def run_hpp(cfg: HPPConfig, adapter, scores, sparsity, alpha=None, beta=None,
            gamma=None, profile="full_sweep", force=False):
    alpha = cfg.alpha if alpha is None else alpha
    beta = cfg.beta if beta is None else beta
    gamma = cfg.gamma if gamma is None else gamma
    assert gamma is not None, "calibrate gamma first (hpp.gamma.calibrate_gamma)"
    if not force and is_done(cfg, profile, "hpp", sparsity, alpha, beta):
        print(f"  [skip] HPP s={sparsity:.0%} a={alpha} b={beta}")
        return
    tag = run_tag("hpp", sparsity, alpha, beta, gamma)
    print(f"\n=== HPP [{cfg.prec_tag}] | s={sparsity:.0%} a={alpha} b={beta} g={gamma:.3f} ===")
    keep, prune, actual, info = select_hpp(cfg, scores, sparsity, alpha, beta, gamma)
    _log_selection(cfg, keep, prune, info, tag)
    tok, model = adapter.load()
    adapter.structural_removal(model, keep)
    metrics = evaluate(cfg, adapter, model, tok, profile, tag)
    log_result(cfg, profile, "hpp", sparsity, actual, metrics, alpha, beta, gamma)
    del model, tok, keep, prune
    free()


def run_reap(cfg: HPPConfig, adapter, scores, sparsity, profile="full_sweep", force=False):
    if not force and is_done(cfg, profile, "reap", sparsity):
        print(f"  [skip] REAP s={sparsity:.0%}")
        return
    tag = run_tag("reap", sparsity)
    keep, _, actual = reap_keep(cfg, scores, sparsity)
    if not np.isclose(actual, sparsity):
        print(f"  ! REAP quantises to whole experts -> actual sparsity {actual:.2%} "
              f"(requested {sparsity:.2%})")
    print(f"\n=== REAP [{cfg.prec_tag}] | s={sparsity:.0%} ===")
    _log_selection(cfg, keep, None, None, tag)
    tok, model = adapter.load()
    adapter.structural_removal(model, keep)
    metrics = evaluate(cfg, adapter, model, tok, profile, tag)
    log_result(cfg, profile, "reap", sparsity, actual, metrics)
    del model, tok, keep
    free()


def run_slimmoe(cfg: HPPConfig, adapter, scores, sparsity, profile="full_sweep", force=False):
    if not force and is_done(cfg, profile, "slimmoe", sparsity):
        print(f"  [skip] SlimMoE s={sparsity:.0%}")
        return
    tag = run_tag("slimmoe", sparsity)
    keep, _, actual = slimmoe_keep(cfg, scores, sparsity)
    print(f"\n=== SlimMoE [{cfg.prec_tag}] | s={sparsity:.0%} ===")
    _log_selection(cfg, keep, None, None, tag)
    tok, model = adapter.load()
    adapter.structural_removal(model, keep)
    metrics = evaluate(cfg, adapter, model, tok, profile, tag)
    log_result(cfg, profile, "slimmoe", sparsity, actual, metrics)
    del model, tok, keep
    free()


def run_sweep(cfg: HPPConfig, adapter, scores, alpha=None, beta=None, gamma=None,
              profile="full_sweep", methods=("hpp", "reap"), plot=True):
    alpha = cfg.alpha if alpha is None else alpha
    beta = cfg.beta if beta is None else beta
    gamma = cfg.gamma if gamma is None else gamma
    print(f"\n{'='*64}\nSWEEP [{cfg.prec_tag}] {cfg.model_tag}: alpha={alpha} beta={beta} "
          f"gamma={gamma:.3f} profile={profile}\n{'='*64}")
    for sparsity in cfg.sparsities:
        if "hpp" in methods:
            run_hpp(cfg, adapter, scores, sparsity, alpha, beta, gamma, profile)
        if "reap" in methods:
            run_reap(cfg, adapter, scores, sparsity, profile)
        if "slimmoe" in methods:
            run_slimmoe(cfg, adapter, scores, sparsity, profile)
    if plot:
        from hpp.plots.comparison import plot_all, plot_task_bars
        plot_all(cfg, profile)
        plot_task_bars(cfg, profile)


def setup(cfg: HPPConfig, adapter, plot=True):
    """Profile -> scores (cached per precision), then every diagnostic. Frees the GPU."""
    from hpp.calibration import build_calibration
    from hpp.plots.convergence import convergence_study
    from hpp.plots.saliency import plot_expert_saliency, plot_layer_saliency
    from hpp.profiler import cached_profile
    from hpp.scores import cached_scores
    from hpp.utils import load_pkl

    cfg.ensure_dirs()
    if cfg.scores_path.exists() and cfg.profile_path.exists():
        print(f"cache hit (scores + profile) [{cfg.prec_tag}]")
        scores, prof = load_pkl(cfg.scores_path), load_pkl(cfg.profile_path)
    else:
        print(f"cache miss [{cfg.prec_tag}] -- loading model")
        tok, model = adapter.load()
        calib = build_calibration(cfg, tok, adapter)
        prof = cached_profile(cfg, adapter, model, calib)
        scores = cached_scores(cfg, prof)
        del model, tok, calib
        free()

    c = prof["hist_cnt"].sum(0)
    leak = int(np.count_nonzero(scores["expert_imp"][c < 128]))
    print(f"  [guard] {(c < 128).sum()} experts below the activation floor, {leak} leak "
          f"(must be 0)")
    if plot:
        plot_layer_saliency(cfg, scores)
        plot_expert_saliency(cfg, scores)
        convergence_study(cfg, prof)
    return scores, prof