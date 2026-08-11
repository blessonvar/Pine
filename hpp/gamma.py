"""Calibrating gamma against the model instead of guessing it.

The chain (see hpp/plots/gamma.py for the figure that draws it):
  1. answer key physically drop N whole experts/layer and measure the primary metric.
     m* = the FEWEST survivors whose metric has NOT collapsed below collapse_frac * best   
     the most aggressive drop that is still safe, i.e. the actual edge of the cliff, not
     just the safest point tested.
     This costs GPU time but it is the only step that knows what the model tolerates.
  2. gamma curve offline, no GPU: for each gamma, what average survivor count do the
     resulting floors produce at the calibration sparsity?
  3. gamma* the gamma whose curve reproduces m*.

alpha=0 in the curve on purpose: gamma must be a property of the model's energy
distribution, not entangled with a depth-allocation choice made later.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hpp.allocator import stage1_layer_budgets, stage2_survivors
from hpp.baselines import drop_keep
from hpp.config import HPPConfig
from hpp.floors import gamma_to_floors
from hpp.utils import free

# Metrics where SMALLER is better. The answer key must know the direction, or a
# perplexity-based probe silently selects the WORST configuration.
LOWER_IS_BETTER = {"perplexity"}


def answer_key_from_drops(cfg: HPPConfig, sparsity=None):
    sparsity = cfg.calib_sparsity if sparsity is None else sparsity
    metric = cfg.calib_primary_metric
    lower = metric in LOWER_IS_BETTER
    df = pd.read_csv(cfg.dropkey_csv)
    df = df[np.isclose(df["sparsity"].astype(float), sparsity)].copy()
    if df.empty:
        raise ValueError(f"no drop rows at sparsity {sparsity} in {cfg.dropkey_csv}")
    df["survivors"] = cfg.num_routed - df["n_drop"].astype(int)
    v = pd.to_numeric(df.get(metric), errors="coerce")
    if v is None or not v.notna().any():
        raise ValueError(
            f"every {metric!r} value in {cfg.dropkey_csv} is empty/NaN.\n"
            f"  The drop probes RAN but {metric} never produced a number. evaluate() prints\n"
            f"    '  {metric} failed: <reason>'\n"
            f"  when that happens scroll up through the ensure_drop_keys output for the\n"
            f"  real error; this exception is only the downstream symptom.\n"
            f"  Fix A: cfg.calib_primary_metric = 'perplexity'   (always produced, dense, cheap)\n"
            f"  Fix B: fix the underlying {metric} failure, delete {cfg.dropkey_csv},\n"
            f"         and re-run so the probes re-execute.")
    n_bad = int((~v.notna()).sum())
    if n_bad:
        print(f"  [WARNING] {n_bad} drop-key row(s) have no {metric} value; ignoring them")
    df = df[v.notna()].copy()
    v = v[v.notna()]

    best = v.min() if lower else v.max()
    thr = best / cfg.calib_collapse_frac if lower else best * cfg.calib_collapse_frac
    ok = (v <= thr) if lower else (v >= thr)
    # the FEWEST survivors that still pass the most aggressive drop that is still safe,
    # i.e. the actual edge of the cliff. NOT .max(): that would settle for the safest point
    # tested even when a more aggressive point tested just as well.
    m_star = int(df[ok]["survivors"].min())

    #  does the probe have any signal at all? 
    lo, hi = float(v.min()), float(v.max())
    spread = (hi - lo) / max(abs(best), 1e-12)
    flat = spread < 0.10
    at_floor = (not lower) and hi < 0.05
    if flat or at_floor:
        why = ("the metric never rises above 5% the model cannot do this task at all"
               if at_floor else
               f"the metric varies by only {spread:.1%} across the whole sweep")
        print(f"  [WARNING] the {metric} probe has NO SIGNAL: {why}.\n"
              f"            Values: {list(np.round(v.values, 4))} at survivors "
              f"{list(df['survivors'].values)}.\n"
              f"            m* is then whichever point noise happened to favour, and gamma "
              f"is fitted to noise.\n"
              f"            Fix: set cfg.calib_primary_metric to a metric this model "
              f"actually moves on 'perplexity' always works and is cheap.")
    print(f"  answer key @ s={sparsity:.0%}: best {metric}={best:.4f} "
          f"({'lower' if lower else 'higher'} is better), threshold={thr:.4f} "
          f"-> {m_star} survivors (drop {cfg.num_routed - m_star})")
    return m_star, df.sort_values("survivors")


def gamma_survivor_curve(cfg: HPPConfig, scores, sparsity=None, alpha=0.0, grid=None):
    sparsity = cfg.calib_sparsity if sparsity is None else sparsity
    grid = cfg.gamma_grid if grid is None else grid
    rows = []
    for g in grid:
        floors = gamma_to_floors(cfg, scores, g)
        B = stage1_layer_budgets(cfg, scores, floors, sparsity, alpha)
        m = np.mean([len(stage2_survivors(cfg, scores, floors, li, int(B[li]))[0])
                     for li in range(cfg.num_moe_layers)])
        mean_floor = np.mean([floors[k] for k in floors])
        rows.append({"gamma": float(g), "avg_survivors": float(m),
                     "mean_floor": float(mean_floor),
                     "mean_floor_pct": float(mean_floor / cfg.moe_inter)})
    return pd.DataFrame(rows)


def calibrate_gamma(cfg: HPPConfig, scores, sparsity=None, alpha=0.0, plot=True):
    from hpp.plots.gamma import plot_gamma_calibration

    sparsity = cfg.calib_sparsity if sparsity is None else sparsity
    m_star, drops = answer_key_from_drops(cfg, sparsity)
    curve = gamma_survivor_curve(cfg, scores, sparsity, alpha)
    j = int(np.argmin(np.abs(curve["avg_survivors"].values - m_star)))
    gamma_star = float(curve["gamma"].iloc[j])

    cfg.calib_dir.mkdir(parents=True, exist_ok=True)
    curve.assign(target_survivors=m_star,
                 chosen=np.isclose(curve["gamma"], gamma_star)).to_csv(
        cfg.calib_dir / f"gamma_calibration_{cfg.model_tag}_s{int(round(sparsity*100))}.csv",
        index=False)
    with open(cfg.calib_dir / f"gamma_chosen_{cfg.model_tag}.json", "w") as f:
        json.dump({"model": cfg.model_tag, "precision": cfg.prec_tag, "sparsity": sparsity,
                   "gamma": gamma_star, "target_survivors": m_star,
                   "mean_floor_pct": float(curve["mean_floor_pct"].iloc[j])}, f, indent=2)
    print(f"  -> calibrated gamma = {gamma_star:.3f} "
          f"(mean floor {curve['mean_floor_pct'].iloc[j]:.1%} of D)")
    if plot:
        plot_gamma_calibration(cfg, curve, drops, m_star, gamma_star, sparsity)
    return gamma_star, curve


def _append_dropkey(cfg, sparsity, n_drop, metrics):
    cols = ["model", "precision", "sparsity", "n_drop", "perplexity"]
    if cfg.calib_primary_metric not in cols:      # primary may BE perplexity
        cols.append(cfg.calib_primary_metric)
    path = Path(cfg.dropkey_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({"model": cfg.model_tag, "precision": cfg.prec_tag, "sparsity": sparsity,
                    "n_drop": n_drop, "perplexity": metrics.get("perplexity"),
                    cfg.calib_primary_metric: metrics.get(cfg.calib_primary_metric)})


def run_drop_probe(cfg, adapter, scores, sparsity, n_drop):
    from hpp.evaluation import evaluate

    tok, model = adapter.load()
    adapter.structural_removal(model, drop_keep(cfg, scores, n_drop))
    m = evaluate(cfg, adapter, model, tok, "single",
                 f"drop{n_drop}_s{int(sparsity * 100)}")
    del model, tok
    free()
    return m


def ensure_drop_keys(cfg, adapter, scores, sparsity=None, drops=None):
    """A row only counts as done if it actually HAS a primary-metric value. A row whose
    metric came back None is worse than a missing row: it is skipped as 'done' and then
    silently poisons the answer key."""
    sparsity = cfg.calib_sparsity if sparsity is None else sparsity
    drops = cfg.calib_drop_sweep if drops is None else drops
    metric = cfg.calib_primary_metric
    have = set()
    path = Path(cfg.dropkey_csv)
    if path.exists():
        d = pd.read_csv(path)
        v = pd.to_numeric(d.get(metric), errors="coerce")
        good = d[v.notna()] if v is not None else d.iloc[0:0]
        n_bad = len(d) - len(good)
        if n_bad:
            print(f"  {n_bad} drop-key row(s) have no {metric} value -> discarding and re-probing")
            good.to_csv(path, index=False)          # replace, do not accumulate duplicates
        have = {(round(float(s), 4), int(n)) for s, n in zip(good["sparsity"], good["n_drop"])}
    for nd in drops:
        if (round(sparsity, 4), nd) in have:
            continue
        print(f"\n=== drop-key probe: s={sparsity:.0%} n_drop={nd} ===")
        m = run_drop_probe(cfg, adapter, scores, sparsity, nd)
        if m.get(metric) is None:
            print(f"  [WARNING] this probe produced NO {metric}. The row is written anyway so "
                  f"you can see it, but calibrate_gamma will reject it. Look for "
                  f"'{metric} failed:' above.")
        _append_dropkey(cfg, sparsity, nd, m)