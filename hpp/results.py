"""CSV logging. One row per (method, sparsity, alpha, beta) per results file.

The `profile` column is not decoration: the repaired-checkpoint pass must re-run the SAME
battery that produced its "before" row, otherwise the before/after table compares
different benchmarks. Recording it removes the guesswork that the old notebooks papered
over with an EVAL_PROFILE_OVERRIDE constant.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from hpp.config import HPPConfig

BASE_COLS = ["method", "sparsity", "actual_sparsity", "alpha", "beta", "gamma", "profile"]
EFF_COLS = ["peak_vram_gb", "avg_vram_gb", "ttft_ms", "decode_tps"]


def csv_path(cfg: HPPConfig, profile: str, reconstructed: bool = False) -> Path:
    if reconstructed:
        return Path(cfg.sweep_recon_csv)
    return Path(cfg.sweep_csv if cfg.profile(profile).is_sweep_like else cfg.single_csv)


def csv_cols(cfg: HPPConfig, profile: str) -> list[str]:
    return BASE_COLS + cfg.metric_cols(profile) + EFF_COLS


def log_result(cfg, profile, method, sparsity, actual, metrics,
               alpha=None, beta=None, gamma=None, reconstructed=False):
    path = csv_path(cfg, profile, reconstructed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = csv_cols(cfg, profile)
    new = not path.exists()
    row = {"method": method, "sparsity": sparsity, "actual_sparsity": round(actual, 4),
           "alpha": alpha, "beta": beta, "gamma": gamma, "profile": profile, **metrics}
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k) for k in cols})


def is_done(cfg, profile, method, sparsity, alpha=None, beta=None, reconstructed=False):
    path = csv_path(cfg, profile, reconstructed)
    if not path.exists():
        return False
    d = pd.read_csv(path)
    q = (d["method"] == method) & np.isclose(d["sparsity"].astype(float), float(sparsity))
    if alpha is not None:
        q &= np.isclose(pd.to_numeric(d["alpha"], errors="coerce"), float(alpha))
    if beta is not None:
        q &= np.isclose(pd.to_numeric(d["beta"], errors="coerce"), float(beta))
    if not q.any():
        return False

    rows = d[q].copy()
    metric_cols = cfg.metric_cols(profile)
    for col in metric_cols:
        if col not in rows.columns:
            return False
        rows = rows[pd.notna(pd.to_numeric(rows[col], errors="coerce"))]
        if rows.empty:
            return False
    return True


def load_results(cfg, profile="full_sweep", reconstructed=False):
    path = csv_path(cfg, profile, reconstructed)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for c in ("alpha", "beta", "gamma", "sparsity", "actual_sparsity"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def before_row(cfg, profile, method, sparsity, alpha=None, beta=None):
    """The exact row a repaired checkpoint must be compared against. Returns the config
    that produced it, so the 'after' run prunes identically instead of re-deriving from
    whatever the current globals happen to say."""
    df = load_results(cfg, profile)
    if df.empty:
        raise ValueError(f"no results in {csv_path(cfg, profile)} -- run the sweep first")
    q = (df.method == method) & np.isclose(df.sparsity, float(sparsity))
    if alpha is not None:
        q &= np.isclose(df.alpha, float(alpha))
    if beta is not None:
        q &= np.isclose(df.beta, float(beta))
    rows = df[q]
    if rows.empty:
        raise ValueError(f"no 'before' row for {method} s={sparsity:.0%} in {csv_path(cfg, profile)}")
    if len(rows) > 1:
        print(f"  [WARNING] {len(rows)} rows match {method} s={sparsity:.0%} "
              f"(alpha/beta: {list(zip(rows['alpha'], rows['beta']))}) -- using the first")
    return rows.iloc[0]


def before_after_table(cfg, profile="full_sweep", metric=None, sparsities=None):
    metric = metric or "perplexity"
    sparsities = sparsities or cfg.recon.sparsities
    before, after = load_results(cfg, profile), load_results(cfg, profile, reconstructed=True)
    rows = []
    for s in sparsities:
        for method in ("hpp", "reap"):
            b = before[(before.method == method) & np.isclose(before.sparsity, s)]
            if b.empty:
                continue
            b = b.iloc[0]
            row = {"Method": method.upper(), "Sparsity": f"{b['actual_sparsity']:.1%}",
                   f"{metric} (before)": b.get(metric), f"{metric} (after)": None,
                   "recovered %": None}
            if not after.empty:
                a = after[(after.method == method) & np.isclose(after.sparsity, s)]
                if not a.empty:
                    a = a.iloc[0]
                    row[f"{metric} (after)"] = a.get(metric)
                    if pd.notna(b.get(metric)) and pd.notna(a.get(metric)) and b.get(metric):
                        row["recovered %"] = round(
                            100 * (b[metric] - a[metric]) / b[metric], 1)
            rows.append(row)
    df = pd.DataFrame(rows)
    out = cfg.res_dir / "before_after_table.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    return df