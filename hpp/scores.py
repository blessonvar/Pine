"""Two saliency signals plus the channel score. One per level of the hierarchy.

  layer_imp  [Lm]   ALPHA driver: 1 - cos(x, x + MoE(LN(x))), how much the block moves
                       the residual. Drives the depth-aware budget split.
  expert_imp [Lm,E] BETA driver AND the drop ordering:
                           S = E[ g * ||f(x)|| | token routed to this expert ]
                       This is REAP's own saliency, and HPP uses it unchanged. Scoring the
                       baseline with a different metric would make HPP-vs-REAP a comparison
                       of "allocator + score" two variables. Sharing the score makes it
                       one: given the SAME ranking, does spending the budget as widths beat
                       spending it as whole experts? That is the thesis.
  cscore  {(li,e): [D]} channel saliency: ||down[:, c]|| * RMS(a_c). Since the expert
                       output is sum_c a_c * down[:, c], cscore^2 is the expected squared
                       norm of channel c's contribution energy in a literal sense, which
                       is why gamma's floors integrate cscore^ENERGY_POWER with power 2.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch

from hpp.config import HPPConfig
from hpp.utils import load_pkl, save_pkl


def expert_saliency_at_N(prof, N, min_count=128):
    """REAP saliency from the first N calibration windows. Conditional on ACTIVE tokens,
    which decouples an expert's functional impact from how often it happens to be routed
    to a rarely-used expert can still be critical when it does fire."""
    num = prof["hist_reap"][:N].sum(0)
    cnt = prof["hist_cnt"][:N].sum(0)
    sal = num / np.maximum(cnt, 1)
    sal[cnt < min_count] = 0.0      # seen too rarely for the mean to be trustworthy
    return sal


def layer_importance_at_N(prof, N):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slice
        return np.nanmean(prof["hist_infl"][:N], axis=0)


def compute_scores(cfg: HPPConfig, prof):
    N = prof["hist_reap"].shape[0]
    cscore = {}
    for li in range(cfg.num_moe_layers):
        for e in range(cfg.num_routed):
            k = (li, e)
            rms = torch.sqrt(prof["g_sq"][k] / max(prof["g_nact"][k], 1))
            cscore[k] = torch.nan_to_num(prof["colnorm"][k] * rms,
                                         nan=0.0, posinf=0.0, neginf=0.0)
    clean = lambda a: np.nan_to_num(np.asarray(a, float), nan=0.0, posinf=0.0, neginf=0.0)
    return {"expert_imp": clean(expert_saliency_at_N(prof, N)),
            "layer_imp": clean(layer_importance_at_N(prof, N)),
            "cscore": cscore,
            "fingerprint": prof["fingerprint"]}


def cached_scores(cfg: HPPConfig, prof):
    p = Path(cfg.scores_path)
    if p.exists():
        print(f"  load cached scores {p.name}")
        return load_pkl(p)
    sc = compute_scores(cfg, prof)
    save_pkl(sc, p)
    return sc