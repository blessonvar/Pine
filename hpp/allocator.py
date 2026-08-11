"""The allocator: total budget -> layers (ALPHA) -> drop (floors) -> widths (BETA) -> channels.

Special cases the hierarchy recovers:
  high gamma           -> floors approach D  -> few experts fit -> REAP (drop whole experts)
  low gamma + beta=0   -> uniform widths, nothing dropped        -> SlimMoE (uniform slim)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from hpp.config import HPPConfig
from hpp.floors import gamma_to_floors


def _waterfill(imp, total, lo, hi, power, round_to):
    """Pay each bin its floor `lo`, then split the surplus proportional to imp**power;
    clip to `hi` and redistribute the clipped residual. lo/hi may be scalar or per-bin."""
    imp = np.maximum(np.asarray(imp, float), 1e-12) ** power
    lo = np.broadcast_to(np.asarray(lo, float), imp.shape).astype(float).copy()
    hi = np.broadcast_to(np.asarray(hi, float), imp.shape).astype(float).copy()
    total = float(np.clip(total, lo.sum(), hi.sum()))
    K = np.clip(lo + (imp / imp.sum()) * (total - lo.sum()), lo, hi)
    for _ in range(300):
        resid = total - K.sum()
        if abs(resid) < max(1.0, round_to / 2.0):
            break
        slack = (K < hi) if resid > 0 else (K > lo)
        if not slack.any():
            break
        sh = imp * slack
        s = sh.sum()
        sh = sh / s if s > 0 else slack / slack.sum()
        K = np.clip(K + resid * sh, lo, hi)
    if round_to > 1:
        K = np.clip(np.round(K / round_to) * round_to, lo, hi)
    return K.astype(int)


def _layer_lower_clip(cfg, scores, floors, li):
    """A layer must fund at least its TOP_K highest-saliency experts at their floors,
    or routing has nowhere to send a token."""
    order = np.argsort(scores["expert_imp"][li])[::-1]
    return int(sum(floors[(li, int(e))] for e in order[:cfg.top_k]))


def stage1_layer_budgets(cfg: HPPConfig, scores, floors, sparsity, alpha):
    B_total = round((1 - sparsity) * cfg.total_channels)
    lo = np.array([_layer_lower_clip(cfg, scores, floors, li)
                   for li in range(cfg.num_moe_layers)], float)
    return _waterfill(scores["layer_imp"], B_total, lo,
                      cfg.num_routed * cfg.moe_inter, alpha, cfg.round_to)


def stage2_survivors(cfg: HPPConfig, scores, floors, li, B_l):
    """Keep the MOST top-saliency experts whose floors fit B_l (>= top_k)."""
    order = np.argsort(scores["expert_imp"][li])[::-1]
    cum, m = 0, 0
    for e in order:
        if cum + floors[(li, int(e))] <= B_l:
            cum += floors[(li, int(e))]
            m += 1
        else:
            break
    m = max(m, cfg.top_k)                       # stage-1 clip guarantees top-k fit
    return [int(e) for e in order[:m]], [int(e) for e in order[m:]]


def stage3_widths(cfg: HPPConfig, scores, floors, li, survivors, B_l, beta):
    """Water-fill B_l over survivors: floor first, surplus ~ saliency**beta, cap at D."""
    sal = np.array([scores["expert_imp"][li, e] for e in survivors])
    lo = np.array([floors[(li, e)] for e in survivors], float)
    K = _waterfill(sal, B_l, lo, cfg.moe_inter, beta, cfg.round_to)
    return {e: int(k) for e, k in zip(survivors, K)}


def select_hpp(cfg: HPPConfig, scores, sparsity, alpha=None, beta=None, gamma=None, floors=None):
    """Full stages 1-3 + channel selection.
    -> keep, prune, actual_sparsity, per-layer info DataFrame."""
    alpha = cfg.alpha if alpha is None else alpha
    beta = cfg.beta if beta is None else beta
    gamma = cfg.gamma if gamma is None else gamma
    if floors is None:
        assert gamma is not None, "select_hpp needs a calibrated gamma (or explicit floors)"
        floors = gamma_to_floors(cfg, scores, gamma)

    B = stage1_layer_budgets(cfg, scores, floors, sparsity, alpha)
    keep, prune, info = {}, {}, []
    for li in range(cfg.num_moe_layers):
        surv, drop = stage2_survivors(cfg, scores, floors, li, int(B[li]))
        Ke = stage3_widths(cfg, scores, floors, li, surv, int(B[li]), beta)
        for e in range(cfg.num_routed):
            k = Ke.get(e, 0)
            order = torch.argsort(scores["cscore"][(li, e)], descending=True)
            kept, _ = torch.sort(order[:k])
            keep[(li, e)] = kept
            pr = order[k:]
            if pr.numel() > 0:
                prune[(li, e)], _ = torch.sort(pr)
        info.append({"layer_idx": li, "layer_id": cfg.idx2lid[li], "budget": int(B[li]),
                     "survivors": len(surv), "drops": len(drop)})
    kept_total = sum(len(v) for v in keep.values())
    return keep, prune, 1 - kept_total / cfg.total_channels, pd.DataFrame(info)