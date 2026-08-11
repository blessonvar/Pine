"""REAP / SlimMoE / cliff-probe expressed as keep-dicts.

Every method HPP included produces the SAME object: {(li, e): kept channel indices}.
Surgery is then one code path, so a baseline cannot accidentally be served by a different
mechanism than HPP (which is how "REAP is faster" artefacts appear in the literature).
"""
from __future__ import annotations

import numpy as np
import torch

from hpp.config import HPPConfig


def _full(cfg):
    return torch.arange(cfg.moe_inter)


def _empty():
    return torch.empty(0, dtype=torch.long)


def reap_keep(cfg: HPPConfig, scores, sparsity):
    """Drop the n lowest-saliency experts per layer, survivors untouched at full width.

    Scores with expert_imp, which IS REAP's own saliency (see scores.py) the same
    ranking HPP uses. So the only difference between the two methods is what they do with
    the ranking, not how they built it."""
    n_prune = min(round(sparsity * cfg.num_routed), cfg.num_routed - cfg.top_k)
    keep, dropped = {}, []
    for li in range(cfg.num_moe_layers):
        drop = set(np.argsort(scores["expert_imp"][li])[:n_prune].tolist())
        for e in range(cfg.num_routed):
            keep[(li, e)] = _empty() if e in drop else _full(cfg)
        dropped += [(li, e) for e in sorted(drop)]
    return keep, dropped, n_prune / cfg.num_routed


def slimmoe_keep(cfg: HPPConfig, scores, sparsity):
    """Uniform channel slim across ALL experts, nothing dropped."""
    K = int(np.clip(round((1 - sparsity) * cfg.moe_inter / cfg.round_to) * cfg.round_to,
                    cfg.min_keep, cfg.moe_inter))
    keep = {}
    for li in range(cfg.num_moe_layers):
        for e in range(cfg.num_routed):
            order = torch.argsort(scores["cscore"][(li, e)], descending=True)
            kept, _ = torch.sort(order[:K])
            keep[(li, e)] = kept
    return keep, [], 1 - K / cfg.moe_inter


def drop_keep(cfg: HPPConfig, scores, n_drop):
    """Cliff probe: physically drop n_drop experts/layer, survivors at full width.
    Goes through the same surgery as everything else, so the probe sees the true pruned
    forward (router renormalised) rather than a masked approximation."""
    keep = {}
    for li in range(cfg.num_moe_layers):
        drop = set(np.argsort(scores["expert_imp"][li])[:n_drop].tolist())
        for e in range(cfg.num_routed):
            keep[(li, e)] = _empty() if e in drop else _full(cfg)
    return keep