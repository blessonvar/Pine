"""gamma -> per-expert channel FLOOR.

F[li,e] = smallest number of channels retaining `gamma` of expert (li,e)'s OWN channel
energy, clipped to [min_keep, D] and rounded up to a multiple of round_to (bnb's 4-bit
kernels want widths that are multiples of 64).

The floor is what makes HPP's expert dropping DYNAMIC: an expert whose energy is spread
across all its channels demands a large floor and therefore survives; an expert whose
energy collapses into a handful of channels demands almost nothing and becomes a drop
candidate when the layer budget is tight.
"""
from __future__ import annotations

import numpy as np

from hpp.config import HPPConfig


def gamma_to_floors(cfg: HPPConfig, scores, gamma):
    D = cfg.moe_inter
    floors = {}
    for (li, e), c in scores["cscore"].items():
        energy = np.sort(np.asarray(c, dtype=np.float64) ** cfg.energy_power)[::-1]
        cum = np.cumsum(energy)
        tot = cum[-1] if cum[-1] > 0 else 1.0
        k = int(np.searchsorted(cum / tot, gamma)) + 1
        k = int(np.ceil(np.clip(k, cfg.min_keep, D) / cfg.round_to) * cfg.round_to)
        floors[(li, e)] = int(min(k, D))
    return floors