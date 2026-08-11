"""ONE forward pass over the calibration set collects everything the scores need.

PERFORMANCE, which is a correctness issue at this scale: every `.cpu()`, `float(...)` or
boolean-mask index inside a hook drains the GPU pipeline. With one hook per expert that is
Lm*E syncs per window -- 1,728 for DeepSeek and 6,144 for Qwen3, times 1024 windows. So
all statistics live in GPU-side [Lm,E,...] tensors and are indexed, never transferred, by
the hooks. The only per-window transfers are the two [Lm,E] history slices.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from hpp.config import HPPConfig
from hpp.utils import load_pkl, save_pkl


class Accum:
    """GPU-resident statistics an adapter fills by INDEXING, e.g.

        acc.reap_num[li, e] += (gate * norms).sum()

    Never call .cpu()/.item()/float() on these inside a hook.

    reap_num / cnt are PER-SAMPLE (cleared by reset_sample) so the profiler can keep a
    history and the expert-convergence panel stays derivable. The rest is cumulative.
    """

    def __init__(self, cfg: HPPConfig, device):
        Lm, E, D, d = cfg.num_moe_layers, cfg.num_routed, cfg.moe_inter, cfg.hidden
        self.device = device
        z = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
        self.g_sq = z(Lm, E, D)         # sum of squared down-proj inputs -> channel energy
        self.g_nact = z(Lm, E)
        self.fp_sum = z(Lm, E, d)       # sum of expert outputs -> functional fingerprint
        self.fp_cnt = z(Lm, E)
        self.reap_num = z(Lm, E)        # sum over active tokens of  gate * ||f(x)||
        self.cnt = z(Lm, E)

    def reset_sample(self):
        self.reap_num.zero_()
        self.cnt.zero_()


def profile(cfg: HPPConfig, adapter, model, calib):
    Lm, E = cfg.num_moe_layers, cfg.num_routed
    dev = getattr(model, "device", None) or torch.device("cuda:0")
    acc = Accum(cfg, dev)

    colnorm = adapter.down_colnorms(model)                       # {(li,e): [D]} on CPU
    CN = torch.stack([torch.stack([colnorm[(li, e)] for e in range(E)])
                      for li in range(Lm)])                       # [Lm,E,D], CPU

    hist_reap, hist_cnt, hist_infl = [], [], []
    cscore_snaps: dict[int, dict] = {}
    checkpoints = sorted(cfg.conv_checkpoints)

    # layer influence: identical for every architecture, so it lives here
    xin: dict[int, torch.Tensor] = {}
    infl_acc, infl_cnt = np.zeros(Lm), np.zeros(Lm)

    def pre_ln(li):
        def h(m, inp):
            xin[li] = inp[0].detach()
        return h

    def post_moe(li):
        def h(m, inp, out):
            d = (out[0] if isinstance(out, tuple) else out).detach()
            a = xin.pop(li, None)
            if a is None:
                return
            a = a.reshape(-1, a.shape[-1]).float()
            b = a + d.reshape(-1, d.shape[-1]).float()
            sim = 1.0 - F.cosine_similarity(a, b, dim=-1)
            good = torch.isfinite(sim)
            infl_acc[li] += float(sim[good].sum().item())     # Lm syncs/window, not Lm*E
            infl_cnt[li] += int(good.sum().item())
        return h

    handles = []
    for lid in cfg.moe_layer_ids:
        li = cfg.lid2idx[lid]
        handles.append(adapter.layernorm_of(model, lid).register_forward_pre_hook(pre_ln(li)))
        handles.append(adapter.moe_of(model, lid).register_forward_hook(post_moe(li)))
    restore_experts = adapter.instrument(model, acc)

    def snapshot():
        g = acc.g_sq.detach().cpu()
        n = acc.g_nact.detach().cpu().clamp(min=1).unsqueeze(-1)
        cs = CN * torch.sqrt(g / n)                              # [Lm,E,D]
        return {(li, e): cs[li, e].numpy().astype(np.float16)
                for li in range(Lm) for e in range(E)}

    with torch.no_grad():
        for i, ids in enumerate(calib):
            acc.reset_sample()
            xin.clear()
            infl_acc[:] = 0
            infl_cnt[:] = 0
            model(input_ids=ids.to(model.device))

            hist_reap.append(acc.reap_num.detach().cpu().numpy().copy())   # 1 sync
            hist_cnt.append(acc.cnt.detach().cpu().numpy().copy())         # 1 sync
            hist_infl.append(np.where(infl_cnt > 0, infl_acc / np.maximum(infl_cnt, 1), np.nan))

            if (i + 1) in checkpoints:
                cscore_snaps[i + 1] = snapshot()
            if (i + 1) % 50 == 0:
                print(f"  profiled {i + 1}/{len(calib)}")

    restore_experts()
    for h in handles:
        h.remove()
    if len(calib) not in cscore_snaps:
        cscore_snaps[len(calib)] = snapshot()

    fp = (acc.fp_sum / acc.fp_cnt.clamp(min=1).unsqueeze(-1)).detach().cpu().numpy()
    fingerprint = {(li, e): np.nan_to_num(fp[li, e], nan=0.0, posinf=0.0,
                                          neginf=0.0).astype(np.float32)
                   for li in range(Lm) for e in range(E)}
    g_sq = {(li, e): acc.g_sq[li, e].detach().cpu() for li in range(Lm) for e in range(E)}
    g_nact = {(li, e): float(acc.g_nact[li, e]) for li in range(Lm) for e in range(E)}
    return {"g_sq": g_sq, "g_nact": g_nact, "fingerprint": fingerprint,
            "hist_reap": np.array(hist_reap), "hist_cnt": np.array(hist_cnt),
            "hist_infl": np.array(hist_infl),
            "cscore_snaps": cscore_snaps, "colnorm": colnorm}


def cached_profile(cfg: HPPConfig, adapter, model, calib):
    p = Path(cfg.profile_path)
    if p.exists():
        print(f"  load cached profile {p.name}")
        return load_pkl(p)
    print("  profiling (cache miss)...")
    prof = profile(cfg, adapter, model, calib)
    save_pkl(prof, p)
    return prof