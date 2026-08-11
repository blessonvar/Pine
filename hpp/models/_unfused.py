"""Shared adapter for the three architectures whose experts are a plain ModuleList of
MLPs with real Linear layers (Mixtral / DeepSeek / Qwen3).

Only two things vary between them and both are declared as class attributes:
  PROJ        the attribute names of the (gate, up, down) Linears
  router      how the routing decision is read and how a dropped expert is silenced

Everything else hooks, statistics, channel surgery, LoRA enumeration is identical.

ROW ORDER (the subtle one). The faithful REAP score needs each routed token's gate weight
paired with THAT token's output norm. All three implementations hand an expert its tokens
in ASCENDING TOKEN INDEX: Mixtral/Qwen build `top_x` via torch.where on the expert mask
(row-major -> ascending), and DeepSeek's moe_infer argsorts the flattened top-k ids
(stable -> ascending within an expert). Indexing the cached weights with the boolean mask
`w[sel == e]` is row-major over [T, k] and therefore lands in the same ascending order.
Transposing first (as an earlier draft did) orders by k-slot instead and silently pairs
the wrong gate with the wrong token.
"""
from __future__ import annotations

import gc

import torch
import torch.nn as nn

from hpp.adapter import ModelAdapter
from hpp.utils import vram_gb

try:
    import bitsandbytes as bnb
    import bitsandbytes.functional as bnbF
    HAS_BNB = True
except Exception:
    HAS_BNB = False


class DeadExpert(nn.Module):
    """Never called once its router weight is zeroed; present so indices stay valid."""

    def forward(self, x):
        return torch.zeros_like(x)


class UnfusedAdapter(ModelAdapter):
    PROJ = ("gate_proj", "up_proj", "down_proj")
    MOE_ATTR = "mlp"

    # structure
    def moe_of(self, model, lid):
        return getattr(model.model.layers[lid], self.MOE_ATTR)

    def expert(self, model, lid, e):
        return self.moe_of(model, lid).experts[e]

    def router_of(self, model, lid):
        return self.moe_of(model, lid).gate

    def _dequant(self, lin):
        w = lin.weight
        if HAS_BNB and getattr(w, "quant_state", None) is not None:
            return bnbF.dequantize_4bit(w.data, w.quant_state).float()
        return w.data.float()

    def down_colnorms(self, model):
        """down_proj.weight is [hidden, D]; intermediate channel c is COLUMN c."""
        g, u, d = self.PROJ
        out = {}
        for lid in self.cfg.moe_layer_ids:
            li = self.cfg.lid2idx[lid]
            for e in range(self.cfg.num_routed):
                out[(li, e)] = self._dequant(getattr(self.expert(model, lid, e), d)).norm(dim=0).cpu()
        return out

    # routing (per-arch)
    def read_routing(self, model, out):
        """-> (sel [T,k] long, w [T,k] float) on CPU, or None if the layer produced
        non-finite logits (do not poison the statistics with a NaN layer)."""
        raise NotImplementedError

    def mask_router(self, model, lid, drop_ids):
        raise NotImplementedError

    # profiling
    def instrument(self, model, acc):
        """SYNC BUDGET: the routing tensors are moved to CPU ONCE PER LAYER in the router
        hook, so the per-expert gate lookup `w[sel == e]` is a CPU op. Everything touching
        GPU tensors accumulates in place with torch.where instead of boolean-mask indexing
        (which syncs, because the output shape depends on the data). Net: ~Lm syncs per
        window instead of ~4*Lm*E 27 instead of 6,912 on DeepSeek."""
        cfg = self.cfg
        _, _, d_name = self.PROJ
        route: dict[int, object] = {}
        handles = []

        def router_hook(lid):
            def h(m, inp, out):
                route[lid] = self.read_routing(model, out)
            return h

        def down_hook(li, lid, e):
            def h(m, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).detach().float()
                if x.numel() == 0:
                    return
                f = out.reshape(-1, out.shape[-1]).detach().float()
                fin = torch.isfinite(x).all(1) & torch.isfinite(f).all(1)
                m2 = fin.unsqueeze(1)
                zx = torch.zeros((), device=x.device, dtype=x.dtype)
                xf = torch.where(m2, x, zx)            # non-finite rows -> 0, no mask index
                ff = torch.where(m2, f, zx)
                n_t = fin.sum().to(acc.device)         # GPU scalar; never .item()

                acc.g_sq[li, e] += (xf * xf).sum(0).to(acc.device)
                acc.g_nact[li, e] += n_t
                acc.fp_sum[li, e] += ff.sum(0).to(acc.device)
                acc.fp_cnt[li, e] += n_t

                cached = route.get(lid)
                if cached is None:
                    return                             # NaN layer: do not poison the score
                sel, w = cached                        # CPU tensors -> next line is CPU work
                gate_e = w[sel == e]                   # ascending token order, see docstring
                if gate_e.numel() != x.shape[0]:
                    return                             # rows/routing disagree: skip, do not guess
                ge = gate_e.to(f.device, non_blocking=True)
                ge = torch.where(fin, ge, torch.zeros((), device=f.device))
                norms = ff.norm(dim=-1)                # already 0 on non-finite rows
                acc.reap_num[li, e] += (ge * norms).sum().to(acc.device)
                acc.cnt[li, e] += n_t
            return h

        for lid in cfg.moe_layer_ids:
            li = cfg.lid2idx[lid]
            handles.append(self.router_of(model, lid).register_forward_hook(router_hook(lid)))
            for e in range(cfg.num_routed):
                handles.append(getattr(self.expert(model, lid, e), d_name)
                               .register_forward_hook(down_hook(li, lid, e)))

        def restore():
            for h in handles:
                h.remove()

        return restore

    # surgery
    def _mk(self, W, dev, force_bf16=False):
        if self.cfg.quantize and not force_bf16:
            lin = bnb.nn.Linear4bit(W.shape[1], W.shape[0], bias=False,
                                    compute_dtype=torch.bfloat16, quant_type="nf4")
            lin.weight = bnb.nn.Params4bit(W.contiguous(), requires_grad=False,
                                           quant_type="nf4", compress_statistics=True)
        else:
            lin = nn.Linear(W.shape[1], W.shape[0], bias=False, dtype=torch.bfloat16)
            lin.weight = nn.Parameter(W.to(torch.bfloat16).contiguous(), requires_grad=False)
        return lin.to(dev)

    def structural_removal(self, model, keep):
        cfg = self.cfg
        g_name, u_name, d_name = self.PROJ
        dead_by_layer: dict[int, list[int]] = {}
        for (li, e), kept in keep.items():
            if kept.numel() == 0:
                dead_by_layer.setdefault(li, []).append(e)

        n_dead = 0
        for li, dead in dead_by_layer.items():
            lid = cfg.idx2lid[li]
            self.mask_router(model, lid, dead)
            for e in dead:
                self.moe_of(model, lid).experts[e] = DeadExpert()
                n_dead += 1

        n_slim = 0
        for (li, e), kept in keep.items():
            k = kept.numel()
            if k == 0 or k >= cfg.moe_inter:
                continue                                  # dropped above / full width untouched
            lid = cfg.idx2lid[li]
            exp = self.expert(model, lid, e)
            dev = getattr(exp, d_name).weight.device
            W1 = self._dequant(getattr(exp, g_name))[kept, :].to(torch.bfloat16)
            W3 = self._dequant(getattr(exp, u_name))[kept, :].to(torch.bfloat16)
            W2 = self._dequant(getattr(exp, d_name))[:, kept].to(torch.bfloat16)
            setattr(exp, g_name, self._mk(W1, dev))
            setattr(exp, u_name, self._mk(W3, dev))
            setattr(exp, d_name, self._mk(W2, dev, force_bf16=cfg.w2_bf16))
            del W1, W2, W3
            n_slim += 1
            if n_slim % 8 == 0:
                # Uniform allocations (e.g. alpha=beta=0) can slim EVERY expert instead of
                # dropping most of them whole, so the fp32 dequant temporaries above must be
                # reclaimed periodically or reserved-but-unallocated fragmentation OOMs long
                # before the final resident footprint would.
                gc.collect()
                torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  surgery [{cfg.prec_tag}] | {n_slim} slimmed, {n_dead} dropped "
              f"| VRAM ~ {vram_gb():.1f} GB")

    def alive_expert_linears(self, model):
        for lid in self.cfg.moe_layer_ids:
            for e in range(self.cfg.num_routed):
                exp = self.expert(model, lid, e)
                if isinstance(exp, DeadExpert):
                    continue
                for proj in self.PROJ:
                    yield f"L{lid}_E{e}_{proj}", exp, proj, getattr(exp, proj)

    # text
    def encode_calib(self, tok, text):
        if getattr(tok, "chat_template", None) is not None:
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           return_tensors="pt")[0]
        return tok(text, return_tensors="pt").input_ids[0]

    def prompt_ids(self, tok, user_text):
        msgs = [{"role": "user", "content": user_text}]
        if getattr(tok, "chat_template", None) is not None:
            return tok.apply_chat_template(msgs, add_generation_prompt=True)
        return tok(user_text).input_ids


class LogitRouterMixin:
    """Router is an nn.Linear returning LOGITS; softmax/top-k happen inside the MoE block.
    Dropping = masking the logit to finfo.min so the expert can never enter the top-k."""

    def read_routing(self, model, out):
        logits = (out if not isinstance(out, tuple) else out[0]).float()
        if not torch.isfinite(logits).all():
            return None
        flat = logits.reshape(-1, logits.shape[-1])
        rw = torch.softmax(flat, dim=-1)
        w, sel = torch.topk(rw, self.cfg.top_k, dim=-1)
        if bool(getattr(model.config, "norm_topk_prob", True)):
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
        return sel.detach().cpu(), w.detach().cpu()

    def mask_router(self, model, lid, drop_ids):
        drop = torch.tensor(sorted({int(i) for i in drop_ids}), dtype=torch.long)
        if drop.numel() == 0:
            return

        def hook(m, args, out):
            logits = out.clone()
            logits[..., drop.to(logits.device)] = torch.finfo(logits.dtype).min
            return logits

        self.router_of(model, lid).register_forward_hook(hook)