"""gpt-oss-20b: architecture facts + config factory.

Model facts (not run knobs, hence here and not in HPPConfig):
* 24 layers, ALL MoE (no dense layer 0, no shared experts)
* 32 routed experts, top-4; hidden 2880, expert width D 2880 (gate_up output = 2D)
* Weights are FUSED & INTERLEAVED 3-D tensors, not per-expert nn.Linear:
      experts.gate_up_proj       [E, hidden, 2D]   (gate = ::2, up = 1::2)
      experts.gate_up_proj_bias  [E, 2D]
      experts.down_proj          [E, D, hidden]
      experts.down_proj_bias     [E, hidden]
* Router: mlp.router (weight[E, hidden] + bias[E] -> top-4 -> softmax)
* SwiGLU: gate.clamp(max=LIMIT); up.clamp(-LIMIT, LIMIT)
          glu = gate * sigmoid(ALPHA * gate); act = (up + 1) * glu; y = act @ down + b
  `act` is the down-proj input -> it is what the channel statistics are collected on.

The model ships natively in MXFP4 and is always loaded dequantized to bf16 for the
pruning math (runs on Ampere/A6000); cfg.quantize therefore controls SURVIVOR serving
precision (NF4 vs bf16), and PREC_TAG keeps the two result trees apart.

Generation: every generative task (gsm8k, math500, humaneval_plus, mbpp_plus) must be
decoded through the harmony chat template with reasoning forced low, then have the
`<|channel|>final<|message|>` wrapper stripped BEFORE answer/code extraction. Skipping
that strip is the single most likely cause of a near-zero score on an unpruned model.
"""
from __future__ import annotations

from hpp.config import HPPConfig

# architecture constants
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0

# generation protocol (harmony template, deterministic greedy decode)
GEN_SYSTEM = "You are a helpful assistant. Reasoning: low"
GEN_KWARGS = "max_gen_toks=512,do_sample=False,temperature=0.0"
GEN_REASONING = "low"

# Perplexity is scored INSIDE the harmony wrapper, in the assistant's final channel --
# the same framing every other metric already uses. Bare prose puts gpt-oss in a regime
# it was post-trained out of and reports the format's surprise as if it were the text's.
PPL_PREFIX_PROMPT = "Continue the following text."


def config(quantize: bool = False, **overrides) -> HPPConfig:
    cfg = HPPConfig(
        model_name="openai/gpt-oss-20b",
        model_tag="gptoss20b",
        num_layers_total=24,
        num_routed=32,
        moe_inter=2880,
        hidden=2880,
        top_k=4,
        first_dense=0,
        n_shared=0,
        quantize=quantize,
        calib_total_windows=1024,
        calib_max_len=2048,
        calib_drop_sweep=(16, 12, 8, 4),   # 32 experts, top-4 -> can drop up to 28
        calib_sparsity=0.50,
        eval_batch_size=2,                 # safe alongside the ~40 GB bf16 model on a 48 GB A6000
        gpu_ids="0",
        base_cache="cache_gptoss",
        base_res="results_gptoss",
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"HPPConfig has no field {k!r}")
        setattr(cfg, k, v)
    return cfg



# ADAPTER
import gc
import re

import torch
import torch.nn as nn

from hpp.adapter import ModelAdapter
from hpp.utils import vram_gb

# The channel header is not always the bare "final" -- gpt-oss routinely emits
# "<|channel|>final code<|message|>" (a qualifier after the channel name) for code
# answers specifically. A literal .split("<|channel|>final<|message|>") never matches
# that, silently falls back to the WHOLE raw text (chain-of-thought included), and the
# code/answer extractors downstream pick up the reasoning instead of the answer -- no
# error, just a near-zero score that looks like a capability failure.
_FINAL_CHANNEL = re.compile(r"<\|channel\|>final[^<]*<\|message\|>")

try:
    from transformers import Mxfp4Config
    HAS_MXFP4 = True
except Exception:
    HAS_MXFP4 = False
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except Exception:
    HAS_BNB = False


class GateMasker:
    """Mask the ROUTER's dense per-expert scores: zero the dropped experts, then
    renormalise so the surviving top-k weights still sum to 1 (gpt-oss already
    softmax-normalises its top-k, so without the renormalise the block would silently
    shrink its own output)."""

    def __init__(self, router, drop_ids):
        self.drop = torch.tensor(sorted({int(i) for i in drop_ids}), dtype=torch.long)
        self.h = router.register_forward_hook(self._post)

    def _post(self, m, args, out):
        if not isinstance(out, tuple):
            return out
        scores = out[0].clone()
        if self.drop.numel() > 0:
            scores[..., self.drop.to(scores.device)] = 0.0
            scores = scores / (scores.sum(-1, keepdim=True) + 1e-20)
        return (scores,) + tuple(out[1:])

    def remove(self):
        self.h.remove()


class PrunedGptOssExperts(nn.Module):
    """Replaces the fused GptOssExperts with per-surviving-expert slimmed gate/up/down
    Linears (de-interleaved) and replays the exact gpt-oss SwiGLU. Dropped experts are
    simply absent; a GateMasker zeroes their router weight."""

    def __init__(self, experts_dict, num_experts, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
        super().__init__()
        self.num_experts = num_experts
        self.alpha, self.limit = alpha, limit
        self.ids = sorted(experts_dict)
        self.gate, self.up, self.down = nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()
        for e, (g, u, d) in experts_dict.items():
            self.gate[str(e)], self.up[str(e)], self.down[str(e)] = g, u, d

    def forward(self, hidden_states, router_indices=None, routing_weights=None):
        shp = hidden_states.shape
        x = hidden_states.reshape(-1, shp[-1])
        rw = routing_weights.reshape(-1, self.num_experts)
        out = torch.zeros_like(x)
        for e in self.ids:
            w_e = rw[:, e]
            mask = w_e > 0
            if not bool(mask.any()):
                continue
            xe = x[mask]
            g = self.gate[str(e)](xe).clamp(max=self.limit)
            u = self.up[str(e)](xe).clamp(min=-self.limit, max=self.limit)
            ye = self.down[str(e)]((u + 1) * (g * torch.sigmoid(g * self.alpha)))
            out[mask] += w_e[mask, None].to(ye.dtype) * ye
        return out.reshape(shp)


class GptOssAdapter(ModelAdapter):
    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cfg = self.cfg
        tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = dict(device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
                  attn_implementation=cfg.attn_impl)
        try:
            if HAS_MXFP4:
                kw["quantization_config"] = Mxfp4Config(dequantize=True)
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
        except Exception as e:
            print(f"  [load] mxfp4-dequantize failed ({e}); retrying plain bf16")
            kw.pop("quantization_config", None)
            model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
        model.eval()
        self._assert_arch(model)
        print(f"  model loaded [{cfg.prec_tag}] | weights VRAM ~ {vram_gb():.1f} GB")
        return tok, model

    def _assert_arch(self, model):
        cfg = self.cfg
        for k, want in [("num_local_experts", cfg.num_routed),
                        ("num_experts_per_tok", cfg.top_k),
                        ("hidden_size", cfg.hidden),
                        ("num_hidden_layers", cfg.num_layers_total)]:
            got = getattr(model.config, k, None)
            if got is not None and got != want:
                print(f"  [arch WARNING] {k}: config={got} vs code={want}")
        found = [i for i, ly in enumerate(model.model.layers)
                 if hasattr(ly.mlp, "experts") and hasattr(ly.mlp, "router")]
        print(f"  arch OK: {len(found)} MoE layers" if found == cfg.moe_layer_ids
              else f"  [arch WARNING] MoE layers {found}")

    def moe_of(self, model, lid):
        return model.model.layers[lid].mlp

    def _weights(self, moe):
        """-> (gate_up_W [E,hidden,2D], gate_up_b [E,2D], down_W [E,D,hidden], down_b [E,hidden]).
        With dequantize=True these are bf16 Parameters on-device, returned by reference."""
        ex = moe.experts
        if isinstance(getattr(ex, "gate_up_proj", None), torch.Tensor):
            return ex.gate_up_proj, ex.gate_up_proj_bias, ex.down_proj, ex.down_proj_bias
        from transformers.integrations.mxfp4 import dequantize as _mxdq
        return (_mxdq(ex.gate_up_proj_blocks, ex.gate_up_proj_scales).to(torch.bfloat16),
                ex.gate_up_proj_bias,
                _mxdq(ex.down_proj_blocks, ex.down_proj_scales).to(torch.bfloat16),
                ex.down_proj_bias)

    def down_colnorms(self, model):
        """down_W[e] is [D, hidden]; intermediate channel c is ROW c -> norm over hidden."""
        out = {}
        for lid in self.cfg.moe_layer_ids:
            li = self.cfg.lid2idx[lid]
            _, _, dn_W, _ = self._weights(self.moe_of(model, lid))
            for e in range(self.cfg.num_routed):
                out[(li, e)] = dn_W[e].detach().float().norm(dim=1).cpu()
        return out

    def instrument(self, model, acc):
        """gpt-oss batches experts into 3-D tensors, so there is no per-expert module to
        hook. Instead the experts forward is replaced by one that recomputes the exact
        gpt-oss SwiGLU and accumulates on the way through."""
        cfg = self.cfg
        alpha, limit, E = SWIGLU_ALPHA, SWIGLU_LIMIT, cfg.num_routed
        originals = {}

        def make(li, moe):
            gu_W, gu_b, dn_W, dn_b = self._weights(moe)

            def fwd(hidden_states, router_indices=None, routing_weights=None):
                shp = hidden_states.shape
                x = hidden_states.reshape(-1, shp[-1])
                rw = routing_weights.reshape(-1, E)
                out = torch.zeros_like(x)
                for e in range(E):
                    w_e = rw[:, e]
                    mask = w_e > 0
                    ne = int(mask.sum())
                    if ne == 0:
                        continue
                    xe = x[mask]
                    gu = xe @ gu_W[e] + gu_b[e]
                    g = gu[..., 0::2].clamp(max=limit)
                    u = gu[..., 1::2].clamp(min=-limit, max=limit)
                    act = (u + 1) * (g * torch.sigmoid(g * alpha))      # down-proj INPUT
                    af = act.detach().float()
                    acc.g_sq[li, e] += (af * af).sum(0).to(acc.device)
                    acc.g_nact[li, e] += ne

                    ye = act @ dn_W[e] + dn_b[e]
                    yf = ye.detach().float()
                    norms = yf.norm(dim=-1)
                    we = w_e[mask].detach().float()
                    # numerator of  E[g * ||f|| | active]. gate and norm are paired per
                    # token by construction here: both are gathered by the same `mask`.
                    acc.reap_num[li, e] += (we * norms).sum().to(acc.device)
                    acc.cnt[li, e] += ne
                    acc.fp_sum[li, e] += yf.sum(0).to(acc.device)
                    acc.fp_cnt[li, e] += ne
                    out[mask] += w_e[mask, None].to(ye.dtype) * ye
                return out.reshape(shp)

            return fwd

        for lid in cfg.moe_layer_ids:
            li = cfg.lid2idx[lid]
            moe = self.moe_of(model, lid)
            originals[lid] = moe.experts.forward
            moe.experts.forward = make(li, moe)

        def restore():
            for lid, f in originals.items():
                self.moe_of(model, lid).experts.forward = f

        return restore

    #- surgery-
    def _mk(self, W, dev, bias=None, force_bf16=False):
        if self.cfg.quantize and not force_bf16:
            lin = bnb.nn.Linear4bit(W.shape[1], W.shape[0], bias=bias is not None,
                                    compute_dtype=torch.bfloat16, quant_type="nf4")
            lin.weight = bnb.nn.Params4bit(W.contiguous(), requires_grad=False,
                                           quant_type="nf4", compress_statistics=True)
        else:
            lin = nn.Linear(W.shape[1], W.shape[0], bias=bias is not None,
                            dtype=torch.bfloat16)
            lin.weight = nn.Parameter(W.to(torch.bfloat16).contiguous(), requires_grad=False)
        if bias is not None:
            lin.bias = nn.Parameter(bias.to(torch.bfloat16).contiguous(), requires_grad=False)
        return lin.to(dev)

    def structural_removal(self, model, keep):
        cfg = self.cfg
        by_layer = {}
        for (li, e), kept in keep.items():
            d = by_layer.setdefault(li, {"surv": {}, "dead": []})
            (d["dead"].append(e) if kept.numel() == 0 else d["surv"].__setitem__(e, kept))

        n_slim = n_dead = 0
        for li, info in by_layer.items():
            moe = self.moe_of(model, cfg.idx2lid[li])
            gu_W, gu_b, dn_W, dn_b = self._weights(moe)
            dev = gu_W.device
            experts = {}
            for e, kept in info["surv"].items():
                kept = kept.to(torch.long)
                W = gu_W[e]                                   # [hidden, 2D] interleaved
                gW = W[:, 0::2][:, kept].t().contiguous()      # [k, hidden]
                uW = W[:, 1::2][:, kept].t().contiguous()
                b = gu_b[e]
                experts[e] = (self._mk(gW, dev, bias=b[0::2][kept].contiguous()),
                              self._mk(uW, dev, bias=b[1::2][kept].contiguous()),
                              self._mk(dn_W[e][kept, :].t().contiguous(), dev,
                                       bias=dn_b[e].contiguous(), force_bf16=cfg.w2_bf16))
                n_slim += 1
            moe.experts = PrunedGptOssExperts(experts, cfg.num_routed).to(dev)
            if info["dead"]:
                GateMasker(moe.router, info["dead"])
                n_dead += len(info["dead"])
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  surgery [{cfg.prec_tag}] | {n_slim} slimmed, {n_dead} dropped "
              f"| VRAM ~ {vram_gb():.1f} GB")

    def alive_expert_linears(self, model):
        for lid in self.cfg.moe_layer_ids:
            ex = self.moe_of(model, lid).experts
            if not isinstance(ex, PrunedGptOssExperts):
                continue
            for proj, bank in (("gate", ex.gate), ("up", ex.up), ("down", ex.down)):
                for e in list(bank.keys()):
                    yield f"L{lid}_E{e}_{proj}", bank, e, bank[e]

    #- text-
    def encode_calib(self, tok, text):
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       return_tensors="pt")[0]

    def prompt_ids(self, tok, user_text):
        msgs = [{"role": "user", "content": user_text}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           reasoning_effort=GEN_REASONING)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True)

    # DISABLED pending diagnosis. The hypothesis was that gpt-oss's ~272 raw-text ppl was
    # surprise at the FORMAT, so scoring inside the harmony final channel would fix it.
    # It did not: measured 272 -> 747, i.e. the prefix made it WORSE. The machinery stays
    # (it is a no-op while this is False and other models rely on the empty default), but
    # the claim does not. Run hpp.selftest.diagnose_perplexity before re-enabling.
    USE_HARMONY_PPL_PREFIX = False

    def ppl_prefix_ids(self, tok):
        if not self.USE_HARMONY_PPL_PREFIX:
            return []
        msgs = [{"role": "user", "content": PPL_PREFIX_PROMPT}]
        try:
            ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               reasoning_effort=GEN_REASONING))
        except TypeError:
            ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True))
        # add_generation_prompt stops at `<|start|>assistant`; open the final channel so the
        # text is scored where an ANSWER lives, not where the chain of thought lives.
        # Guarded: if the template already opened a channel, leave it alone.
        if "<|channel|>" not in tok.decode(ids, skip_special_tokens=False):
            try:
                ids += tok.encode("<|channel|>final<|message|>", add_special_tokens=False)
            except Exception as ex:
                print(f"  [ppl] could not append harmony final-channel opener ({ex})")
        return ids

    def final_text(self, generated):
        """Pull the harmony `final` channel. Without this the extractors see the whole
        chain of thought and pick the wrong number / never find the code fence."""
        parts = _FINAL_CHANNEL.split(generated)
        tail = parts[-1] if len(parts) > 1 else generated
        return tail.split("<|return|>")[0].split("<|end|>")[0]


def adapter(cfg):
    return GptOssAdapter(cfg)