"""DeepSeek-MoE-16B-Chat: 28 layers, layer 0 DENSE and layers 1-27 MoE (hence
first_dense=1), 64 routed experts (top-6) PLUS 2 shared experts that are immutable to
every method here.

CHAT, not base. The dissertation measures capability RETENTION under pruning, so the
model must have the capability to begin with: -base has no chat template, scored 2-3% on
GSM8K (noise, which then fitted gamma to noise), and forced the code eval down a
completion path no other model in the study uses. -chat is the same architecture the
adapter is unchanged but it can be instructed, which also makes its calibration format
consistent with gpt-oss / Qwen3 / Mixtral-Instruct. REAP evaluates instruct models;
deployment prunes instruct models.

The one structural difference from Qwen/Mixtral: MoEGate does the softmax and top-k
ITSELF and returns (topk_idx, topk_weight, aux_loss). So dropping cannot mask a logit
it must zero the returned weight for tokens routed to a dropped expert and renormalise
(only if the model itself renormalises, i.e. norm_topk_prob).
"""
from __future__ import annotations

import torch

from hpp.config import HPPConfig
from hpp.models._unfused import UnfusedAdapter
from hpp.utils import vram_gb


def config(quantize: bool = False, **overrides) -> HPPConfig:
    cfg = HPPConfig(
        model_name="deepseek-ai/deepseek-moe-16b-chat", model_tag="deepseekmoe16b_chat",
        num_layers_total=28, num_routed=64, moe_inter=1408, hidden=2048, top_k=6,
        first_dense=1, n_shared=2, quantize=quantize,
        calib_drop_sweep=(32, 24, 16, 8),       # 64 experts, top-6 -> at most 58 dropped
        # gamma is calibrated on perplexity (HPPConfig default) for all four models
        # uniformly dense, cheap, and immune to the small-sample ceiling/noise effects
        # that made gsm8k unusable for gpt-oss (8.5% spread across the whole drop sweep).
        eval_batch_size=4, gpu_ids="0",
        base_cache="cache_deepseek_chat", base_res="results_deepseek_chat",
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"HPPConfig has no field {k!r}")
        setattr(cfg, k, v)
    return cfg


class DeepSeekAdapter(UnfusedAdapter):
    MOE_ATTR = "mlp"

    @staticmethod
    def _patch_dynamic_cache_compat():
        """Bridge HF cache API differences used by DeepSeek remote code.

        Without this the KV cache raises, which is what motivated disabling it and a
        disabled cache costs one full forward per generated token (~10-20x) and makes
        decode_tps incomparable with the other models. Keep this; do not disable the cache.
        """
        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            return
        if hasattr(DynamicCache, "get_usable_length"):
            return
        if not hasattr(DynamicCache, "get_seq_length"):
            return

        def _get_usable_length(self, new_seq_length=None, layer_idx=0):
            _ = new_seq_length
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length

        if not hasattr(DynamicCache, "seen_tokens"):
            DynamicCache.seen_tokens = property(lambda self: self.get_seq_length())

        if not hasattr(DynamicCache, "get_max_length"):
            DynamicCache.get_max_length = lambda self: None

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.cfg
        self._patch_dynamic_cache_compat()
        tok = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = dict(device_map="auto", dtype=torch.bfloat16, trust_remote_code=True)
        if cfg.quantize:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
        model.eval()
        self._assert_arch(model)
        if getattr(tok, "chat_template", None) is None:
            print("  [WARNING] no chat template on this tokenizer calibration and every "
                  "generative task will fall back to raw text. Is this the -base model?")
        print(f"  model loaded [{cfg.prec_tag}] | weights VRAM ~ {vram_gb():.1f} GB")
        return tok, model

    def _assert_arch(self, model):
        cfg = self.cfg
        for k, want in [("n_routed_experts", cfg.num_routed),
                        ("n_shared_experts", cfg.n_shared),
                        ("num_experts_per_tok", cfg.top_k),
                        ("moe_intermediate_size", cfg.moe_inter)]:
            got = getattr(model.config, k, None)
            if got is not None and got != want:
                print(f"  [arch WARNING] {k}: config={got} vs code={want}")
        found = [i for i, ly in enumerate(model.model.layers) if hasattr(ly.mlp, "gate")]
        print(f"  arch OK: {len(found)} MoE layers" if found == cfg.moe_layer_ids
              else f"  [arch WARNING] MoE layers {found}")

    # routing: MoEGate returns (idx, weight, aux)
    def read_routing(self, model, out):
        if not isinstance(out, tuple):
            return None
        idx, w = out[0], out[1]
        if not torch.isfinite(w).all():
            return None
        return (idx.detach().reshape(-1, self.cfg.top_k).cpu(),
                w.detach().float().reshape(-1, self.cfg.top_k).cpu())

    def mask_router(self, model, lid, drop_ids):
        drop = torch.tensor(sorted({int(i) for i in drop_ids}), dtype=torch.long)
        if drop.numel() == 0:
            return
        gate = self.router_of(model, lid)
        norm = bool(getattr(gate, "norm_topk_prob", False))   # renormalise only if the model does

        def hook(m, args, out):
            if not isinstance(out, tuple):
                return out
            idx, w = out[0], out[1].clone()
            kill = torch.isin(idx, drop.to(idx.device))
            w = w.masked_fill(kill, 0.0)
            if norm:
                w = w / (w.sum(-1, keepdim=True) + 1e-20)
            return (idx, w) + tuple(out[2:])

        gate.register_forward_hook(hook)


def adapter(cfg):
    return DeepSeekAdapter(cfg)