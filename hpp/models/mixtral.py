"""Mixtral-8x7B-Instruct: 32 layers all MoE, 8 routed experts (top-2), no shared experts.
Expert = MixtralBlockSparseTop2MLP(w1 = gate, w3 = up, w2 = down). Router
= block_sparse_moe.gate -> LOGITS, so dropping masks the logit.

E=8 with top-2 is the tightest architecture in the study: at 75% sparsity REAP has already
dropped 6 of 8 and hit its floor, which is why the sweep stops at 75% for every model --
past that there is no REAP point to compare against.
"""
from __future__ import annotations

import torch

from hpp.config import HPPConfig
from hpp.models._unfused import LogitRouterMixin, UnfusedAdapter
from hpp.utils import vram_gb


def config(quantize: bool = True, **overrides) -> HPPConfig:
    cfg = HPPConfig(
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1", model_tag="mixtral8x7b",
        num_layers_total=32, num_routed=8, moe_inter=14336, hidden=4096, top_k=2,
        first_dense=0, n_shared=0, quantize=quantize,
        calib_drop_sweep=(4, 3, 2, 1),          # 8 experts, top-2 -> at most 6 dropped
        eval_batch_size=4, gpu_ids="0",
        base_cache="cache_mixtral", base_res="results_mixtral",
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"HPPConfig has no field {k!r}")
        setattr(cfg, k, v)
    return cfg


class MixtralAdapter(LogitRouterMixin, UnfusedAdapter):
    PROJ = ("w1", "w3", "w2")          # gate, up, down
    MOE_ATTR = "block_sparse_moe"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.cfg
        tok = AutoTokenizer.from_pretrained(cfg.model_name)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = dict(device_map="auto", dtype=torch.bfloat16, trust_remote_code=True,
                  attn_implementation=cfg.attn_impl)
        if cfg.quantize:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name, **kw)
        model.eval()
        print(f"  model loaded [{cfg.prec_tag}] | weights VRAM ~ {vram_gb():.1f} GB")
        return tok, model


def adapter(cfg):
    return MixtralAdapter(cfg)
