"""Qwen3-30B-A3B: 48 layers all MoE, 128 routed experts (top-8), no shared experts.
Expert = Qwen3MoeMLP(gate_proj/up_proj/down_proj), SiLU-gated. Router = nn.Linear ->
LOGITS; softmax/top-k happen inside Qwen3MoeSparseMoeBlock, so experts are dropped by
masking the LOGIT (not by rewriting an (idx, weight) tuple).

Requires an UNFUSED expert layout: `pip install 'transformers>=4.51,<4.55'`.
Reasoning is off for every generation (REAP's setting).
"""
from __future__ import annotations

import torch

from hpp.config import HPPConfig
from hpp.models._unfused import LogitRouterMixin, UnfusedAdapter
from hpp.utils import vram_gb

ENABLE_THINKING = False


def config(quantize: bool = False, **overrides) -> HPPConfig:
    cfg = HPPConfig(
        model_name="Qwen/Qwen3-30B-A3B", model_tag="qwen3_30b_a3b",
        num_layers_total=48, num_routed=128, moe_inter=768, hidden=2048, top_k=8,
        first_dense=0, n_shared=0, quantize=quantize,
        calib_drop_sweep=(96, 80, 64, 48),      # 128 experts -> survivors 32/48/64/80
        eval_batch_size=4, gpu_ids="0", max_mem_per_gpu="44GiB",
        base_cache="cache_qwen3", base_res="results_qwen3",
    )
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"HPPConfig has no field {k!r}")
        setattr(cfg, k, v)
    return cfg


class Qwen3Adapter(LogitRouterMixin, UnfusedAdapter):
    MOE_ATTR = "mlp"

    def load(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        cfg = self.cfg
        src = cfg.model_local_dir or cfg.model_name
        local = bool(cfg.model_local_dir)
        tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True, local_files_only=local)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        n_dev = max(1, len(cfg.gpu_ids.split(",")))
        max_memory = None
        if cfg.max_mem_per_gpu:
            max_memory = {i: cfg.max_mem_per_gpu for i in range(n_dev)}
            max_memory["cpu"] = "120GiB"
        kw = dict(device_map="auto", max_memory=max_memory, local_files_only=local,
                  dtype=torch.bfloat16, attn_implementation=cfg.attn_impl,
                  trust_remote_code=True)
        if cfg.quantize:
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(src, **kw)
        model.eval()
        self._assert_arch(model)
        print(f"  model loaded [{cfg.prec_tag}|attn={cfg.attn_impl}] "
              f"| weights VRAM ~ {vram_gb():.1f} GB")
        return tok, model

    def _assert_arch(self, model):
        cfg = self.cfg
        for k, want in [("num_experts", cfg.num_routed), ("num_experts_per_tok", cfg.top_k),
                        ("moe_intermediate_size", cfg.moe_inter),
                        ("hidden_size", cfg.hidden),
                        ("num_hidden_layers", cfg.num_layers_total)]:
            got = getattr(model.config, k, None)
            if got is not None and got != want:
                print(f"  [arch WARNING] {k}: config={got} vs code={want}")
        for ly in model.model.layers:
            ex = ly.mlp.experts
            if hasattr(ex, "gate_up_proj") or (len(ex) > 0 and not hasattr(ex[0], "gate_proj")):
                raise RuntimeError("Qwen3 experts are FUSED; this adapter needs the "
                                   "ModuleList layout. pip install 'transformers>=4.51,<4.55'")
        print(f"  arch OK: {cfg.num_moe_layers} MoE layers, ModuleList experts")

    def encode_calib(self, tok, text):
        try:
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           return_tensors="pt",
                                           enable_thinking=ENABLE_THINKING)[0]
        except TypeError:
            return tok.apply_chat_template([{"role": "user", "content": text}],
                                           return_tensors="pt")[0]

    def prompt_ids(self, tok, user_text):
        msgs = [{"role": "user", "content": user_text}]
        try:
            return tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           enable_thinking=ENABLE_THINKING)
        except TypeError:
            return tok.apply_chat_template(msgs, add_generation_prompt=True)


def adapter(cfg):
    return Qwen3Adapter(cfg)
