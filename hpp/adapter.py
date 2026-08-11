"""The ONLY thing that varies between architectures.

Everything else in the package (floors, allocator, gamma, scores, plots, results,
reconstruction) is identical across Mixtral / DeepSeek / Qwen3 / gpt-oss once the
globals are gone. What genuinely differs is four things, and this is the contract:

  1. loading            (quantisation config, dequantisation of expert weights)
  2. profiler wiring    (per-expert hooks vs an instrumented fused forward)
  3. surgery            (per-Linear replacement vs rebuilding a fused expert bank)
  4. generation/decode  (chat template; harmony channel stripping)

Layer-level influence (pre-LN input vs post-MoE delta) is NOT here: it is identical
everywhere given `moe_of`, so profiler.py owns it.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from hpp.config import HPPConfig


class ModelAdapter(ABC):
    """One instance per (model, precision). Stateless w.r.t. a loaded model: every
    method takes the model it should act on."""

    def __init__(self, cfg: HPPConfig):
        self.cfg = cfg

    # loading
    @abstractmethod
    def load(self):
        """-> (tokenizer, model). Must call model.eval()."""

    @abstractmethod
    def moe_of(self, model, lid):
        """The MoE block of decoder layer `lid` (the module whose output is the MoE delta)."""

    def layernorm_of(self, model, lid):
        """Pre-MoE LayerNorm. Same for every architecture we support."""
        return model.model.layers[lid].post_attention_layernorm

    # profiling
    @abstractmethod
    def down_colnorms(self, model) -> dict:
        """{(li, e): tensor[D]} -- per-intermediate-channel L2 norm of the down projection.
        Fixed weights, so computed once. This is the `colnorm` half of the channel score."""

    @abstractmethod
    def instrument(self, model, acc):
        """Wire whatever is needed to fill `acc` (an hpp.profiler.Accum) per forward pass,
        and return a callable that removes/restores everything."""

    # surgery
    @abstractmethod
    def structural_removal(self, model, keep) -> None:
        """keep: {(li, e): LongTensor of kept channel indices}. Empty -> drop the expert
        entirely and zero+renormalise its router weight. Honours cfg.quantize."""

    @abstractmethod
    def alive_expert_linears(self, model):
        """Yield (key, owner, attr_name, linear) for every gate/up/down Linear of every
        SURVIVING expert, so reconstruction.py can wrap them in LoRA without knowing the
        layout. `key` must be stable across runs (it names the saved LoRA tensors)."""

    # text
    @abstractmethod
    def encode_calib(self, tok, text):
        """Full document -> 1-D LongTensor. NEVER pre-truncate: the packer consumes the
        whole document across window boundaries, so truncating here silently drops tails."""

    @abstractmethod
    def prompt_ids(self, tok, user_text) -> list[int]:
        """A generation prompt as token ids (keeps any special/harmony tokens exact)."""

    def ppl_prefix_ids(self, tok) -> list[int]:
        """Token ids prepended to every perplexity / held-out-loss window, and NOT scored.

        Empty for any model that reads raw text natively (DeepSeek, Qwen3, Mixtral) --
        so for them this is exactly a no-op and their numbers are unchanged.

        gpt-oss overrides it: it is a harmony-native post-trained model, and scoring bare
        prose measures how surprised it is by the FORMAT rather than by the text (~272 ppl
        on WikiText-2 while GSM8K is 0.89 -- the model is fine, the framing was not).
        """
        return []

    def final_text(self, generated: str) -> str:
        """Strip any wrapper the model emits around its answer. Identity by default;
        gpt-oss overrides this to pull the harmony `final` channel."""
        return generated