"""Single source of truth for every knob (replaces CELL 1 of all model notebooks).

One HPPConfig instance describes one model at one precision. Anything that was a
module-level global is a field; anything that was *derived* from those globals
(PREC_TAG, MOE_LAYER_IDS, LID2IDX, the path set, _calib_key) is a property, so it
cannot drift when a field changes mid-session.

Architecture *facts* (SwiGLU clamp limits, harmony system prompt) are NOT here --
they live in hpp/models/<arch>.py, because they describe the model, not the run.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# TASKS AND REPORTING CATEGORIES
#
# Two orthogonal groupings, deliberately kept apart:
#   * HOW a task RUNS    -> EvalProfile.mc_tasks  (log-likelihood, via lm-eval-harness)
#                           EvalProfile.gen_tasks (greedy generation + answer extraction)
#   * HOW a task REPORTS -> CATEGORY below -> math_avg / mc_avg / code_avg
# A task therefore cannot silently belong to one grouping and not the other.
#
# Code is EvalPlus ONLY. The task ids carry the `_plus` suffix on purpose: HumanEval+
# and MBPP+ are the SAME problems as vanilla HumanEval/MBPP but graded with ~80x more
# tests, and vanilla is known to over-estimate correctness. Naming the columns for what
# was actually measured is what stops a test-suite-strictness gap being read as a
# pruning-quality gap when set beside REAP's numbers. code_avg = mean(HE+, MBPP+) is
# REAP's `Eval+` by construction.
#
# Grading is done by EvalPlus's OWN CLI, not by an in-loop checker: generations are
# written to samples.jsonl and passed to `evalplus.evaluate`, which applies EvalPlus's
# sanitisation and time limits. Those defaults ARE the protocol every published Eval+
# number was produced under, so re-implementing the checker would silently produce a
# differently-calibrated score. This is also why there is no code_exec_timeout knob.

MC_8 = ("arc_challenge", "arc_easy", "boolq", "hellaswag",
        "mmlu", "openbookqa", "winogrande", "rte")
MATH_TASKS = ("gsm8k", "math500")
CODE_TASKS = ("humaneval_plus", "mbpp_plus")

# our task id -> the --dataset name the evalplus CLI expects
EVALPLUS_DATASET = {"humaneval_plus": "humaneval", "mbpp_plus": "mbpp"}

CATEGORY: dict[str, str] = {
    **{t: "mc" for t in MC_8},
    **{t: "math" for t in MATH_TASKS},
    **{t: "code" for t in CODE_TASKS},
}
CATEGORIES = ("math", "mc", "code")          # order used in CSVs, tables and figures
AVG_COLS = tuple(f"{c}_avg" for c in CATEGORIES)


@dataclass
class EvalProfile:
    """One evaluation battery.

    ppl_windows : WikiText-2 windows (None = whole test set)  -> reported as `perplexity`
    mc_tasks    : log-likelihood tasks                        -> aggregated into `mc_avg`
    mc_limit    : items per MC task (None = full set)
    mmlu_limit  : PER-SUBJECT override for mmlu. mmlu is a GROUP task over 57 subject
                  subtasks and lm-eval applies `limit` per subtask, so an uncapped mmlu
                  is ~57x any other MC task and silently dominates sweep wall-clock.
                  Kept explicit rather than hidden inside mc_limit. None -> use mc_limit.
    gen_tasks   : task -> item limit (None = full set)        -> aggregated by CATEGORY
    """
    ppl_windows: int | None = 256
    mc_tasks: tuple[str, ...] = ()
    mc_limit: int | None = None
    mmlu_limit: int | None = None
    gen_tasks: dict[str, int | None] = field(default_factory=dict)

    @property
    def is_sweep_like(self) -> bool:
        return bool(self.mc_tasks)

    @property
    def all_tasks(self) -> tuple[str, ...]:
        return tuple(self.mc_tasks) + tuple(self.gen_tasks)

    def limit_for(self, task: str) -> int | None:
        if task == "mmlu":
            return self.mc_limit if self.mmlu_limit is None else self.mmlu_limit
        if task in self.mc_tasks:
            return self.mc_limit
        return self.gen_tasks.get(task)

    def tasks_in(self, category: str) -> tuple[str, ...]:
        return tuple(t for t in self.all_tasks if CATEGORY.get(t) == category)

    @property
    def categories(self) -> tuple[str, ...]:
        """Categories this profile can actually report an average for."""
        return tuple(c for c in CATEGORIES if self.tasks_in(c))


@dataclass
class ReconConfig:
    """LoRA post-pruning repair: does HPP's channel selection preserve more repairable
    structure than REAP's whole-expert selection?

    Scope (state this explicitly in the methodology section):
      * Only the pruned FFN (gate/up/down of SURVIVING experts) is adapted. Router,
        attention and any shared experts stay frozen, so the comparison isolates FFN
        capacity lost to pruning  not router or attention adaptation.
      * train_src (C4, generic) is disjoint from BOTH the calibration mix (which chose
        what to prune) and the eval corpus. The held-out eval windows come from
        cfg.ppl_dataset, so the repair curve and the reported perplexity are the same
        corpus by construction.
      * HPP and REAP get identical seed, data order, LR schedule and step budget.
        Trainable parameter counts are intentionally NOT matched: HPP survivors are
        narrower on average, so HPP has fewer LoRA parameters at the same rank. If HPP
        still recovers as well or better, that is a stronger result, not a confound.
    """
    sparsities: tuple[float, ...] = (0.75,)
    steps: int = 500
    eval_every: int = 50
    lr: float = 2e-4
    batch: int = 2                      # micro-batch
    grad_accum: int = 4                 # effective batch = batch * grad_accum
    seq_len: int = 512
    grad_ckpt: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_src: tuple[str, str] = ("allenai/c4", "realnewslike")
    eval_hold: int = 64                 # fixed held-out windows, identical across steps


def default_profiles() -> dict[str, EvalProfile]:
    """Three profiles, no overlap:

    single      the gamma cliff-probe loop (run_drop_probe -> answer_key_from_drops).
                  It reads nothing but calib_primary_metric and runs many times per
                  sparsity, so it deliberately excludes MC / MATH-500 / code.
    fast_sweep  alpha/beta/sparsity grid search. Moderate power, quick turnaround.
    full_sweep  the dissertation-table numbers. Full benchmarks wherever feasible
                  (humaneval_plus is a fixed 164 problems and always cheap; math500 is
                  a fixed 500 by construction; gsm8k/mbpp_plus run their full splits).
    """
    return {
        "single": EvalProfile(
            ppl_windows=128, mc_tasks=(), mc_limit=None,
            gen_tasks={"gsm8k": 129},
        ),
        "fast_sweep": EvalProfile(
            ppl_windows=128, mc_tasks=MC_8, mc_limit=128, mmlu_limit=20,
            # code limits are None on purpose: EvalPlus grades a COMPLETE sample set and
            # rejects partial ones, so a limit here would be silently ignored.
            gen_tasks={"gsm8k": 128, "math500": 128,
                       "humaneval_plus": None, "mbpp_plus": None},
        ),
        "full_sweep": EvalProfile(
            ppl_windows=None, mc_tasks=MC_8, mc_limit=None, mmlu_limit=100,
            gen_tasks={"gsm8k": 500, "math500": None,
                       "humaneval_plus": None, "mbpp_plus": None},
        ),
    }


@dataclass
class HPPConfig:
    # identity 
    model_name: str
    model_tag: str

    # architecture 
    num_layers_total: int
    num_routed: int
    moe_inter: int              # D: expert intermediate width
    hidden: int                 # d: hidden / residual width
    top_k: int
    first_dense: int = 0        # index of first MoE layer (DeepSeek=1, others=0)
    n_shared: int = 0

    # precision 
    quantize: bool = False      # True -> NF4 survivors, False -> bf16 survivors
    w2_bf16: bool = False       # force survivor down_proj to bf16 even when quantize=True

    # pruning 
    gamma: float | None = None  # set by calibrate_gamma()
    alpha: float = 1.0          # LAYER allocation exponent
    beta: float = 1.0           # EXPERT allocation exponent
    energy_power: float = 2.0
    min_keep: int = 64
    round_to: int = 64
    # Standard grid for every model. REAP drops whole experts, so its sparsity quantises
    # to n_prune/E and is floored at E - top_k; 25/50/75 happen to be EXACTLY reachable on
    # all four architectures (incl. Mixtral, E=8/k=2, which saturates at 75%), so no
    # per-model override is needed and every point is a matched HPP-vs-REAP comparison.
    sparsities: tuple[float, ...] = (0.25, 0.50, 0.75)

    # calibration (packed, REAP-style, for every model) -
    calib_total_windows: int = 1024
    calib_max_len: int = 2048
    conv_checkpoints: tuple[int, ...] = (25, 50, 100, 200, 400, 1000)
    conv_threshold: float = 0.02

    # reconstruction 
    recon: ReconConfig = field(default_factory=ReconConfig)

    # evaluation 
    eval_profiles: dict[str, EvalProfile] = field(default_factory=default_profiles)
    ppl_dataset: tuple[str, str] = ("wikitext", "wikitext-2-raw-v1")
    ppl_seq_len: int = 2048
    eval_batch_size: int = 2
    num_fewshot_mc: int = 0     # REAP: zero-shot, standard log-likelihood
    # Decoding WITHOUT a KV cache costs one full forward per generated token instead of a
    # prefill plus cheap steps  a 10-20x slowdown that turns a fast_sweep into hours.
    # Only set False if a model's cache API genuinely cannot be made to work, and say so.
    use_kv_cache: bool = True
    gen_max_new: int = 512      # gsm8k / math500 decode budget
    code_max_new: int = 768     # code needs a longer budget than a numeric answer

    # gamma calibration -
    calib_primary_metric: str = "perplexity"
    calib_collapse_frac: float = 0.90
    calib_sparsity: float = 0.50
    calib_drop_sweep: tuple[int, ...] = (16, 12, 8, 4)
    gamma_grid: np.ndarray = field(default_factory=lambda: np.linspace(0.50, 0.999, 60))

    # runtime 
    seed: int = 42
    gpu_ids: str = "0"
    max_mem_per_gpu: str | None = None
    model_local_dir: str = ""
    attn_impl: str = "eager"

    # paths (base_* default to the model tag) 
    base_cache: str | None = None
    base_res: str | None = None

    def __post_init__(self):
        if self.base_cache is None:
            self.base_cache = f"cache_{self.model_tag}"
        if self.base_res is None:
            self.base_res = f"results_{self.model_tag}"

    # derived: precision / layers 
    @property
    def prec_tag(self) -> str:
        return "nf4" if self.quantize else "bf16"

    @property
    def moe_layer_ids(self) -> list[int]:
        return list(range(self.first_dense, self.num_layers_total))

    @property
    def num_moe_layers(self) -> int:
        return self.num_layers_total - self.first_dense

    @property
    def lid2idx(self) -> dict[int, int]:
        return {lid: i for i, lid in enumerate(self.moe_layer_ids)}

    @property
    def idx2lid(self) -> dict[int, int]:
        return {i: lid for i, lid in enumerate(self.moe_layer_ids)}

    @property
    def total_channels(self) -> int:
        """Denominator of every sparsity figure: Lm * E * D."""
        return self.num_moe_layers * self.num_routed * self.moe_inter

    def reap_actual_sparsity(self, sparsity: float) -> float:
        """What REAP can ACTUALLY reach for a requested sparsity on this architecture.
        Figures must plot actual_sparsity, never the requested value."""
        n_prune = min(round(sparsity * self.num_routed), self.num_routed - self.top_k)
        return n_prune / self.num_routed

    # derived: paths 
    @property
    def cache_dir(self) -> Path:
        return Path(f"{self.base_cache}_{self.prec_tag}")

    @property
    def res_dir(self) -> Path:
        return Path(f"{self.base_res}_{self.prec_tag}")

    @property
    def fig_dir(self) -> Path:
        return self.res_dir / "figures"

    @property
    def log_dir(self) -> Path:
        return self.res_dir / "logs"

    @property
    def conv_dir(self) -> Path:
        return self.res_dir / "convergence"

    @property
    def calib_dir(self) -> Path:
        return self.res_dir / "calibration"

    @property
    def pca_dir(self) -> Path:
        return self.res_dir / "pca"

    @property
    def recon_dir(self) -> Path:
        return self.res_dir / "reconstruction"

    @property
    def code_samples_dir(self) -> Path:
        """Generations handed to the evalplus CLI: <dir>/<tag>/<dataset>/samples.jsonl"""
        return self.res_dir / "code_samples"

    @property
    def single_csv(self) -> Path:
        return self.res_dir / "single_runs.csv"

    @property
    def sweep_csv(self) -> Path:
        return self.res_dir / "sweep.csv"

    @property
    def sweep_recon_csv(self) -> Path:
        return self.res_dir / "sweep_reconstructed.csv"

    @property
    def dropkey_csv(self) -> Path:
        return self.calib_dir / f"drop_keys_{self.model_tag}.csv"

    def ensure_dirs(self) -> "HPPConfig":
        for d in (self.cache_dir, self.fig_dir, self.log_dir, self.conv_dir,
                  self.calib_dir, self.pca_dir, self.recon_dir, self.code_samples_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # derived: cache identity 
    def calib_key(self) -> str:
        """Identity of a profile/scores cache. Contains ONLY what the statistics depend on:
        the weights (model + precision  4-bit and bf16 give genuinely different scores)
        and the calibration corpus shape. Evaluation, pruning and reconstruction knobs are
        deliberately absent, so changing the eval battery never invalidates a profile."""
        s = f"{self.model_name}|{self.prec_tag}|{self.calib_total_windows}|{self.calib_max_len}|packed"
        return hashlib.md5(s.encode()).hexdigest()[:10]

    @property
    def profile_path(self) -> Path:
        return self.cache_dir / f"profile_{self.model_tag}_{self.calib_key()}.pkl"

    @property
    def scores_path(self) -> Path:
        return self.cache_dir / f"scores_{self.model_tag}_{self.calib_key()}.pkl"

    def profile(self, name: str) -> EvalProfile:
        return self.eval_profiles[name]

    # derived: result schema 
    def metric_cols(self, profile: str) -> list[str]:
        """Stable column order: perplexity, per-task columns grouped by category,
        then the category averages this profile can actually report."""
        p = self.profile(profile)
        cols = ["perplexity"]
        for c in CATEGORIES:
            cols += list(p.tasks_in(c))
        return cols + [f"{c}_avg" for c in p.categories]