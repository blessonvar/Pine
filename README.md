# HPP: Hierarchical Projection Pruning of MoE LLMs

**Keep the most, safely.** Coarse (whole-expert dropping) + fine (per-channel slimming),
driven by one calibrated energy constant `gamma` and two allocation exponents.

```
total budget -> layer budgets (ALPHA) -> dynamic expert drop (gamma floors) -> survivor widths (BETA) -> channels
```

| symbol | role |
|---|---|
| `gamma` | per-expert energy-retention threshold -> each expert's channel FLOOR |
| `alpha` | LAYER allocation exponent (depth-aware budget split) |
| `beta`  | EXPERT allocation exponent (saliency-weighted survivor widths) |

Special cases recovered: high `gamma` -> REAP (drop whole experts); 

## Layout

```
hpp/
  config.py         every knob. One HPPConfig per (model, precision). Derived values are
                    properties, so they cannot drift.
  adapter.py        the ONLY thing that varies per architecture (4 methods).
  utils.py          seed / VRAM / pickle / run_tag
  calibration.py    packed REAP-style windows, one builder for all models
  profiler.py       ONE forward pass -> every statistic
  scores.py         layer_imp (ALPHA) | expert_imp (BETA) | reap_imp (baseline) | cscore
  floors.py         gamma -> per-expert channel floor
  allocator.py      water-fill stages 1-3 + channel selection
  gamma.py          cliff probe -> answer key -> gamma*
  baselines.py      REAP / SlimMoE / drop-probe as keep-dicts
  evaluation.py     ppl | MC (lm-eval) | Math | Code (EvalPlus CLI) | latency
  results.py        CSV logging, before/after table
  reconstruction.py LoRA repair on C4
  runners.py        run_baseline / run_hpp / run_reap / run_sweep / setup
  plots/            style.py is the single visual language; one figure per metric
  models/           one file per architecture: config + adapter
notebooks/          thin: choose knobs, call runners
```

## Run

```python
from hpp.models import build
from hpp import runners, gamma as gamma_mod, reconstruction

cfg, adapter = build("gptoss20b", quantize=False)
scores, prof = runners.setup(cfg, adapter)          # profile + scores + diagnostics
cfg.gamma, _ = gamma_mod.calibrate_gamma(cfg, scores)
runners.run_baseline(cfg, adapter, "fast_sweep")
runners.run_sweep(cfg, adapter, scores, profile="fast_sweep")
reconstruction.run_all_reconstructions(cfg, adapter, scores)
```

## Protocol

* **Sparsities** 25 / 50 / 75%. Exactly reachable by REAP on all four architectures
  (it quantises to `n_prune/E`, floored at `E - top_k`), so every point is a matched comparison.
  90% is excluded because Mixtral (E=8, top-2) saturates at 75%.
* **Reconstruction** at 75%, C4, 500 steps.
* **Categories** Math = mean(GSM8K, MATH-500) · MC = 8-suite mean · Code = mean(HumanEval+, MBPP+)
  = REAP's `Eval+` · plus WikiText-2 perplexity. Accuracies are fractions (0-1).
* **Figures** plot `actual_sparsity`, never the requested value.

## Adding a model

One file in `hpp/models/`: a `config()` factory and a `ModelAdapter`. If its experts are a
ModuleList of MLPs, subclass `UnfusedAdapter` (+ `LogitRouterMixin` if the router returns
logits) and you only declare `PROJ` / `MOE_ATTR` / `load`.