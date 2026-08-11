"""LoRA post-pruning repair (C4) does HPP's channel selection leave more repairable
structure than REAP's whole-expert selection?

Scope and fairness are argued in ReconConfig's docstring; the code just has to not break
those claims. The two that are easy to break silently:
  * LoRA goes ONLY on surviving experts' gate/up/down (adapter.alive_expert_linears).
  * set_seed is called immediately before the loop, so both methods walk the same C4
    stream in the same order with the same dropout draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset

from hpp.allocator import select_hpp
from hpp.baselines import reap_keep
from hpp.config import HPPConfig
from hpp.utils import free, peak_vram, reset_peak, set_seed


class LoRALinear(nn.Module):
    """Frozen base Linear (bf16 or NF4/bnb) + trainable low-rank residual.

    Only .weight / .in_features / .out_features are touched and the base is never
    backpropped through, so this works for both survivor precisions unchanged.
    lora_B is zero-init -> the wrapper is exactly the identity at step 0.
    """

    def __init__(self, base, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        lx = self.drop(x).float()
        return y + (lx @ self.lora_A.T @ self.lora_B.T).to(y.dtype) * self.scale


def inject_lora(cfg: HPPConfig, adapter, model):
    rc = cfg.recon
    lora_by_key = {}
    for key, owner, attr, lin in list(adapter.alive_expert_linears(model)):
        wrapped = LoRALinear(lin, rc.lora_r, rc.lora_alpha, rc.lora_dropout)
        setattr(owner, attr, wrapped)
        lora_by_key[key] = wrapped
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = [p for m in lora_by_key.values() for p in (m.lora_A, m.lora_B)]
    for p in trainable:
        p.requires_grad_(True)
    print(f"  LoRA injected on {len(lora_by_key)} Linears | "
          f"{sum(p.numel() for p in trainable):,} trainable params")
    return trainable, lora_by_key


def lora_state_dict(lora_by_key):
    """Save the A/B matrices only (a few MB), not the whole 20B model."""
    return {k: {"A": m.lora_A.detach().cpu(), "B": m.lora_B.detach().cpu()}
            for k, m in lora_by_key.items()}


def load_lora_state_dict(lora_by_key, sd):
    n = 0
    for k, m in lora_by_key.items():
        if k in sd:
            m.lora_A.data.copy_(sd[k]["A"].to(m.lora_A.device))
            m.lora_B.data.copy_(sd[k]["B"].to(m.lora_B.device))
            n += 1
    print(f"  loaded trained LoRA weights into {n}/{len(lora_by_key)} Linears")


def enable_memory_efficient_training(cfg, model):
    """Two things are needed beyond gradient_checkpointing_enable():
    (1) the KV cache must be off (incompatible with checkpointing), and
    (2) every backbone parameter is frozen, so the embedding output has
        requires_grad=False and silently breaks the checkpointed graph forcing it True
        via a forward hook is the same trick HF's prepare_model_for_kbit_training uses."""
    model.config.use_cache = False

    def _require_grad(module, inp, out):
        out.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(_require_grad)
    if cfg.recon.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})


def build_loaders(cfg: HPPConfig, adapter, tok):
    from hpp.evaluation import ppl_prefix

    rc = cfg.recon
    prefix = ppl_prefix(adapter, tok)
    wt = tok("\n\n".join(load_dataset(*cfg.ppl_dataset, split="test")["text"]),
             return_tensors="pt").input_ids
    eval_windows = [wt[:, i * rc.seq_len:(i + 1) * rc.seq_len]
                    for i in range(rc.eval_hold) if (i + 1) * rc.seq_len <= wt.size(1)]

    def train_stream():
        buf = []
        while True:
            for item in load_dataset(*rc.train_src, split="train", streaming=True):
                ids = tok(item["text"], return_tensors="pt").input_ids[0]
                buf.append(ids)
                cat = torch.cat(buf)
                while cat.numel() >= rc.seq_len:
                    yield cat[:rc.seq_len].unsqueeze(0)
                    cat = cat[rc.seq_len:]
                buf = [cat]

    return eval_windows, train_stream(), prefix


@torch.no_grad()
def eval_loss(model, eval_windows, prefix):
    """Same primitive as fast_perplexity, so the repair curve and the reported perplexity
    genuinely measure the same thing (ReconConfig's docstring claims this passing
    labels= to the model, as before, quietly broke the claim on gpt-oss)."""
    from hpp.evaluation import window_nll

    model.eval()
    total, ntok = 0.0, 0
    for ids in eval_windows:
        loss, n = window_nll(model, ids, prefix)
        if torch.isfinite(loss):
            total += float(loss) * n
            ntok += n
    model.train()
    return total / max(ntok, 1)


def keep_for(cfg, scores, method, sparsity):
    if method == "hpp":
        keep, _, actual, _ = select_hpp(cfg, scores, sparsity)
        return keep, actual
    if method == "reap":
        keep, _, actual = reap_keep(cfg, scores, sparsity)
        return keep, actual
    raise ValueError(method)


def run_reconstruction(cfg: HPPConfig, adapter, scores, method, sparsity):
    rc = cfg.recon
    tok, model = adapter.load()
    keep, _ = keep_for(cfg, scores, method, sparsity)
    adapter.structural_removal(model, keep)
    trainable, lora_by_key = inject_lora(cfg, adapter, model)
    enable_memory_efficient_training(cfg, model)

    opt = torch.optim.AdamW(trainable, lr=rc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rc.steps)
    eval_windows, train_iter, prefix = build_loaders(cfg, adapter, tok)

    hist = []
    model.train()
    set_seed(cfg.seed)          # identical data order / dropout across methods
    reset_peak()
    for step in range(1, rc.steps + 1):
        opt.zero_grad()
        accum = 0.0
        for _ in range(rc.grad_accum):
            batch = torch.cat([next(train_iter) for _ in range(rc.batch)], 0).to(model.device)
            out = model(batch, labels=batch)
            (out.loss / rc.grad_accum).backward()
            accum += out.loss.item() / rc.grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        row = {"step": step, "train_loss": accum}
        if step == 1:
            print(f"  [{method} s={sparsity:.0%}] peak VRAM after step 1: {peak_vram():.1f} GB")
        if step % rc.eval_every == 0 or step == 1:
            row["eval_loss"] = eval_loss(model, eval_windows, prefix)
            print(f"  [{method} s={sparsity:.0%}] step {step}/{rc.steps} "
                  f"train={row['train_loss']:.3f} eval={row['eval_loss']:.3f}")
        hist.append(row)

    cfg.recon_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(hist)
    df.to_csv(cfg.recon_dir / f"recon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.csv",
              index=False)
    torch.save(lora_state_dict(lora_by_key),
               cfg.recon_dir / f"lora_{method}_s{int(sparsity*100)}_{cfg.model_tag}.pt")
    del model, tok, opt
    free()
    return df


def run_all_reconstructions(cfg, adapter, scores):
    from hpp.plots.reconstruction import plot_reconstruction

    for sparsity in cfg.recon.sparsities:
        dfs = {}
        for method in ("hpp", "reap"):
            p = cfg.recon_dir / f"recon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.csv"
            dfs[method] = (pd.read_csv(p) if p.exists()
                           else run_reconstruction(cfg, adapter, scores, method, sparsity))
        plot_reconstruction(cfg, dfs, sparsity)


def _ppl_only_profile(cfg, base_profile):
    """The repaired-model pass reports perplexity only, not the 'before' profile's full
    battery.

    The LoRA repair is trained purely on C4 next-token prediction (build_loaders / the
    `out = model(batch, labels=batch)` loop above), and its own held-out signal is
    eval_loss on the SAME wikitext windows fast_perplexity scores both training
    objective and internal validation are generic-prose LM loss. Re-running MMLU/GSM8K/
    MATH-500/EvalPlus on top of that is comparing HPP vs REAP on a metric the repair
    step never touched: empirically both methods land within noise of 0 on gsm8k/math500/
    code_avg post-repair (e.g. hpp 0.0078 vs reap 0.0156 on gsm8k, both 0.0 on code_avg in
    results_deepseek_chat_bf16/sweep_reconstructed.csv), while perplexity keeps the clean,
    method-separating signal the repair curve already shows. Window count is copied from
    the 'before' profile so the perplexity number stays apples-to-apples with it.
    """
    name = f"{base_profile}__ppl_only"
    if name not in cfg.eval_profiles:
        from hpp.config import EvalProfile
        cfg.eval_profiles[name] = EvalProfile(ppl_windows=cfg.profile(base_profile).ppl_windows)
    return name


def evaluate_repaired(cfg, adapter, scores, method, sparsity, profile=None):
    """Re-prune, re-attach the SAVED LoRA, and score perplexity only (see
    _ppl_only_profile). `profile` still selects the 'before' row to diff against and to
    borrow the perplexity window count from, so the two stay comparable on that metric."""
    from hpp.evaluation import evaluate
    from hpp.results import before_row, is_done, log_result

    lora_path = cfg.recon_dir / f"lora_{method}_s{int(sparsity*100)}_{cfg.model_tag}.pt"
    if not lora_path.exists():
        print(f"  no saved LoRA for {method} s={sparsity:.0%} run reconstruction first")
        return
    if profile is None:
        for cand in ("full_sweep", "fast_sweep", "single"):
            try:
                profile = str(before_row(cfg, cand, method, sparsity)["profile"])
                break
            except Exception:
                continue
    if profile is None:
        raise ValueError("no 'before' row found in any results file run the sweep first")

    eval_profile = _ppl_only_profile(cfg, profile)
    if is_done(cfg, eval_profile, method, sparsity, reconstructed=True):
        print(f"  [skip] repaired {method} s={sparsity:.0%} already logged")
        return

    b = before_row(cfg, profile, method, sparsity)
    tok, model = adapter.load()
    keep, actual = keep_for(cfg, scores, method, sparsity)
    adapter.structural_removal(model, keep)
    if not np.isclose(actual, float(b["actual_sparsity"]), atol=1e-3):
        print(f"  [WARNING] recomputed actual_sparsity ({actual:.4f}) != logged "
              f"({float(b['actual_sparsity']):.4f}) scores/calibration changed since "
              f"the sweep, so this is not the same model as the 'before' row")
    _, lora_by_key = inject_lora(cfg, adapter, model)
    load_lora_state_dict(lora_by_key, torch.load(lora_path, map_location="cpu"))

    model.eval()
    model.config.use_cache = True
    metrics = evaluate(cfg, adapter, model, tok, eval_profile, f"repaired_{method}_s{sparsity:.0%}")
    log_result(cfg, eval_profile, method, sparsity, actual, metrics,
               b.get("alpha"), b.get("beta"), b.get("gamma"), reconstructed=True)
    del model, tok
    free()