"""Ablation: does calibrating ONLY on math data change HPP-vs-REAP math performance?

REAP's paper reports that domain-specific calibration skews a pruned model heavily
toward that domain's evals. This tests that on the existing pipeline: build a math-only
calibration set (GSM8K + MATH train, no round-robin with the other domains in
hpp.calibration), score the model on it, then prune with the SAME select_hpp/reap_keep
the main sweep uses and evaluate on GSM8K/MATH-500.

Deliberately self-contained so the main pipeline is never at risk:
  * its own calibration builder (reuses hpp.calibration's _domain_stream, but only over
    math sources the main build_calibration is untouched)
  * its own cache files (profile_*_mathcalib_*.pkl / scores_*_mathcalib_*.pkl), so this
    can never collide with or overwrite the main sweep's scores/profile cache
  * its own results CSV (ablation_math_calib.csv), never sweep.csv
  * gamma is REUSED from cfg.gamma (already calibrated on mixed-domain scores) rather
    than recalibrated gamma is meant to be a property of the model's tolerance, not
    the calibration domain (see hpp/gamma.py's module docstring); floors are still
    recomputed fresh from the math-domain channel scores using that gamma.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch

from datasets import load_dataset

from hpp.allocator import select_hpp
from hpp.baselines import reap_keep
from hpp.calibration import _domain_stream
from hpp.config import EvalProfile, HPPConfig
from hpp.utils import free, load_pkl, peak_vram, reset_peak, save_pkl, set_seed

MATH_SOURCES = {
    "gsm8k": ("gsm8k", "main", "train", lambda x: x.get("question", "")),
    "math":  ("qwedsacf/competition_math", None, "train", lambda x: x.get("problem", "")),
}
MATH_SEED = {"gsm8k": "What is 23 times 47?", "math": "Solve for x: 2x + 3 = 7."}


MATH_QA_SOURCES = {
    "orca": ("microsoft/orca-math-word-problems-200k", None, "train",
             lambda x: (x.get("question", ""), x.get("answer", ""))),
}
MATH_QA_SEED = {
    "orca": ("Jungkook is in 5th place. How many people finished before him?",
             "If Jungkook is in 5th place, then 4 people finished before him.\n#### 4"),
}


def build_math_calibration(cfg: HPPConfig, tok, adapter):
    """Same packing scheme as hpp.calibration.build_calibration (no truncation, full
    windows, nothing dropped) but round-robins over GSM8K + MATH train only, so every
    packed window is math-domain instead of mixed."""
    seq_len = cfg.calib_max_len
    total = cfg.calib_total_windows
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    sep = torch.tensor([eos_id], dtype=torch.long)

    streams = {d: _domain_stream(p, c, s, g, MATH_SEED[d])
               for d, (p, c, s, g) in MATH_SOURCES.items()}
    domains = list(streams)

    buf, windows, n_docs, di = [], [], 0, 0
    while len(windows) < total:
        dom = domains[di % len(domains)]
        di += 1
        text = next(streams[dom])
        n_docs += 1
        buf.append(adapter.encode_calib(tok, text))
        buf.append(sep)
        cat = torch.cat(buf)
        while cat.numel() >= seq_len and len(windows) < total:
            windows.append(cat[:seq_len].unsqueeze(0))   # full window, nothing dropped
            cat = cat[seq_len:]                          # remainder carries into next doc
        buf = [cat]

    random.shuffle(windows)
    print(f"  math calibration: {len(windows)} packed windows x {seq_len} tokens "
          f"from {n_docs} source docs (GSM8K + MATH train only)")
    return windows


def _cache_path(cfg: HPPConfig, kind: str):
    return cfg.cache_dir / (f"{kind}_{cfg.model_tag}_mathcalib_"
                             f"{cfg.calib_total_windows}x{cfg.calib_max_len}.pkl")


def math_calib_scores(cfg: HPPConfig, adapter):
    """cached_profile/cached_scores equivalent, but reading the raw profile()/
    compute_scores() functions directly and writing to filenames namespaced with
    'mathcalib' never touches or reads the main pipeline's scores_*/profile_*.pkl."""
    from hpp.profiler import profile as run_profile
    from hpp.scores import compute_scores

    prof_p, sc_p = _cache_path(cfg, "profile"), _cache_path(cfg, "scores")
    if prof_p.exists() and sc_p.exists():
        print(f"  cache hit (math-calib scores) [{cfg.prec_tag}]")
        return load_pkl(sc_p), load_pkl(prof_p)

    print(f"  cache miss (math-calib scores) [{cfg.prec_tag}] loading model")
    tok, model = adapter.load()
    calib = build_math_calibration(cfg, tok, adapter)
    prof = run_profile(cfg, adapter, model, calib)
    save_pkl(prof, prof_p)
    scores = compute_scores(cfg, prof)
    save_pkl(scores, sc_p)
    del model, tok, calib
    free()
    return scores, prof


def _ensure_eval_profile(cfg: HPPConfig, gsm8k_limit, math500_limit):
    """Perplexity + GSM8K + MATH-500 only MC/code are off-topic for a math-calibration
    ablation. Inserted at runtime so this never touches config.py's default_profiles()."""
    name = "math_calib_eval"
    if name not in cfg.eval_profiles:
        cfg.eval_profiles[name] = EvalProfile(
            ppl_windows=128, gen_tasks={"gsm8k": gsm8k_limit, "math500": math500_limit})
    return name


def _ensure_math_recon_eval_profile(cfg: HPPConfig, gsm8k_limit, math500_limit, include_code):
    name = "math_recon_eval_code" if include_code else "math_recon_eval"
    if name not in cfg.eval_profiles:
        gen_tasks = {"gsm8k": gsm8k_limit, "math500": math500_limit}
        if include_code:
            gen_tasks.update({"humaneval_plus": None, "mbpp_plus": None})
        cfg.eval_profiles[name] = EvalProfile(ppl_windows=128, gen_tasks=gen_tasks)
    return name


def run_math_calib_ablation(cfg: HPPConfig, adapter, sparsities=(0.25, 0.50, 0.75),
                            gamma=None, gsm8k_limit=128, math500_limit=128):
    """Prune with math-only calibration scores at each sparsity, both methods, and log
    perplexity/gsm8k/math500 to results_.../ablation_math_calib.csv (never sweep.csv)."""
    from hpp.evaluation import evaluate

    gamma = cfg.gamma if gamma is None else gamma
    assert gamma is not None, "cfg.gamma must be calibrated first (reused as-is here, not recalibrated for math)"
    eval_profile = _ensure_eval_profile(cfg, gsm8k_limit, math500_limit)
    scores, _ = math_calib_scores(cfg, adapter)

    out_path = cfg.res_dir / "ablation_math_calib.csv"
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["method"], round(float(r["sparsity"]), 4)) for r in rows}

    for sparsity in sparsities:
        for method in ("hpp", "reap"):
            if (method, round(sparsity, 4)) in done:
                print(f"  [skip] math-calib {method} s={sparsity:.0%} already logged")
                continue
            print(f"\n=== math-calib {method.upper()} [{cfg.prec_tag}] | s={sparsity:.0%} ===")
            if method == "hpp":
                keep, _, actual, _ = select_hpp(cfg, scores, sparsity, gamma=gamma)
            else:
                keep, _, actual = reap_keep(cfg, scores, sparsity)
            tok, model = adapter.load()
            adapter.structural_removal(model, keep)
            metrics = evaluate(cfg, adapter, model, tok, eval_profile,
                                f"mathcalib_{method}_s{sparsity:.0%}")
            del model, tok, keep
            free()
            row = {"method": method, "sparsity": sparsity, "actual_sparsity": actual, **metrics}
            gsm, m500 = metrics.get("gsm8k"), metrics.get("math500")
            if gsm is not None and m500 is not None:
                row["math_avg"] = (gsm + m500) / 2
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    if "math_avg" not in df.columns or df["math_avg"].isna().any():
        df["math_avg"] = df[["gsm8k", "math500"]].mean(axis=1)   # backfill older/loaded rows
    print(df.to_string(index=False))
    _plot_math_calib(cfg, df)
    return df


def evaluate_math_calib_generalization(cfg: HPPConfig, adapter, sparsity=0.25,
                                       profile=None, gamma=None):
    from hpp.evaluation import evaluate
    from hpp.results import before_row

    gamma = cfg.gamma if gamma is None else gamma
    assert gamma is not None, "cfg.gamma must be calibrated first"
    if profile is None:
        for cand in ("full_sweep", "fast_sweep", "single"):
            try:
                before_row(cfg, cand, "hpp", sparsity)
                profile = cand
                break
            except Exception:
                continue
    if profile is None:
        raise ValueError(f"no mixed-domain 'before' row for s={sparsity:.0%} in any "
                          "results file run the main sweep at this sparsity first")

    scores, _ = math_calib_scores(cfg, adapter)
    out_path = cfg.res_dir / "ablation_math_calib_generalization.csv"
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    done = {(r["method"], round(float(r["sparsity"]), 4)) for r in rows}

    for method in ("hpp", "reap"):
        if (method, round(sparsity, 4)) in done:
            print(f"  [skip] math-calib generalization {method} s={sparsity:.0%} already logged")
            continue
        print(f"\n=== math-calib scores -> FULL profile [{profile}] "
              f"{method.upper()} s={sparsity:.0%} ===")
        if method == "hpp":
            keep, _, actual, _ = select_hpp(cfg, scores, sparsity, gamma=gamma)
        else:
            keep, _, actual = reap_keep(cfg, scores, sparsity)
        tok, model = adapter.load()
        adapter.structural_removal(model, keep)
        metrics = evaluate(cfg, adapter, model, tok, profile,
                            f"mathcalib_gen_{method}_s{sparsity:.0%}")
        del model, tok, keep
        free()
        rows.append({"method": method, "sparsity": sparsity, "actual_sparsity": actual,
                     **metrics})
        pd.DataFrame(rows).to_csv(out_path, index=False)

    df = pd.DataFrame(rows)
    _report_generalization(cfg, df, profile, sparsity)
    _plot_generalization(cfg, df, profile, sparsity)
    return df


def _report_generalization(cfg: HPPConfig, df, profile, sparsity):
    """lower mc_avg/math_avg/code_avg or higher perplexity than the mixed-domain 'before'
    row means math-only calibration collapsed that axis relative to mixed calibration."""
    from hpp.results import before_row

    print(f"\n  mixed-domain calibration (sweep.csv, profile={profile}) vs "
          f"math-only calibration @ s={sparsity:.0%}:")
    cols = [c for c in ("perplexity", "mc_avg", "math_avg", "code_avg") if c in df.columns]
    for method in ("hpp", "reap"):
        try:
            b = before_row(cfg, profile, method, sparsity)
        except Exception as e:
            print(f"  [WARN] no mixed-domain row for {method}: {e}")
            continue
        row = df[df["method"] == method]
        if row.empty:
            continue
        a = row.iloc[0]
        parts = []
        for c in cols:
            bv, av = b.get(c), a.get(c)
            if pd.notna(bv) and pd.notna(av):
                worse = (float(av) > float(bv)) if c == "perplexity" else (float(av) < float(bv))
                parts.append(f"{c}: mixed={float(bv):.4f} math={float(av):.4f}"
                             + (" COLLAPSED" if worse else ""))
        print(f"  {method.upper():5s} " + " | ".join(parts))


def _plot_generalization(cfg: HPPConfig, df, profile, sparsity):
    from hpp.plots.style import METHODS, finalize, new_fig, save
    from hpp.results import before_row

    metrics = [c for c in ("mc_avg", "math_avg", "code_avg") if c in df.columns]
    if not metrics:
        return
    fig, ax = new_fig()
    x = np.arange(len(metrics))
    width = 0.18
    for i, method in enumerate(("hpp", "reap")):
        st = METHODS.get(method, {})
        try:
            b = before_row(cfg, profile, method, sparsity)
        except Exception:
            b = None
        a_rows = df[df["method"] == method]
        a = a_rows.iloc[0] if not a_rows.empty else None
        mixed = [float(b.get(m)) if b is not None and pd.notna(b.get(m)) else np.nan for m in metrics]
        math_only = [float(a.get(m)) if a is not None and pd.notna(a.get(m)) else np.nan for m in metrics]
        base = (i - 0.5) * 2.2 * width
        ax.bar(x + base - width / 2, mixed, width,
               label=f"{st.get('label', method.upper())} (mixed calib)", color=st.get("color"))
        ax.bar(x + base + width / 2, math_only, width, hatch="//", alpha=0.55,
               label=f"{st.get('label', method.upper())} (math-only calib)", color=st.get("color"))
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    finalize(ax, xlabel="Metric", ylabel="Score",
             subtitle="Solid = default mixed-domain calibration, hatched = math-only; higher is better",
             pct_y=True, grid_axis="y")
    save(fig, cfg, f"ablation_math_calib_generalization_s{int(sparsity*100)}")


def _plot_math_calib(cfg: HPPConfig, df):
    from hpp.plots.style import METHODS, finalize, new_fig, save

    fig, ax = new_fig()
    sparsities = sorted(df["sparsity"].unique())
    x = np.arange(len(sparsities))
    width = 0.35
    for i, method in enumerate(("hpp", "reap")):
        sub = df[df["method"] == method].set_index("sparsity").reindex(sparsities)
        st = METHODS.get(method, {})
        ax.bar(x + (i - 0.5) * width, sub["math_avg"].values, width,
               label=st.get("label", method.upper()), color=st.get("color"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0%}" for s in sparsities])
    finalize(ax, xlabel="Requested sparsity", ylabel="math_avg (mean GSM8K, MATH-500)",
             subtitle="Both methods scored/pruned using GSM8K+MATH-calibrated saliency; higher is better",
             pct_y=True, grid_axis="y")
    save(fig, cfg, "ablation_math_calib")


def _domain_qa_stream(path, cfg_name, split, get, seed_qa, min_chars=20):
    """Like hpp.calibration._domain_stream, but yields (question, answer) pairs instead
    of a single text reconstruction needs the worked solution, not just the prompt."""
    try:
        while True:
            for item in load_dataset(path, cfg_name, split=split, streaming=True):
                q, a = get(item)
                q, a = q.strip(), a.strip()
                if len(q) > min_chars and len(a) > min_chars:
                    yield q, a
    except Exception as e:
        print(f"  domain QA stream failed ({path}): {e}; using seed Q/A")
        while True:
            yield seed_qa


def _chat_example(tok, adapter, question, answer, max_len):
    prompt_ids = list(adapter.prompt_ids(tok, question))
    vocab = tok.get_vocab()
    target_prefix_ids = []
    if "<|channel|>" in vocab and "<|channel|>" not in tok.decode(prompt_ids, skip_special_tokens=False):
        target_prefix_ids = tok.encode("<|channel|>final<|message|>", add_special_tokens=False)
    answer_ids = tok.encode(answer, add_special_tokens=False)
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    full_ids = prompt_ids + target_prefix_ids + answer_ids + eos
    if not answer_ids or len(full_ids) > max_len:
        return None
    ids = torch.tensor(full_ids, dtype=torch.long)
    labels = ids.clone()
    labels[:len(prompt_ids)] = -100          # mask ONLY the actual prompt
    return ids, labels


def build_math_recon_loaders(cfg: HPPConfig, adapter, tok):
    from hpp.evaluation import ppl_prefix

    rc = cfg.recon
    prefix = ppl_prefix(adapter, tok)
    wt = tok("\n\n".join(load_dataset(*cfg.ppl_dataset, split="test")["text"]),
             return_tensors="pt").input_ids
    eval_windows = [wt[:, i * rc.seq_len:(i + 1) * rc.seq_len]
                    for i in range(rc.eval_hold) if (i + 1) * rc.seq_len <= wt.size(1)]

    def example_stream():
        streams = {d: _domain_qa_stream(p, c, s, g, MATH_QA_SEED[d])
                   for d, (p, c, s, g) in MATH_QA_SOURCES.items()}
        domains = list(streams)
        di, seen, kept = 0, 0, 0
        while True:
            dom = domains[di % len(domains)]
            di += 1
            q, a = next(streams[dom])
            seen += 1
            ex = _chat_example(tok, adapter, q, a, rc.seq_len)
            if ex is not None:
                kept += 1
                yield ex
            if seen == 500 and kept == 0:
                raise RuntimeError(
                    "math-recon: 0/500 examples passed the chat-template prefix check. "
                    "This tokenizer's add_generation_prompt=True prefix likely isn't a "
                    "stable prefix of the full 2-turn tokenization (e.g. a special kwarg "
                    "like Qwen3's enable_thinking changes the template) would otherwise "
                    "hang forever finding zero valid training examples.")
            if seen % 500 == 0:
                print(f"  math-recon examples: kept {kept}/{seen} (rest exceeded "
                      f"seq_len={rc.seq_len} or failed the chat-template prefix check)")

    return eval_windows, example_stream(), prefix


def _pad_examples(tok, examples, device):
    pad_id = tok.pad_token_id
    n, maxlen = len(examples), max(ids.numel() for ids, _ in examples)
    input_ids = torch.full((n, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((n, maxlen), -100, dtype=torch.long)
    attn = torch.zeros((n, maxlen), dtype=torch.long)
    for i, (ids, lab) in enumerate(examples):
        L = ids.numel()
        input_ids[i, :L], labels[i, :L], attn[i, :L] = ids, lab, 1
    return input_ids.to(device), attn.to(device), labels.to(device)


def run_math_reconstruction(cfg: HPPConfig, adapter, scores, method, sparsity=None):
    from hpp.reconstruction import (enable_memory_efficient_training, inject_lora,
                                    keep_for, lora_state_dict)
    from hpp.reconstruction import eval_loss as recon_eval_loss

    rc = cfg.recon
    sparsity = rc.sparsities[0] if sparsity is None else sparsity
    tok, model = adapter.load()
    keep, _ = keep_for(cfg, scores, method, sparsity)
    adapter.structural_removal(model, keep)
    trainable, lora_by_key = inject_lora(cfg, adapter, model)
    enable_memory_efficient_training(cfg, model)

    opt = torch.optim.AdamW(trainable, lr=rc.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rc.steps)
    eval_windows, train_iter, prefix = build_math_recon_loaders(cfg, adapter, tok)

    hist = []
    model.train()
    set_seed(cfg.seed)          # identical data order / dropout to the C4 reconstruction
    reset_peak()
    for step in range(1, rc.steps + 1):
        opt.zero_grad()
        accum = 0.0
        for _ in range(rc.grad_accum):
            examples = [next(train_iter) for _ in range(rc.batch)]
            ids, attn, labels = _pad_examples(tok, examples, model.device)
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / rc.grad_accum).backward()
            accum += out.loss.item() / rc.grad_accum
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()

        row = {"step": step, "train_loss": accum}
        if step == 1:
            print(f"  [math-recon {method} s={sparsity:.0%}] peak VRAM after step 1: "
                  f"{peak_vram():.1f} GB")
        if step % rc.eval_every == 0 or step == 1:
            row["eval_loss"] = recon_eval_loss(model, eval_windows, prefix)
            print(f"  [math-recon {method} s={sparsity:.0%}] step {step}/{rc.steps} "
                  f"train={row['train_loss']:.3f} eval={row['eval_loss']:.3f}")
        hist.append(row)

    cfg.recon_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(hist)
    df.to_csv(cfg.recon_dir / f"recon_mathrecon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.csv",
              index=False)
    torch.save(lora_state_dict(lora_by_key),
               cfg.recon_dir / f"lora_mathrecon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.pt")
    del model, tok, opt
    free()
    return df


def evaluate_math_reconstruction(cfg: HPPConfig, adapter, scores, method, sparsity=None,
                                 profile=None, gsm8k_limit=128, math500_limit=128,
                                 include_code=False):
    from hpp.evaluation import evaluate
    from hpp.reconstruction import inject_lora, keep_for, load_lora_state_dict
    from hpp.results import before_row

    rc = cfg.recon
    sparsity = rc.sparsities[0] if sparsity is None else sparsity
    lora_path = cfg.recon_dir / f"lora_mathrecon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.pt"
    if not lora_path.exists():
        print(f"  no saved math-recon LoRA for {method} s={sparsity:.0%} run "
              "run_math_reconstruction first")
        return None

    if profile is None:
        for cand in ("full_sweep", "fast_sweep", "single"):
            try:
                before_row(cfg, cand, method, sparsity)
                profile = cand
                break
            except Exception:
                continue
    if profile is None:
        raise ValueError(f"no 'before' row for {method} s={sparsity:.0%} in any results "
                          "file run the main sweep at this sparsity first")

    eval_profile = _ensure_math_recon_eval_profile(cfg, gsm8k_limit, math500_limit, include_code)
    out_path = cfg.res_dir / "ablation_math_reconstruction.csv"
    rows = pd.read_csv(out_path).to_dict("records") if out_path.exists() else []
    existing = [r for r in rows if r["method"] == method
                and round(float(r["sparsity"]), 4) == round(sparsity, 4)]
    required_cols = cfg.metric_cols(eval_profile)
    if existing and all(c in existing[-1] and pd.notna(existing[-1].get(c)) for c in required_cols):
        print(f"  [skip] math-recon eval {method} s={sparsity:.0%} already logged")
        return pd.DataFrame(rows)
    if existing:
        missing = [c for c in required_cols
                   if c not in existing[-1] or pd.isna(existing[-1].get(c))]
        print(f"  [refresh] math-recon eval {method} s={sparsity:.0%} logged but missing "
              f"{missing} (eval profile grew since that row was written) re-evaluating, "
              f"reusing the already-trained LoRA (no retraining needed)")
        rows = [r for r in rows if not (r["method"] == method
                and round(float(r["sparsity"]), 4) == round(sparsity, 4))]

    b = before_row(cfg, profile, method, sparsity)
    keep, actual = keep_for(cfg, scores, method, sparsity)
    if not np.isclose(actual, float(b["actual_sparsity"]), atol=1e-3):
        print(f"  [WARNING] recomputed actual_sparsity ({actual:.4f}) != logged "
              f"({float(b['actual_sparsity']):.4f}) scores/calibration changed since "
              f"the sweep, so this is not the same model as the 'before' row")
    tok, model = adapter.load()
    adapter.structural_removal(model, keep)
    _, lora_by_key = inject_lora(cfg, adapter, model)
    load_lora_state_dict(lora_by_key, torch.load(lora_path, map_location="cpu"))
    model.eval()
    model.config.use_cache = True
    metrics = evaluate(cfg, adapter, model, tok, eval_profile,
                       f"mathrecon_{method}_s{sparsity:.0%}")
    del model, tok
    free()

    rows.append({"method": method, "sparsity": sparsity, "actual_sparsity": actual, **metrics})
    pd.DataFrame(rows).to_csv(out_path, index=False)
    df = pd.DataFrame(rows)

    a = df[(df["method"] == method) & np.isclose(df["sparsity"].astype(float), sparsity)].iloc[-1]
    cols = [c for c in ("perplexity", "gsm8k", "math500", "code_avg") if c in df.columns]
    parts = [f"{c}: before(unrepaired)={float(b.get(c)):.4f} math-recon={float(a.get(c)):.4f}"
             for c in cols if pd.notna(b.get(c)) and pd.notna(a.get(c))]
    print(f"  {method.upper():5s} s={sparsity:.0%} " + " | ".join(parts))
    return df


def _plot_math_reconstruction(cfg: HPPConfig, df, sparsity):
    from hpp.plots.style import METHODS, finalize, new_fig, save
    from hpp.results import before_row

    cols = [c for c in ("gsm8k", "math500", "code_avg") if c in df.columns]
    if not cols:
        return
    profile = None
    for cand in ("full_sweep", "fast_sweep", "single"):
        try:
            before_row(cfg, cand, "hpp", sparsity)
            profile = cand
            break
        except Exception:
            continue
    if profile is None:
        print("  [plot skipped] no mixed-domain 'before' row found for the reconstruction plot")
        return

    fig, ax = new_fig()
    x = np.arange(len(cols))
    width = 0.18
    for i, method in enumerate(("hpp", "reap")):
        st = METHODS.get(method, {})
        try:
            b = before_row(cfg, profile, method, sparsity)
        except Exception:
            b = None
        a_rows = df[(df["method"] == method) & np.isclose(df["sparsity"].astype(float), sparsity)]
        a = a_rows.iloc[-1] if not a_rows.empty else None
        before_vals = [float(b.get(c)) if b is not None and pd.notna(b.get(c)) else np.nan for c in cols]
        recon_vals = [float(a.get(c)) if a is not None and pd.notna(a.get(c)) else np.nan for c in cols]
        base = (i - 0.5) * 2.2 * width
        ax.bar(x + base - width / 2, before_vals, width, hatch="//", alpha=0.55,
               label=f"{st.get('label', method.upper())} (unrepaired)", color=st.get("color"))
        ax.bar(x + base + width / 2, recon_vals, width,
               label=f"{st.get('label', method.upper())} (math-recon)", color=st.get("color"))
    ax.set_xticks(x)
    ax.set_xticklabels(cols)
    finalize(ax, xlabel="Task", ylabel="Accuracy",
             subtitle="Hatched = pruned/unrepaired, solid = after math-domain LoRA repair; higher is better",
             pct_y=True, grid_axis="y")
    save(fig, cfg, f"ablation_math_reconstruction_s{int(sparsity*100)}")


def _run_math_reconstruction_one(cfg: HPPConfig, adapter, scores, sparsity, include_code=False):
    df = pd.DataFrame()
    for method in ("hpp", "reap"):
        lora_path = cfg.recon_dir / f"lora_mathrecon_{method}_s{int(sparsity*100)}_{cfg.model_tag}.pt"
        if lora_path.exists():
            print(f"  [skip] math-recon training {method} s={sparsity:.0%} already done "
                  f"({lora_path.name} exists)")
        else:
            run_math_reconstruction(cfg, adapter, scores, method, sparsity)
        result = evaluate_math_reconstruction(cfg, adapter, scores, method, sparsity,
                                              include_code=include_code)
        if result is not None:
            df = result          # each call returns the FULL accumulated table so far
    if not df.empty:
        _plot_math_reconstruction(cfg, df, sparsity)
    return df


def run_math_reconstruction_ablation(cfg: HPPConfig, adapter, scores, sparsity=None,
                                     include_code=False):
    rc = cfg.recon
    if sparsity is None:
        sparsities = rc.sparsities
    elif isinstance(sparsity, (int, float)):
        sparsities = (float(sparsity),)
    else:
        sparsities = tuple(float(s) for s in sparsity)

    df = pd.DataFrame()
    for s in sparsities:
        df = _run_math_reconstruction_one(cfg, adapter, scores, s, include_code=include_code)

    if not df.empty:
        df["math_avg"] = df[["gsm8k", "math500"]].mean(axis=1)
    return df
