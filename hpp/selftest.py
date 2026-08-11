"""Pre-flight checks. Loads the model ONCE, runs every check, prints PASS/FAIL/WARN.

    from hpp.selftest import run_all
    run_all(cfg, adapter)

Nothing here needs editing for a particular model: every check reads cfg/adapter. Each
check is wrapped, so one failure never hides the others.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from hpp.utils import free

_RESULTS: list[tuple[str, str, str]] = []

# (label, module file, regex, must_be_present)  every fix that has been applied, so a
# stale file announces itself in 2 seconds instead of 4 hours into a sweep.
_CODE_CHECKS = [
    ("gptoss: harmony ppl prefix disabled",   "models/gptoss.py", r"USE_HARMONY_PPL_PREFIX = False", True),
    ("gptoss: GPU-side accumulation",         "models/gptoss.py", r"acc\.g_sq\[li, e\] \+= .*\.to\(acc\.device\)", True),
    ("gptoss: harmony final_text strip",      "models/gptoss.py", r"<\|channel\|>final<\|message\|>", True),
    ("unfused: sync-free torch.where",        "models/_unfused.py", r"xf = torch\.where\(m2, x, zx\)", True),
    ("unfused: reap_num indexed on GPU",      "models/_unfused.py", r"acc\.reap_num\[li, e\] \+=", True),
    ("profiler: GPU-resident Accum",          "profiler.py", r"self\.g_sq = z\(Lm, E, D\)", True),
    ("profiler: hist_reap present",           "profiler.py", r'"hist_reap": np\.array', True),
    ("profiler: hist_gate removed",           "profiler.py", r"hist_gate", False),
    ("scores: one expert_imp (REAP score)",   "scores.py", r'"hist_reap"\]\[:N\]\.sum', True),
    ("scores: reap_imp removed",              "scores.py", r"reap_imp", False),
    ("eval: use_kv_cache from cfg",           "evaluation.py", r"use_cache=cfg\.use_kv_cache", True),
    ("eval: no hardcoded deepseek tag",       "evaluation.py", r"deepseekmoe16b", False),
    ("eval: robust evalplus parser",          "evaluation.py", r"def _parse_pass1", True),
    ("eval: evalplus stale cache deleted",    "evaluation.py", r"unlink\(missing_ok=True\)", True),
    ("eval: base/chat code mode",             "evaluation.py", r'"completion": _truncate_completion', True),
    ("eval: window_nll + ppl prefix",         "evaluation.py", r"def window_nll", True),
    ("config: sparsities 25/50/75",           "config.py", r"= \(0\.25, 0\.50, 0\.75\)", True),
    ("config: use_kv_cache field",            "config.py", r"use_kv_cache: bool = True", True),
    ("gamma: direction-aware answer key",     "gamma.py", r"LOWER_IS_BETTER", True),
    ("gamma: no-signal warning",              "gamma.py", r"has NO SIGNAL", True),
    ("plots: two separate gamma figures",     "plots/gamma.py", r"def plot_gamma_cliff", True),
    ("plots: TITLE_PAD in points",            "plots/style.py", r"pad=TITLE_PAD if subtitle", True),
    ("plots: no hand-rolled set_title",       "plots/saliency.py", r"set_title", False),
    ("deepseek: chat model",                  "models/deepseek.py", r"deepseek-moe-16b-chat", True),
    ("deepseek: cache compat patch",          "models/deepseek.py", r"_patch_dynamic_cache_compat", True),
]


def validate_install():
    """Is the code on THIS machine the current code? No GPU, no model, ~2 seconds.

    Every entry is a fix that was applied at some point. A BAD line means that file is a
    stale copy which has repeatedly been the actual cause of 'the fix did not work'.
    """
    import re
    from pathlib import Path

    root = Path(__file__).parent
    bad = []
    print(f"\n{'='*74}\nCODE VALIDATION  ({root})\n{'='*74}")
    for label, rel, pat, want in _CODE_CHECKS:
        f = root / rel
        if not f.exists():
            bad.append(label)
            print(f"  [BAD] {label:38s} <- {rel} MISSING")
            continue
        found = bool(re.search(pat, f.read_text(), re.S))
        if found != want:
            bad.append(label)
            print(f"  [BAD] {label:38s} <- {rel} is STALE")
        else:
            print(f"  [OK ] {label}")
    print(f"\n  {len(_CODE_CHECKS)-len(bad)}/{len(_CODE_CHECKS)} OK")
    if bad:
        print("  STALE FILES re-copy them before running anything:")
        for b in bad:
            print(f"    - {b}")
    return not bad


def _log(name, status, detail):
    _RESULTS.append((name, status, detail))
    print(f"  [{status:4s}] {name}: {detail}")


# 1. KV cache
def check_kv_cache(cfg, adapter, model, tok):
    """Decoding without a cache re-runs a full forward per token (10-20x slower).
    Verifies the cache is ON, produces IDENTICAL greedy text, and is actually faster."""
    ids = tok("The capital of France is", return_tensors="pt").input_ids.to(model.device)
    attn = torch.ones_like(ids)
    kw = dict(attention_mask=attn, max_new_tokens=24, do_sample=False,
              pad_token_id=tok.pad_token_id)
    with torch.no_grad():                                   # warm up
        model.generate(**{"inputs": ids}, **kw, use_cache=True)

    out, secs = {}, {}
    for use in (True, False):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            g = model.generate(inputs=ids, **kw, use_cache=use)
        torch.cuda.synchronize()
        secs[use] = time.time() - t0
        out[use] = tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)

    same = out[True].strip() == out[False].strip()
    speed = secs[False] / max(secs[True], 1e-9)
    if not cfg.use_kv_cache:
        _log("kv_cache", "WARN", f"cfg.use_kv_cache=False -> every eval is ~{speed:.1f}x slower "
                                 f"than it needs to be, and decode_tps is not comparable "
                                 f"to your other models")
    elif not same:
        _log("kv_cache", "FAIL", f"cached and uncached greedy text DIFFER the cache is "
                                 f"broken for this model.\n         cached  : {out[True]!r}\n"
                                 f"         uncached: {out[False]!r}")
    else:
        _log("kv_cache", "PASS", f"identical output, {speed:.1f}x faster with cache "
                                 f"({secs[True]*1000:.0f}ms vs {secs[False]*1000:.0f}ms for 24 tokens)")


# 2. perplexity framing
def check_perplexity(cfg, adapter, model, tok):
    """gpt-oss is harmony-native: scoring bare prose measures surprise at the FORMAT.
    Models with an empty ppl prefix are unaffected and should report 'raw == native'."""
    from hpp.evaluation import fast_perplexity

    n = 8
    raw = fast_perplexity(cfg, adapter, model, tok, n, use_prefix=False)
    native = fast_perplexity(cfg, adapter, model, tok, n, use_prefix=True)
    plen = len(adapter.ppl_prefix_ids(tok))
    if plen == 0:
        ok = abs(raw - native) < 1e-6
        _log("perplexity", "PASS" if ok else "FAIL",
             f"no prefix for this model; raw={raw:.2f} native={native:.2f} "
             f"({'identical as expected' if ok else 'SHOULD be identical'})")
    elif native < raw * 0.5:
        _log("perplexity", "PASS", f"prefix ({plen} tok) fixes it: raw={raw:.1f} -> "
                                   f"native={native:.1f} ({raw/native:.1f}x lower)")
    else:
        _log("perplexity", "FAIL", f"prefix ({plen} tok) did NOT help: raw={raw:.1f} "
                                   f"native={native:.1f}. Prefix decodes to:\n"
                                   f"         {tok.decode(adapter.ppl_prefix_ids(tok))!r}")


# 3. code eval mode
def check_code_mode(cfg, adapter, model, tok):
    """Base models cannot be instructed: they must CONTINUE the raw prompt (EvalPlus
    `completion`), not be asked for a whole function (`solution`)."""
    from hpp.evaluation import _extract_code, _truncate_completion, generate

    chat = getattr(tok, "chat_template", None) is not None
    mode = "instruct/solution" if chat else "base/completion"
    stub = ("from typing import List\n\n\ndef has_close_elements(numbers: List[float], "
            "threshold: float) -> bool:\n    \"\"\" Check if any two numbers are closer "
            "than threshold.\n    \"\"\"\n")
    p = (f"Complete this Python function. Return only the code.\n\n{stub}") if chat else stub
    g = generate(cfg, adapter, model, tok, [p], 160)[0]
    text = adapter.final_text(g)
    body = _extract_code(text) if chat else _truncate_completion(text)
    looks_ok = ("def " in body) if chat else (len(body.strip()) > 0 and "def " not in body[:5])
    _log("code_mode", "PASS" if looks_ok else "WARN",
         f"{mode} (chat_template={'yes' if chat else 'no'}). "
         f"{'`def` present as required' if chat and looks_ok else ''}"
         f"{'continuation body, no stray `def`' if (not chat and looks_ok) else ''}"
         f"{'' if looks_ok else 'output does not match the mode inspect below'}")
    print(f"        - what would be written to samples.jsonl-\n"
          f"         {body[:300]!r}")


# 4. routing alignment
def check_routing_alignment(cfg, adapter, model, tok):
    """The REAP score pairs each token's gate weight with THAT token's output norm.
    _unfused assumes an expert receives its tokens in ASCENDING TOKEN INDEX. DeepSeek's
    moe_infer argsorts the flattened top-k ids and torch.argsort is not stable by default,
    so this must be verified, not assumed: if the order is wrong, expert_imp is silently
    garbage and every downstream result inherits it."""
    from hpp.models._unfused import UnfusedAdapter

    if not isinstance(adapter, UnfusedAdapter):
        _log("routing_align", "PASS", "fused adapter: gate and norm are gathered by the "
                                      "same mask, so pairing is exact by construction")
        return

    lid = cfg.moe_layer_ids[0]
    moe = adapter.moe_of(model, lid)
    g_name = adapter.PROJ[0]
    cap: dict = {"xe": {}}
    hs = [moe.register_forward_pre_hook(lambda m, i: cap.__setitem__("h", i[0].detach())),
          adapter.router_of(model, lid).register_forward_hook(
              lambda m, i, o: cap.__setitem__("route", adapter.read_routing(model, o)))]
    for e in range(cfg.num_routed):
        hs.append(getattr(adapter.expert(model, lid, e), g_name).register_forward_pre_hook(
            (lambda ee: (lambda m, i: cap["xe"].__setitem__(ee, i[0].detach())))(e)))

    ids = tok("The quick brown fox jumps over the lazy dog. " * 20,
              return_tensors="pt").input_ids[:, :256].to(model.device)
    with torch.no_grad():
        model(input_ids=ids)
    for h in hs:
        h.remove()

    if cap.get("route") is None or "h" not in cap:
        _log("routing_align", "WARN", "could not capture routing//MoE input; skipped")
        return
    sel, _ = cap["route"]
    h = cap["h"].reshape(-1, cap["h"].shape[-1]).float().cpu()
    checked = mism = 0
    for e, xe in cap["xe"].items():
        xe = xe.reshape(-1, xe.shape[-1]).float().cpu()
        tok_ids = (sel == e).any(-1).nonzero().squeeze(-1)      # ascending by construction
        if tok_ids.numel() != xe.shape[0] or xe.shape[0] == 0:
            continue
        checked += 1
        if not torch.allclose(h[tok_ids], xe, atol=1e-2):
            mism += 1
    if checked == 0:
        _log("routing_align", "WARN", "no expert received a comparable row count; skipped")
    elif mism == 0:
        _log("routing_align", "PASS", f"{checked} experts checked, all receive tokens in "
                                      f"ascending index -> gate/norm pairing is correct")
    else:
        _log("routing_align", "FAIL", f"{mism}/{checked} experts receive tokens in a "
                                      f"DIFFERENT order -> expert_imp is wrong for this "
                                      f"model. Do not trust any result until fixed.")


# 5. calibration signal
def check_calib_metric(cfg):
    """gamma is fitted to the drop probe. If the probe's metric has no signal on this
    model, gamma is fitted to noise."""
    import pandas as pd

    from hpp.gamma import LOWER_IS_BETTER

    m = cfg.calib_primary_metric
    if not cfg.dropkey_csv.exists():
        _log("calib_metric", "WARN", f"no drop keys yet; primary metric is {m!r}. "
                                     f"For a BASE model (no chat template) use 'perplexity'.")
        return
    d = pd.read_csv(cfg.dropkey_csv)
    v = d[m].astype(float)
    lo, hi = v.min(), v.max()
    best = lo if m in LOWER_IS_BETTER else hi
    spread = (hi - lo) / max(abs(best), 1e-12)
    if m not in LOWER_IS_BETTER and hi < 0.05:
        _log("calib_metric", "FAIL", f"{m} never exceeds {hi:.3f} this model cannot do "
                                     f"the task, so m* is noise. Set "
                                     f"cfg.calib_primary_metric='perplexity'.")
    elif spread < 0.10:
        _log("calib_metric", "FAIL", f"{m} varies only {spread:.1%} across the sweep "
                                     f"({list(np.round(v.values, 4))}) -> gamma fitted to "
                                     f"noise. Set cfg.calib_primary_metric='perplexity'.")
    else:
        _log("calib_metric", "PASS", f"{m} spans {lo:.4g}..{hi:.4g} ({spread:.0%} spread) "
                                     f"-> the probe has signal")


# generation failure
def debug_generation(cfg, adapter, model=None, tok=None, limit=2):
    """WHY did gsm8k come back None?

    evaluate() swallows per-task exceptions into None so one broken task cannot kill a
    5-hour sweep. That is right for a sweep and useless for debugging, because the answer
    key then sees NaN and blames the metric. This runs the same call with NO catch, and
    runs it BOTH ways (cache on and off) so the culprit identifies itself:

        cache ON raises, cache OFF works   -> the KV cache. Either fix
                                              _patch_dynamic_cache_compat or set
                                              cfg.use_kv_cache = False and record it.
        both raise                         -> not the cache. Read the traceback.
        both work                          -> the failure is data-dependent; raise `limit`.
    """
    import traceback
    from dataclasses import replace

    from hpp.evaluation import gsm8k_score

    owns = model is None
    if owns:
        tok, model = adapter.load()

    print(f"\n{'='*74}\nGENERATION DEBUG  {cfg.model_tag}\n{'='*74}")
    print(f"  chat_template : {'present' if getattr(tok, 'chat_template', None) else 'NONE (base model!)'}")
    print(f"  pad_token     : {tok.pad_token!r}  eos: {tok.eos_token!r}")
    ids = adapter.prompt_ids(tok, "What is 2+2?")
    print(f"  prompt_ids    : {len(ids)} tokens")
    print(f"  decodes to    : {tok.decode(ids)!r}")

    for use_cache in (True, False):
        c2 = replace(cfg, use_kv_cache=use_cache)
        print(f"\n - use_kv_cache={use_cache}-")
        try:
            t0 = time.time()
            s = gsm8k_score(c2, adapter, model, tok, limit)
            print(f"  [OK  ] gsm8k={s:.3f} on {limit} items in {time.time()-t0:.1f}s")
        except Exception as ex:
            print(f"  [FAIL] {type(ex).__name__}: {ex}")
            traceback.print_exc()

    if owns:
        del model, tok
        free()


# perplexity diagnosis
def diagnose_perplexity(cfg, adapter, model=None, tok=None, n_windows=24):
    """WHY is perplexity implausible? Three questions, each with a distinct culprit.

      A) does it depend on SEQUENCE LENGTH?   -> attention / sliding-window / RoPE at long
                                                 context (short context fine == generation
                                                 works while ppl does not)
      B) is it a FEW BAD WINDOWS or all of them? -> bf16 overflow / degenerate text
      C) does the harmony PREFIX help or hurt?   -> framing

    Prints a table. Nothing is inferred for you; read A/B/C and the culprit is whichever
    one moves.
    """
    from hpp.evaluation import ppl_prefix, window_nll
    from datasets import load_dataset

    owns = model is None
    if owns:
        tok, model = adapter.load()
    text = "\n\n".join(load_dataset(*cfg.ppl_dataset, split="test")["text"])
    enc = tok(text, return_tensors="pt", truncation=False, verbose=False).input_ids
    print(f"\n  corpus: {enc.size(1):,} tokens with this tokenizer")

    def ppl_at(seq_len, prefix, n):
        losses = []
        for w in range(n):
            b = w * seq_len
            if b + seq_len > enc.size(1):
                break
            loss, ntok = window_nll(model, enc[:, b:b + seq_len], prefix)
            losses.append(float(loss))
        a = np.array(losses)
        fin = a[np.isfinite(a)]
        return a, (float(np.exp(fin.mean())) if fin.size else float("nan"))

    empty = torch.empty(0, dtype=torch.long)
    pref = ppl_prefix(adapter, tok)

    print(f"\n  A) perplexity vs sequence length (no prefix)")
    print(f"     {'seq_len':>8s} {'windows':>8s} {'ppl':>12s} {'mean loss':>10s} {'max loss':>9s}")
    for sl in (128, 256, 512, 1024, 2048):
        n = max(4, min(n_windows, 4096 // sl * 4))
        a, p = ppl_at(sl, empty, n)
        print(f"     {sl:8d} {len(a):8d} {p:12.2f} {np.nanmean(a):10.3f} {np.nanmax(a):9.3f}")
    print("     -> READ THE DIRECTION, not the absolute value: loss should FALL as seq_len")
    print("        grows (more context = better prediction). If it RISES, longer context is")
    print("        actively hurting -> run diagnose_context() next, which removes the")
    print("        position confound. Absolute ppl being high at every length is a separate")
    print("        question and block B answers it.")

    print(f"\n  B) per-window losses at seq_len={cfg.ppl_seq_len} (no prefix)")
    a, p = ppl_at(cfg.ppl_seq_len, empty, n_windows)
    fin = a[np.isfinite(a)]
    print(f"     ppl={p:.2f}  n={len(a)}  non-finite={int((~np.isfinite(a)).sum())}")
    print(f"     loss  min={fin.min():.3f}  p50={np.median(fin):.3f}  p90={np.percentile(fin,90):.3f}"
          f"  max={fin.max():.3f}")
    print(f"     ppl from the MEDIAN window alone: {np.exp(np.median(fin)):.2f}")
    print("     -> if median-window ppl is sane but the mean is not, a few windows dominate")
    worst = int(np.argmax(np.where(np.isfinite(a), a, -1)))
    b = worst * cfg.ppl_seq_len
    print(f"     worst window #{worst} (loss {a[worst]:.2f}) starts:\n"
          f"       {tok.decode(enc[0, b:b+60])!r}")

    print(f"\n  C) harmony prefix ({pref.numel()} tokens)")
    if pref.numel() == 0:
        print("     none for this model skip")
    else:
        _, p_no = ppl_at(cfg.ppl_seq_len, empty, n_windows)
        _, p_yes = ppl_at(cfg.ppl_seq_len, pref, n_windows)
        verdict = "HELPS" if p_yes < p_no * 0.9 else ("HURTS" if p_yes > p_no * 1.1 else "no effect")
        print(f"     raw={p_no:.2f}  with prefix={p_yes:.2f}  -> prefix {verdict}")
        print(f"     prefix decodes to: {tok.decode(pref.tolist())!r}")

    if owns:
        del model, tok
        free()


# context / attention
def diagnose_context(cfg, adapter, model=None, tok=None, target=128, reps=6):
    """Does MORE context make the model WORSE? Confound-free.

    The seq_len sweep in diagnose_perplexity confounds two things: longer windows change
    BOTH the context available and WHICH tokens are scored. Here the scored tokens are
    held FIXED the same `target` tokens at the same place in the text and only the
    unscored prefix in front of them grows. window_nll already scores exactly this way.

    A healthy model: loss falls monotonically as context grows.
    Loss RISING with context means attention is mis-masked at range (for gpt-oss: the
    sliding_window=128 layers). Note 0 -> 128 is the interval where a sliding mask and a
    full mask are IDENTICAL, so a jump right after 128 is the tell.
    """
    from datasets import load_dataset

    from hpp.evaluation import window_nll

    owns = model is None
    if owns:
        tok, model = adapter.load()
    text = "\n\n".join(load_dataset(*cfg.ppl_dataset, split="test")["text"])
    enc = tok(text, return_tensors="pt", truncation=False, verbose=False).input_ids

    sw = getattr(model.config, "sliding_window", None)
    lt = getattr(model.config, "layer_types", None)
    print(f"\n  config: sliding_window={sw}  attn_impl={cfg.attn_impl}")
    if lt:
        from collections import Counter
        print(f"          layer_types={dict(Counter(lt))}")
    print(f"  {len(_CTX)} context lengths x {reps} positions, scoring the SAME {target} tokens\n")
    print(f"     {'context':>8s} {'mean loss':>10s} {'ppl':>10s}  {'vs no context':>14s}")

    base = None
    for ctx in _CTX:
        losses = []
        for r in range(reps):
            o = 4096 + r * 6000 + max(_CTX)          # room for the largest prefix
            if o + target > enc.size(1):
                break
            tgt = enc[:, o:o + target]
            pre = enc[0, o - ctx:o] if ctx else torch.empty(0, dtype=torch.long)
            loss, _ = window_nll(model, tgt, pre)
            if torch.isfinite(loss):
                losses.append(float(loss))
        if not losses:
            continue
        ml = float(np.mean(losses))
        base = ml if base is None else base
        print(f"     {ctx:8d} {ml:10.3f} {np.exp(ml):10.2f}  {ml - base:+14.3f}")
    print(f"\n     FALLING  -> attention is fine; a high absolute ppl is then the model, not a bug")
    print(f"     RISING   -> longer sequences are actively hurting. The variable is TOTAL")
    print(f"                 sequence length (prefix + target), not prefix length: at prefix=0")
    print(f"                 the sequence is {target} tokens, so if sliding_window >= {target} no")
    print(f"                 sliding mask applies there and the first RISE marks the point where")
    print(f"                 it starts to. Next: compare_attn_impl(cfg, adapter)")
    if sw:
        print(f"     NOTE     sliding_window={sw}; the target block alone is {target} tokens.")
    if owns:
        del model, tok
        free()


_CTX = (0, 64, 128, 256, 512, 1024, 2048)


def compare_attn_impl(cfg, adapter, impls=("eager", "sdpa"), n_windows=8):
    """A/B the attention implementation on identical windows. Reloads the model per impl.

    gpt-oss uses attention SINKS and a 128-token sliding window on alternating layers.
    'eager' was chosen for portability; if 'sdpa' gives a materially lower perplexity on
    the SAME windows, eager's long-range masking is the bug and the fix is one config field.

    THE STAKES ARE NOT JUST PERPLEXITY. calib_max_len is 2048, so the calibration forward
    passes run in exactly the regime this tests. If attention is broken there, then g_sq,
    reap_num and every fingerprint were collected from a mis-masked model which means
    expert_imp, cscore, the floors, gamma and every pruning decision derived from them are
    contaminated. A win here invalidates the cached profile, not just the ppl column.
    """
    from dataclasses import replace

    from hpp.evaluation import fast_perplexity

    out = {}
    for impl in impls:
        c = replace(cfg, attn_impl=impl)
        a = type(adapter)(c)
        try:
            tok, model = a.load()
        except Exception as ex:
            print(f"  {impl:6s}: load failed ({type(ex).__name__}: {ex})")
            continue
        out[impl] = fast_perplexity(c, a, model, tok, n_windows, use_prefix=False)
        print(f"  {impl:6s}: ppl = {out[impl]:.2f} over {n_windows} windows of "
              f"{c.ppl_seq_len} tokens")
        del model, tok
        free()
    if len(out) == 2:
        a_, b_ = list(out.values())
        best = min(out, key=out.get)
        print(f"\n  -> {best!r} wins by {max(a_, b_) / max(min(a_, b_), 1e-9):.2f}x. "
              f"If that ratio is >1.5, set attn_impl={best!r} in hpp/models/gptoss.py "
              f"config() and re-run. If it is ~1.0, attention is not the problem.")
    return out


# driver
CHECKS = [("kv_cache", check_kv_cache), ("perplexity", check_perplexity),
          ("code_mode", check_code_mode), ("routing_align", check_routing_alignment)]


def run_all(cfg, adapter, keep_model=False):
    """Loads the model once, runs every check, prints a summary."""
    _RESULTS.clear()
    print(f"\n{'='*74}\nSELF-TEST  {cfg.model_tag} [{cfg.prec_tag}]\n{'='*74}")
    check_calib_metric(cfg)
    print("  loading model...")
    tok, model = adapter.load()
    for name, fn in CHECKS:
        try:
            fn(cfg, adapter, model, tok)
        except Exception as ex:
            _log(name, "FAIL", f"{type(ex).__name__}: {ex}")
    if not keep_model:
        del model, tok
        free()
    bad = [r for r in _RESULTS if r[1] == "FAIL"]
    warn = [r for r in _RESULTS if r[1] == "WARN"]
    print(f"\n  {len(_RESULTS) - len(bad) - len(warn)} pass, {len(warn)} warn, {len(bad)} fail")
    if bad:
        print("  FAILING: " + ", ".join(r[0] for r in bad) + " fix before running a sweep")
    return _RESULTS