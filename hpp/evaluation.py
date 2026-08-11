"""Evaluation: perplexity + MC (log-likelihood) + Math/Code (generative).

Four things that are easy to get wrong and are handled explicitly here:

1. Perplexity does NOT pass `labels=` to the model. Several MoE implementations attach an
   auxiliary router loss when labels are present, which contaminates the LM loss. Logits
   are shifted and cross-entropy is computed here instead.
2. Perplexity is scored in the MODEL'S NATIVE FORMAT via adapter.ppl_prefix_ids(). Empty
   for DeepSeek / Qwen3 / Mixtral -> an exact no-op, their numbers are unchanged. gpt-oss
   is harmony-native and post-trained out of raw-text continuation: bare prose reported
   ~272 ppl while GSM8K was 0.89, i.e. the number measured surprise at the FORMAT, not at
   WikiText. The prefix is context only and is never scored, so the perplexity is still
   over the eval corpus and nothing else.
3. MMLU gets its own limit. It is a GROUP of 57 subject subtasks and lm-eval applies
   `limit` PER SUBTASK, so an uncapped mmlu is ~57x any other MC task.
4. Every generative task is decoded WITHOUT skip_special_tokens and then passed through
   adapter.final_text(), so gpt-oss's harmony `<|channel|>final<|message|>` wrapper is
   stripped before answer/code extraction. Skipping that yields near-zero scores that
   look like pruning damage but are a parsing bug.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch
import transformers
from datasets import load_dataset

from hpp.config import CATEGORIES, EVALPLUS_DATASET, HPPConfig
from hpp.utils import GPUMemorySampler, peak_vram, reset_peak

# Quiet library-internal chatter that isn't actionable from here and drowns out this
# project's own prints: transformers' `from_legacy_cache`/fast-tokenizer advisories, and
# lm-eval-harness's per-task config notes (num_fewshot overrides  expected, this project
# always requests zero-shot; missing aggregation/higher_is_better on boolq/rte  lm-eval's
# own defaults, nothing this project configures). A real failure in either library still
# raises normally; this only silences their INFO/WARNING-level narration.
transformers.logging.set_verbosity_error()
logging.getLogger("lm_eval").setLevel(logging.ERROR)


# perplexity
def ppl_prefix(adapter, tok, use_prefix=True):
    """The (unscored) context every window is conditioned on. Empty -> raw-text scoring."""
    ids = adapter.ppl_prefix_ids(tok) if use_prefix else []
    return torch.tensor(list(ids), dtype=torch.long)


@torch.no_grad()
def window_nll(model, window, prefix):
    """Summed NLL and token count for ONE window, conditioned on `prefix`.

    The prefix is context only  never scored  so ntok counts the window's own tokens
    and the perplexity is over the eval corpus alone.

    P = 0 reduces to exactly the previous code (labels = ids[1:], L-1 tokens), which is
    why the models that were already fine come out bit-identical.
    """
    P = prefix.numel()
    full = torch.cat([prefix, window.reshape(-1)]).unsqueeze(0).to(model.device)
    out = model(full, attention_mask=torch.ones_like(full))
    start = max(P - 1, 0)                    # first position whose LABEL is a window token
    logits = out.logits[:, start:-1, :]
    labels = full[:, start + 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    return loss.float(), labels.numel()


def fast_perplexity(cfg: HPPConfig, adapter, model, tok, n_windows, use_prefix=True):
    text = "\n\n".join(load_dataset(*cfg.ppl_dataset, split="test")["text"])
    # We chunk manually below; suppress the tokenizer's max-length advisory.
    enc = tok(text, return_tensors="pt", truncation=False, verbose=False).input_ids
    seq_len = cfg.ppl_seq_len
    prefix = ppl_prefix(adapter, tok, use_prefix)
    n_windows = n_windows if n_windows is not None else max(1, enc.size(1) // seq_len)
    nlls, ntok, per_window = [], 0, []
    for w in range(n_windows):
        b = w * seq_len
        if b >= enc.size(1):
            break
        ids = enc[:, b:b + seq_len]
        if ids.size(1) < 2:
            continue
        loss, n = window_nll(model, ids, prefix)
        if torch.isfinite(loss):
            nlls.append(loss * n)
            ntok += n
            per_window.append(float(loss))
        else:
            print(f"  [WARN] non-finite loss at window {w}")
    if not nlls:
        return float("nan")
    ppl = float(torch.exp(torch.stack(nlls).sum() / ntok))

    # Perplexity is exp(MEAN loss), so a handful of catastrophic windows can dominate a
    # corpus the model otherwise handles fine. Comparing against the MEDIAN window says
    # whether this number describes the corpus or describes a few outliers  worth
    # knowing BEFORE it becomes the y axis of a figure and the held-out signal for the
    # LoRA reconstruction.
    a = np.array(per_window)
    med = float(np.exp(np.median(a)))
    if len(a) >= 8 and ppl > 3 * med:
        print(f"  [WARNING] perplexity {ppl:.1f} is outlier-dominated: the MEDIAN window "
              f"gives {med:.1f} ({ppl/med:.1f}x lower).\n"
              f"            loss min={a.min():.2f} p50={np.median(a):.2f} "
              f"p90={np.percentile(a, 90):.2f} max={a.max():.2f} over {len(a)} windows.\n"
              f"            Run hpp.selftest.diagnose_perplexity(cfg, adapter) before "
              f"trusting this number.")
    return ppl


# MC via lm-eval
def _pick(d):
    for k in ("acc_norm,none", "acc,none", "exact_match,flexible-extract",
              "exact_match,strict-match", "exact_match,none", "pass@1,none"):
        if k in d:
            return float(d[k])
    return next((float(v) for v in d.values() if isinstance(v, (int, float))), None)


def mc_scores(cfg: HPPConfig, model, tok, prof):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=cfg.eval_batch_size)
    groups: dict[object, list[str]] = {}
    for t in prof.mc_tasks:                      # mmlu lands in its own group by construction
        groups.setdefault(prof.limit_for(t), []).append(t)

    out = {}
    for lim, tasks in groups.items():
        r = simple_evaluate(model=lm, tasks=list(tasks), limit=lim,
                            num_fewshot=cfg.num_fewshot_mc, bootstrap_iters=0)["results"]
        for t in tasks:
            if t in r:
                out[t] = _pick(r[t])
            else:                                # group task -> mean over its subtasks
                subs = [_pick(v) for k, v in r.items()
                        if k.startswith(t) and _pick(v) is not None]
                out[t] = float(np.mean(subs)) if subs else None
    return out


# generation
def generate(cfg: HPPConfig, adapter, model, tok, user_texts, max_new):
    """-> list of RAW decoded strings (special tokens kept, so the harmony wrapper
    survives for adapter.final_text to strip).

    Batched generate() pads sequences that finish early up to the longest one in the
    batch using pad_token_id  which every adapter sets equal to eos_token_id. With
    special tokens kept, that padding decodes as literal repeated EOS text (DeepSeek:
    dozens of copies of '<|end of sentence|>'). Harmless for tasks that regex-search the
    text (gsm8k/math500), but code_score treats the whole string as literal Python source
    when there is no fence, so this garbage makes every sample a syntax error  pass@1=0
    that looks like the model can't code. Strip the trailing run here, once, for every
    model, rather than patching each downstream consumer.
    """
    prompts = [adapter.prompt_ids(tok, t) for t in user_texts]
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs = []
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), cfg.eval_batch_size):
                chunk = prompts[i:i + cfg.eval_batch_size]
                feats = tok.pad({"input_ids": chunk}, return_tensors="pt",
                                padding=True, return_attention_mask=True).to(model.device)
                if "attention_mask" not in feats:
                    feats["attention_mask"] = (feats["input_ids"] != tok.pad_token_id).long()
                gen = model.generate(**feats, max_new_tokens=max_new, do_sample=False,
                                     use_cache=cfg.use_kv_cache,
                                     pad_token_id=tok.pad_token_id)
                inlen = feats["input_ids"].shape[1]
                for j in range(len(chunk)):
                    text = tok.decode(gen[j, inlen:], skip_special_tokens=False)
                    if tok.eos_token:
                        while text.endswith(tok.eos_token):
                            text = text[:-len(tok.eos_token)]
                    outs.append(text)
    finally:
        tok.padding_side = old_side
    return outs


# math
_NUM = re.compile(r"-?\d*\.?\d+")


def _extract_num(text):
    if text is None:
        return None
    nums = _NUM.findall(text.replace(",", ""))
    try:
        return float(nums[-1]) if nums else None
    except Exception:
        return None


def gsm8k_score(cfg, adapter, model, tok, limit):
    ds = load_dataset("gsm8k", "main", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def gold(ans):
        m = re.search(r"####\s*(-?[\d,]+)", ans)
        return _extract_num(m.group(1)) if m else None

    golds = [gold(a) for a in ds["answer"]]
    gens = generate(cfg, adapter, model, tok, list(ds["question"]), cfg.gen_max_new)
    correct = 0
    for g, gt in zip(gens, golds):
        p = _extract_num(adapter.final_text(g))
        if p is not None and gt is not None and abs(p - gt) < 1e-4:
            correct += 1
    return correct / max(len(gens), 1)


def _extract_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def _norm_math(a):
    if a is None:
        return None
    a = a.strip().replace(" ", "").replace("\\left", "").replace("\\right", "")
    a = a.replace("\\!", "").replace("\\,", "").replace("$", "").rstrip(".")
    return a.replace("dfrac", "frac").replace("tfrac", "frac")


def math500_score(cfg, adapter, model, tok, limit):
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    prompts = [p + "\n\nPut the final answer in \\boxed{}." for p in ds["problem"]]
    gens = generate(cfg, adapter, model, tok, prompts, cfg.gen_max_new)
    correct = 0
    for g, gold in zip(gens, ds["answer"]):
        text = adapter.final_text(g)
        p = _norm_math(_extract_boxed(text))
        if p is None:
            p = _norm_math(str(_extract_num(text)))
        gt = _norm_math(str(gold))
        if p is None or gt is None:
            continue
        if p == gt:
            correct += 1
            continue
        try:
            if abs(float(p) - float(gt)) < 1e-4:
                correct += 1
        except Exception:
            pass
    return correct / max(len(gens), 1)


# code (EvalPlus CLI)
def _require_evalplus():
    if importlib.util.find_spec("evalplus") is None:
        raise RuntimeError(
            "evalplus is not installed, so code_avg cannot be produced.\n"
            "  pip install evalplus break-system-packages\n"
            "This is raised rather than returned as None on purpose: a silent None would "
            "only surface after the whole sweep had finished.")


def _extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    code = m.group(1) if m else text
    # Un-fenced chat completions routinely open with one stray leading space/newline (a
    # tokenization artifact of where the assistant turn begins). Invisible as prose, fatal
    # as top-level Python  IndentationError: unexpected indent  since EvalPlus executes
    # `solution` as a complete, column-0 script.
    return code.lstrip(" \n\t")


_STOP = ("\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n#", "\n@")


def _truncate_completion(text):
    """Base-model completion: keep the function body and stop at the first line that
    dedents back to column 0, otherwise the next hallucinated function is executed too."""
    cut = len(text)
    for s in _STOP:
        i = text.find(s)
        if 0 <= i < cut:
            cut = i
    return text[:cut]


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# The "+" line is the one we want; "(base tests)" is the loose vanilla grade.
_PLUS = re.compile(r"\+\s*\(base \+ extra tests\)[^\d]*?pass@1:\s*([0-9.]+)", re.S)
_ANY = re.compile(r"pass@1:\s*([0-9.]+)")


def _parse_pass1(stdout, stderr):
    """EvalPlus's console output is not a stable API: it renders through `rich` (so a
    colour escape can land exactly where a digit is expected) and has moved between
    streams across releases. Read BOTH streams, strip ANSI, prefer the explicitly
    labelled '+' line, and fall back to the last pass@1 seen (base prints first, plus
    second)."""
    text = _ANSI.sub("", (stdout or "") + "\n" + (stderr or ""))
    m = _PLUS.search(text)
    if m:
        return float(m.group(1))
    vals = _ANY.findall(text)
    return float(vals[-1]) if vals else None


def code_score(cfg, adapter, model, tok, task, limit, tag):
    """Generate greedily, dump samples.jsonl, and let EvalPlus's OWN CLI grade it.
    The CLI applies EvalPlus's sanitisation and time limits  those defaults ARE the
    protocol every published Eval+ number was produced under.

    WARNING: this executes untrusted model-generated code. Disposable machine only.
    """
    _require_evalplus()
    from evalplus.data import get_human_eval_plus, get_mbpp_plus

    ds_name = EVALPLUS_DATASET[task]
    raw = get_human_eval_plus() if ds_name == "humaneval" else get_mbpp_plus()
    items = list(raw.items())
    # EvalPlus expects a complete samples.jsonl for the chosen dataset; partial code
    # subsets are rejected as missing problems. We therefore ignore `limit` here.
    if limit:
        print(f"  [evalplus] ignoring partial limit={limit} for {task}; using full {ds_name} set")

    # A BASE model (no chat template) cannot be instructed. "Return only the code" is not an
    # instruction to it  it is just text to continue, and it emits a bare function BODY
    # with no ```fence and no `def` line. Writing that as EvalPlus's `solution` (which must
    # be COMPLETE, runnable code) executes a body with no signature -> pass@1 ~ 0, which
    # looks like the model cannot code rather than like a prompting mismatch.
    #
    # So: chat models are ASKED and give a whole function (`solution`); base models are given
    # the raw prompt to CONTINUE and their output is appended to it by EvalPlus (`completion`).
    chat = getattr(tok, "chat_template", None) is not None
    if chat:
        prompts = [f"Complete this Python function. Return only the code.\n\n{v['prompt']}"
                   for _, v in items]
    else:
        prompts = [v["prompt"] for _, v in items]
    print(f"  [evalplus] {ds_name}: {'instruct/solution' if chat else 'base/completion'} mode, "
          f"{len(items)} problems")
    gens = generate(cfg, adapter, model, tok, prompts, cfg.code_max_new)

    out_dir = cfg.code_samples_dir / tag / ds_name
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = out_dir / "samples.jsonl"
    with open(samples, "w") as f:
        for (task_id, _), g in zip(items, gens):
            text = adapter.final_text(g)
            if chat:
                rec = {"task_id": task_id, "solution": _extract_code(text)}
            else:
                rec = {"task_id": task_id, "completion": _truncate_completion(text)}
            f.write(json.dumps(rec) + "\n")

    # EvalPlus caches its grades beside the samples and silently reuses them
    # ("Load from previous results from ..."). Re-running a tag with different weights
    # would then report the PREVIOUS model's pass@1. Delete the cache so it always grades.
    (out_dir / "samples_eval_results.json").unlink(missing_ok=True)

    r = subprocess.run([sys.executable, "-m", "evalplus.evaluate",
                        "--dataset", ds_name, "--samples", str(samples)],
                       capture_output=True, text=True)
    v = _parse_pass1(r.stdout, r.stderr)
    if v is None:
        print(f"  [evalplus] could not parse pass@1 for {task} (rc={r.returncode}).\n"
              f"  - stdout tail -\n{(r.stdout or '')[-1200:]}\n"
              f"  - stderr tail -\n{(r.stderr or '')[-600:]}")
        return None
    print(f"  [evalplus] {ds_name}+ pass@1 = {v:.4f}  ({len(items)} problems)")
    return v


# latency
def measure_latency(cfg, model, tok, prompt="The meaning of life is", new_tok=128, runs=3):
    """decode_tps is a REPORTED METRIC. Measuring it with the cache disabled does not make
    a model look slow  it makes the number mean something else entirely, and it would be
    incomparable with every other model in the study."""
    ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
    attn = torch.ones_like(ids)
    use_kv_cache = cfg.use_kv_cache
    with torch.no_grad():
        model.generate(ids, attention_mask=attn, max_new_tokens=8, do_sample=False,
                       use_cache=use_kv_cache, pad_token_id=tok.pad_token_id)

    ttft, dtps = [], []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.generate(ids, attention_mask=attn, max_new_tokens=1, do_sample=False,
                           use_cache=use_kv_cache, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t1 = time.time()
        ttft.append((t1 - t0) * 1000)

        torch.cuda.synchronize()
        t2 = time.time()
        with torch.no_grad():
            model.generate(ids, attention_mask=attn, max_new_tokens=new_tok, do_sample=False,
                           use_cache=use_kv_cache, pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize()
        t3 = time.time()
        dtps.append(new_tok / max(t3 - t2, 1e-9))

    return statistics.median(ttft), statistics.median(dtps)


# driver
GEN_SCORERS = {"gsm8k": gsm8k_score, "math500": math500_score}


def evaluate(cfg: HPPConfig, adapter, model, tok, profile="single", tag=""):
    p = cfg.profile(profile)
    reset_peak()
    res: dict = {}
    with GPUMemorySampler() as sampler:
        res["perplexity"] = round(fast_perplexity(cfg, adapter, model, tok, p.ppl_windows), 4)

        if p.mc_tasks:
            try:
                mc = mc_scores(cfg, model, tok, p)
                res.update({k: round(v, 4) for k, v in mc.items() if v is not None})
            except Exception as e:
                print(f"  MC suite failed: {e}")

        for task, limit in p.gen_tasks.items():
            try:
                if task in GEN_SCORERS:
                    v = GEN_SCORERS[task](cfg, adapter, model, tok, limit)
                else:
                    v = code_score(cfg, adapter, model, tok, task, limit, tag or "eval")
                res[task] = round(v, 4) if v is not None else None
            except Exception as e:
                print(f"  {task} failed: {e}")
                res[task] = None

        for c in p.categories:
            vals = [res.get(t) for t in p.tasks_in(c)]
            vals = [v for v in vals if v is not None]
            res[f"{c}_avg"] = round(float(np.mean(vals)), 4) if vals else None

        ttft, dtps = measure_latency(cfg, model, tok)

    res.update({"peak_vram_gb": round(peak_vram(), 2), "avg_vram_gb": sampler.avg_gb,
                "ttft_ms": round(ttft, 1), "decode_tps": round(dtps, 1)})
    print(f"  [{tag}|{cfg.prec_tag}] ppl={res['perplexity']} "
          + " ".join(f"{c}={res.get(f'{c}_avg')}" for c in p.categories)
          + f" peak={res['peak_vram_gb']}GB avg={res['avg_vram_gb']}GB tps={res['decode_tps']}")
    return res