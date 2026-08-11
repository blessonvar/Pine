"""Packed, REAP-style calibration one builder for every model.

REAP calibrates with ~1024 samples packed to 2048 tokens. Two properties matter and
both were broken in the per-domain builders this replaces:

  * Documents are NOT pre-truncated. A document longer than the window continues into
    the next one instead of having its tail dropped.
  * Domains are interleaved round-robin, so every packed window is domain-mixed rather
    than each window belonging to one domain. That mirrors REAP's combined-calibration
    setting; per-domain blocks would let a domain's routing behaviour dominate whole
    windows and skew the expert statistics.
"""
from __future__ import annotations

import random

import torch
from datasets import load_dataset

from hpp.config import HPPConfig

SOURCES = {
    "conversation": ("Open-Orca/OpenOrca", None, "train", lambda x: x.get("question", "")),
    "reasoning":    ("allenai/ai2_arc", "ARC-Challenge", "train", lambda x: x.get("question", "")),
    "math":         ("gsm8k", "main", "train", lambda x: x.get("question", "")),
    "code":         ("mbpp", None, "train", lambda x: "Write a Python function: " + x.get("text", "")),
    "translation":  ("Helsinki-NLP/opus-100", "en-fr", "test",
                     lambda x: "Translate to English: " + x.get("translation", {}).get("fr", "")),
}

SEED_TEXT = {
    "conversation": "Explain how a transformer works.",
    "reasoning": "Why does ice float on water?",
    "math": "What is 23 times 47?",
    "code": "Write a Python function to reverse a string.",
    "translation": "Translate to English: Bonjour le monde.",
}


def _domain_stream(path, cfg_name, split, get, seed_text, min_chars=20):
    """Infinite generator over one domain's texts; falls back to a fixed seed string on
    load failure so packing always has something to consume."""
    try:
        while True:
            for item in load_dataset(path, cfg_name, split=split, streaming=True):
                t = get(item).strip()
                if len(t) > min_chars:
                    yield t
    except Exception as e:
        print(f"  domain stream failed ({path}): {e}; using seed text")
        while True:
            yield seed_text


def build_calibration(cfg: HPPConfig, tok, adapter):
    seq_len = cfg.calib_max_len
    total = cfg.calib_total_windows
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    sep = torch.tensor([eos_id], dtype=torch.long)

    streams = {d: _domain_stream(p, c, s, g, SEED_TEXT[d]) for d, (p, c, s, g) in SOURCES.items()}
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
    print(f"  calibration: {len(windows)} packed windows x {seq_len} tokens "
          f"from {n_docs} source docs across {len(domains)} domains")
    return windows