"""Cross-cutting helpers. Nothing here knows about models or pruning."""
from __future__ import annotations

import gc
import os
import pickle
import random
import threading

import numpy as np

try:
    import torch
    HAS_TORCH = True
except Exception:                                    # pragma: no cover
    HAS_TORCH = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def fmt2(x) -> str:
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""


def run_tag(method, sparsity, alpha=None, beta=None, gamma=None, suffix="") -> str:
    """Canonical, collision-free name per experiment -> figures, CSV rows and npz files
    all share it."""
    t = f"{method}_s{int(round(float(sparsity) * 100))}"
    if alpha is not None:
        t += f"_a{alpha:g}"
    if beta is not None:
        t += f"_b{beta:g}"
    if gamma is not None:
        t += f"_g{gamma:.3f}"
    return t + suffix


# VRAM
def vram_gb() -> float:
    if not HAS_TORCH:
        return 0.0
    return sum(torch.cuda.memory_allocated(i)
               for i in range(torch.cuda.device_count())) / 1e9


def reset_peak() -> None:
    if HAS_TORCH:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)


def peak_vram() -> float:
    if not HAS_TORCH:
        return 0.0
    return sum(torch.cuda.max_memory_allocated(i)
               for i in range(torch.cuda.device_count())) / 1e9


class GPUMemorySampler:
    """Time-averaged ALLOCATED VRAM via background polling (summed across GPUs).

    Allocated memory tracks live tensors and averages down as pruning frees experts;
    reserved memory stays pinned by the caching allocator, so it is ignored. Peak is
    taken separately from torch, because a sub-interval spike can be missed by polling.
    """

    def __init__(self, interval_s: float = 0.05):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = None
        self._samples: list[float] = []

    def _loop(self):
        while not self._stop.is_set():
            self._samples.append(vram_gb())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def avg_gb(self) -> float:
        return round(float(np.mean(self._samples)), 2) if self._samples else 0.0


# io
def save_pkl(obj, path) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)                            # atomic


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def free() -> None:
    """Call after `del model, tok`. Deleting the caller's names is the caller's job."""
    gc.collect()
    if HAS_TORCH:
        torch.cuda.empty_cache()