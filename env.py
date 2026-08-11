# hpp/env.py
import os
from pathlib import Path

def configure_hf_cache(path: str = "/tmp/an261-hf-cache") -> None:
    """Must be called before torch/transformers/hpp.models are imported."""
    cache = Path(path)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")