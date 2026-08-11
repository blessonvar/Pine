"""One module per architecture: config factory + ModelAdapter.

Adding a model = adding one file here. Nothing else in the package changes.

get_adapter() looks up cfg.model_tag, so every model_tag MUST be a key here. DeepSeek is
registered under both its tag and its old name, so `build("deepseekmoe16b")` in an existing
notebook keeps working while artefacts are written under the unambiguous "_chat" tag.
"""
from hpp.models import deepseek, gptoss, mixtral, qwen3

_REG = {
    "gptoss20b": gptoss,
    "deepseekmoe16b_chat": deepseek,
    "deepseekmoe16b": deepseek,          # alias: the notebook's build("deepseekmoe16b")
    "qwen3_30b_a3b": qwen3,
    "mixtral8x7b": mixtral,
}


def get_config(tag: str, **kwargs):
    if tag not in _REG:
        raise KeyError(f"unknown model tag {tag!r}; have {sorted(_REG)}")
    return _REG[tag].config(**kwargs)


def get_adapter(cfg):
    if cfg.model_tag not in _REG:
        raise KeyError(f"unknown model tag {cfg.model_tag!r}; have {sorted(_REG)}")
    return _REG[cfg.model_tag].adapter(cfg)


def build(tag: str, **kwargs):
    """-> (cfg, adapter), directories created."""
    cfg = get_config(tag, **kwargs)
    cfg.ensure_dirs()
    return cfg, get_adapter(cfg)