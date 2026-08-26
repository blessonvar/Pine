# Installation

## Requirements

- Linux (tested on Ubuntu 22.04)
- Python 3.10+
- CUDA 12.1+ with a matching NVIDIA driver
- At least one GPU with sufficient memory to hold the dense model in the chosen precision (see the table below)

## Step 1: create the environment

```
conda create -n pine python=3.10 -y
conda activate pine
```

## Step 2: install PyTorch

Install the build matching your CUDA version:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Step 3: install Pine and its dependencies

```
git clone https://github.com/blessonvar/hpp
cd pine
pip install -e .
```

This installs the core dependencies: `transformers`, `accelerate`, `datasets`, `safetensors`, `bitsandbytes`, `peft`, `numpy`, `pandas`, and `matplotlib`.

## Step 4: install the evaluation harnesses

Multiple-choice benchmarks are evaluated with `lm-evaluation-harness`, and code generation with the `EvalPlus` CLI:

```
pip install lm-eval evalplus
```

## Step 5: verify the installation

```
python -c "from hpp.models import build; cfg, adapter = build('gptoss20b', quantize=True); print(cfg)"
```

A successful run prints the resolved `HPPConfig`, including the derived expert count, top-k, and channel dimensions for the selected architecture.

## Model weights

Weights are downloaded from the Hugging Face hub on first use and stored in `cache_dir` (default `llm_weights`). Gated models require authentication:

```
huggingface-cli login
```

## Memory notes

Profiling requires a single forward pass over the dense model, so peak memory during profiling is bounded by the dense model footprint rather than by the pruning itself. Set `quantize=True` in `build(...)` to load in 4-bit precision when the dense model does not fit; saliency statistics are collected on the dequantised activations and remain comparable across precisions.
