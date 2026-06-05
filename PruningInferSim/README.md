# 🚀 LLM Pruning Wrapper for Inference Acceleration Simulation

**PruningInferSim** is a research-oriented profiling framework for simulating and benchmarking the inference acceleration potential of LLM pruning methods. It follows a GEMM-centric pruning taxonomy and maps pruning strategies onto the logical **M**, **N**, and **K** dimensions of matrix multiplication, making it easier to compare pruning families under the same model, kernels, cache behavior, and measurement pipeline.

It wraps Hugging Face causal language models, replaces selected Transformer submodules with pruning-aware wrappers, and profiles prefill/decode throughput with Triton, TileLang, FlashAttention, cuSPARSELt, and CUDA Graph support.

## ✨ Highlights

| Capability | What it gives you |
|---|---|
| 🧭 **GEMM-centric taxonomy** | Compare static/dynamic **M**, static **K**, low-rank **K**, static **NK**, cross-layer **NK**, and dynamic sparse attention under one abstraction. |
| 🧩 **Wrapper-based transformation** | Inject pruning behavior by replacing model, layer, attention, and MLP modules without rewriting the whole model stack. |
| ⚙️ **Kernel-level extensibility** | Plug in attention, MLP, mask, index, cache, router, threshold, and low-rank operators through registries. |
| 📊 **End-to-end profiling** | Measure TTFT/TPOT with warmup, repeated timing, CUDA Graph replay, L2 cache flushing, NVTX ranges, and optional PyTorch/NSYS/NCU profiling. |
| 🧪 **Simulation-first pruning** | Emulate target sparsity budgets with random masks/indices, without requiring a full calibration, retraining, or checkpoint release pipeline. |

## 🗺️ Contents

- [🚀 LLM Pruning Wrapper for Inference Acceleration Simulation](#-llm-pruning-wrapper-for-inference-acceleration-simulation)
  - [✨ Highlights](#-highlights)
  - [🗺️ Contents](#️-contents)
  - [📦 Repository Structure](#-repository-structure)
  - [🧬 Pruning Families and Configurations](#-pruning-families-and-configurations)
  - [🛠️ Environment](#️-environment)
  - [⚡ Quick Start](#-quick-start)
  - [🏃 Running Individual Profiles](#-running-individual-profiles)
    - [🧱 Dense Baseline](#-dense-baseline)
    - [🧬 Static NK Width Propagation](#-static-nk-width-propagation)
    - [⚡ Dynamic M Token-Wise Pruning](#-dynamic-m-token-wise-pruning)
    - [🔎 Torch Profiler](#-torch-profiler)
    - [🧪 NCU Single-Layer Profile](#-ncu-single-layer-profile)
  - [🧩 Wrapper Interface](#-wrapper-interface)
  - [➕ Adding a New Wrapper](#-adding-a-new-wrapper)
    - [1. Create a Wrapper File](#1-create-a-wrapper-file)
    - [2. Register the Wrapper](#2-register-the-wrapper)
    - [3. Add a YAML Config](#3-add-a-yaml-config)
  - [⚙️ Operator \& Kernel Interface](#️-operator--kernel-interface)
    - [Attention Operator](#attention-operator)
    - [MLP Operator](#mlp-operator)
  - [Mask, Index, Router, Cache, and Threshold Operators](#mask-index-router-cache-and-threshold-operators)
  - [📊 Profiling Notes](#-profiling-notes)
  - [📝 TODO / Roadmap](#-todo--roadmap)
  - [🧭 Known Assumptions and Scope](#-known-assumptions-and-scope)

## 📦 Repository Structure

```text
PruningInferSim/
├── main.py                     # CLI entrypoint for model loading, wrapping, and profiling
├── prologue.py                 # Hugging Face model initialization and wrapper construction
├── config.py                   # YAML loading and wrapper registry
├── run.sh                      # Example profiling script covering all pruning families
├── utils.py                    # Result table formatting and metric persistence
├── benchmark/
│   └── profiler.py             # TTFT/TPOT timing, CUDA Graph, torch profiler, and NCU helpers
├── wrapper/
│   ├── base.py                 # Base pruned MLP/attention/layer/causal-LM classes
│   ├── static/
│   │   ├── dense.py            # Dense baseline wrapper
│   │   ├── propagate.py        # Static M/NK propagation wrapper
│   │   ├── lowrank.py          # Static low-rank K wrapper
│   │   ├── unstructured.py     # Static mask/semi-structured K wrapper
│   │   └── *.yaml              # Static pruning configuration files
│   └── dynamic/
│       ├── propagate.py        # Dynamic token-wise M pruning wrapper
│       ├── sparse_attention.py # Dynamic NK sparse-attention wrapper
│       └── *.yaml              # Dynamic pruning configuration files
├── ops/
│   ├── __init__.py             # Operator registries
│   ├── attention/              # Dense, query-sparse, and block-sparse attention kernels
│   ├── mlp/                    # Dense, low-rank, dynamic-M, and semi-structured MLP kernels
│   ├── cache.py                # Sequence-first dynamic KV cache
│   ├── index.py                # Structured M/NK monkey-patching utilities
│   ├── mask.py                 # Unstructured and semi-structured mask utilities
│   ├── lowrank.py              # Low-rank monkey-patching utilities
│   ├── router.py               # Linear and bottleneck routers
│   ├── attention_threshold.py  # Sparse-attention threshold and block-mask generation
│   └── utils.py                # Triton RMSNorm/RoPE helper kernels
└── 3rdparty/
    ├── sglang/                 # Header dependency for JIT sparse kernels
    ├── cutlass/
    ├── dlpack/
    └── tvm-ffi/
```

## 🧬 Pruning Families and Configurations

| Family | CLI `--dynamic` | CLI `--style` | Example `--config_name` | Main wrapper |
|---|---:|---:|---:|---|
| Dense baseline | `static` | `dense` | `dense` | `wrapper/static/dense.py` |
| Static M depth/sublayer pruning | `static` | `propagate` | `static_m` | `wrapper/static/propagate.py` |
| Static K semi-structured pruning | `static` | `unstructured` | `static_k` | `wrapper/static/unstructured.py` |
| Static K low-rank pruning | `static` | `lowrank` | `static_k_lowrank` | `wrapper/static/lowrank.py` |
| Static NK width propagation | `static` | `propagate` | `static_nk` | `wrapper/static/propagate.py` |
| Static NK cross-layer propagation | `static` | `propagate` | `static_nk_cross` | `wrapper/static/propagate.py` |
| Dynamic M token-wise pruning | `dynamic` | `propagate` | `dynamic_m` | `wrapper/dynamic/propagate.py` |
| Dynamic NK sparse attention | `dynamic` | `sparse_attention` | `dynamic_nk` | `wrapper/dynamic/sparse_attention.py` |

Each YAML file describes the pruning type, target sparsity, backend, router, threshold, and kernel options for one pruning family. Passing `--sparsity` on the command line replaces YAML entries whose `estimated_sparsity` is set to `-1`.

## 🛠️ Environment

The profiling setup used in the accompanying paper is:

- CUDA 12.8
- PyTorch 2.9.1
- Transformers 4.57.1
- Triton 3.6.0
- TileLang 0.1.8
- FlashAttention 2
- cuSPARSELt 0.9.0
- SGLang JIT headers/runtime for cuSPARSELt integration

Additional Python packages used by the framework include:

```bash
pip install torch transformers accelerate pandas tqdm einops triton nvtx flash-attn
```

Optional features require extra packages:

```bash
pip install liger-kernel tilelang
```

For semi-structured static K profiling, install cuSPARSELt and update the paths in `ops/mlp/utils.py` if your installation differs from:

```text
/usr/include/libcusparseLt/13
/usr/lib/x86_64-linux-gnu/libcusparseLt/13
```

Most kernels are Python-level Triton/TileLang/JIT kernels and are compiled on first use. A separate CMake build is not required for the normal profiling path.

## ⚡ Quick Start

Set the model path in `run.sh`:

```bash
model_name=llama3.1-8b
model_path=/path/to/Llama-3.1-8B
```

Then run the full benchmark script:

```bash
bash run.sh
```

By default, `run.sh` profiles TTFT with:

- batch sizes: `1 8 16 32`
- sequence length: `1024`
- sparsity samples: `0.5 0.5 0.5`
- CUDA Graph enabled
- Liger RMSNorm/SwiGLU patches enabled where supported
- in-place KV-cache update enabled

Results are written to:

```text
metric_results/*.txt
```

## 🏃 Running Individual Profiles

### 🧱 Dense Baseline

```bash
python main.py \
  --model_name llama3.1-8b \
  --model_path /path/to/Llama-3.1-8B \
  --dynamic static \
  --style dense \
  --config_name dense \
  --benchmark_metric ttft \
  --batch_size 1 8 16 32 \
  --seq_len 1024 \
  --sparsity 0.0 \
  --num_warmup 10 \
  --num_repeat 50 \
  --cuda_graph \
  --inplace_update_kvcache
```

### 🧬 Static NK Width Propagation

```bash
python main.py \
  --model_name llama3.1-8b \
  --model_path /path/to/Llama-3.1-8B \
  --dynamic static \
  --style propagate \
  --config_name static_nk \
  --benchmark_metric ttft \
  --batch_size 1 \
  --seq_len 32768 \
  --sparsity 0.5 \
  --num_warmup 10 \
  --num_repeat 50 \
  --cuda_graph \
  --inplace_update_kvcache
```

### ⚡ Dynamic M Token-Wise Pruning

```bash
python main.py \
  --model_name llama3.1-8b \
  --model_path /path/to/Llama-3.1-8B \
  --dynamic dynamic \
  --style propagate \
  --config_name dynamic_m \
  --benchmark_metric tpot \
  --batch_size 1 8 16 \
  --seq_len 32768 \
  --sparsity 0.5 \
  --num_warmup 10 \
  --num_repeat 50 \
  --cuda_graph \
  --inplace_update_kvcache
```

### 🔎 Torch Profiler

```bash
python main.py \
  --model_name llama3.1-8b \
  --model_path /path/to/Llama-3.1-8B \
  --dynamic static \
  --style dense \
  --config_name dense \
  --benchmark_metric ttft \
  --batch_size 1 \
  --seq_len 2048 \
  --sparsity 0.0 \
  --torch_profiler
```

Profiler traces are saved under:

```text
profiler_logs/
```

### 🧪 NCU Single-Layer Profile

```bash
python main.py \
  --model_name llama3.1-8b \
  --model_path /path/to/Llama-3.1-8B \
  --dynamic static \
  --style dense \
  --config_name dense \
  --benchmark_metric ttft \
  --batch_size 1 \
  --seq_len 2048 \
  --sparsity 0.0 \
  --ncu_profiler
```

The NCU path profiles a representative decoder layer after constructing dummy inputs, position embeddings, and pruning kwargs.

## 🧩 Wrapper Interface

The wrapper stack has three layers:

1. `config.py` loads `wrapper/{dynamic}/{config_name}.yaml`.
2. `wrapper/{dynamic}/__init__.py` registers wrapper classes via `register_wrapper(style, dynamic)`.
3. `prologue.wrap_model()` instantiates the selected wrapper around the Hugging Face model and calls `post_load()`.

A wrapper class should subclass `PrunedModelForCausalLM` from `wrapper/base.py` and implement:

```python
def generate_pruning_kwargs(self, **kwargs) -> dict:
    """Create or update pruning masks, routes, indices, thresholds, or kernel metadata."""

def post_load(self, **kwargs):
    """Load or initialize router, LoRA, mask, low-rank, or full-model state after wrapping."""
```

Model-level wrappers usually contain:

- a model wrapper that owns embeddings, decoder layers, final norm, and RoPE
- a decoder-layer wrapper
- an attention wrapper
- an MLP wrapper

The dense implementation in `wrapper/static/dense.py` is the best starting point for a new wrapper.

## ➕ Adding a New Wrapper

### 1. Create a Wrapper File

Add a new file, for example:

```text
wrapper/static/my_pruning.py
```

Define a model wrapper:

```python
from wrapper.base import PrunedModelForCausalLM

class MyPruningForCausalLM(PrunedModelForCausalLM):
    def __init__(self, config, pruning_config, block, **kwargs):
        super().__init__(config, pruning_config, block, **kwargs)
        self.model = MyPruningModel(config, pruning_config, block.model, **kwargs)

    def generate_pruning_kwargs(self, **kwargs):
        # Return per-layer pruning metadata, or update modules in place.
        return {}

    def post_load(self, **kwargs):
        pass
```

### 2. Register the Wrapper

Edit the matching package initializer:

```python
# wrapper/static/__init__.py
from .my_pruning import MyPruningForCausalLM
from config import register_wrapper

register_wrapper("my_pruning", "static")(MyPruningForCausalLM)
```

Then invoke it with:

```bash
python main.py \
  --dynamic static \
  --style my_pruning \
  --config_name my_pruning \
  ...
```

### 3. Add a YAML Config

Create:

```text
wrapper/static/my_pruning.yaml
```

Example:

```yaml
cache_type: base

attention:
  estimated_sparsity: -1
  pruning_type: base
  backend: triton
  threshold: -1
  enable_autotune: false

mlp:
  estimated_sparsity: -1
  pruning_type: m
  backend: triton
  prefill_impl: auto
  decode_impl: auto
  enable_autotune: false
```

Use `estimated_sparsity: -1` when the CLI `--sparsity` value should overwrite the config.

## ⚙️ Operator & Kernel Interface

Operators are exposed through registries in `ops/__init__.py`.

### Attention Operator

1. Implement a kernel class under `ops/attention/`.
2. Follow the interface used by `DenseAttentionKernel`, `MSparseAttentionKernel`, or `BlockSparseAttentionKernel`.
3. Register it in `ops/__init__.py`:

```python
from .attention.my_attention import MyAttentionKernel

__ATTENTION__["my_attention"] = MyAttentionKernel
```

or, for block-sparse attention:

```python
__SPARSE_ATTENTION__["my_sparse_attention"] = MyAttentionKernel
```

YAML usage:

```yaml
attention:
  pruning_type: my_sparse_attention
  backend: triton
```

### MLP Operator

1. Implement a kernel class under `ops/mlp/`.
2. Follow the `DenseMLPKernel` call convention:

```python
kernel(
  x,
  w_up=...,
  w_gate=...,
  w_down=...,
  route_mask=...,
  activation=...,
  estimated_sparsity=...,
  backend=...,
)
```

3. Register it:

```python
from .mlp.my_mlp import MyMLPKernel

__MLP__["my_mlp"] = MyMLPKernel
```

YAML usage:

```yaml
mlp:
  pruning_type: my_mlp
  backend: triton
  estimated_sparsity: -1
```

### Mask, Index, Router, Cache, and Threshold Operators

The same registry pattern is used for other extension points:

| Registry | Purpose |
|---|---|
| `__MASK__` | Static weight mask generation and monkey-patching |
| `__INDEX__` | Structured M/NK index sampling and dimension propagation |
| `__ROUTER__` | Dynamic route prediction modules |
| `__KV_CACHE__` | KV-cache implementations |
| `__LOW_RANK__` | Low-rank factorization and monkey-patching |
| `__THRESHOLD__` | Sparse-attention threshold/mask generation |

## 📊 Profiling Notes

- `ttft` profiles the prompt/prefill forward pass.
- `tpot` first builds a dummy KV cache, then profiles one-token decode steps.
- `--cuda_graph` captures a warmed-up forward pass and replays it for stable timing.
- `--inplace_update_kvcache` avoids dynamic KV-cache growth during decode replay.
- `--liger_kernel` patches supported Llama/Qwen kernels before model loading.
- `--torch_profiler` writes TensorBoard traces to `profiler_logs/`.
- `--ncu_profiler` runs a representative single-layer path for low-level kernel inspection.

The profiler reports:

```text
mean latency, min latency, max latency, mean throughput, min throughput, max throughput
```

and persists a table under `metric_results/`.

## 📝 TODO / Roadmap

- [ ] Support fine-grained per-layer attention/FFN-wise sparsity control.

## 🧭 Known Assumptions and Scope

- The benchmark simulates pruning distributions with random masks/indices for latency studies; it does not train pruning policies.
- The current wrappers assume decoder-only Hugging Face causal LMs with Llama/Qwen-style module names such as `model.layers`, `self_attn.q_proj`, `mlp.up_proj`, and `rotary_emb`.
- Current kernels have been tested on NVIDIA RTX Pro 6000 (sm120) and NVIDIA A100 (sm80), without customized optimizations for sm90/sm100, e.g., TMA/WASP/Persist kernel.
