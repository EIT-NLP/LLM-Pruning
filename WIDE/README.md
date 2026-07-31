<h1 align="center">🚀 WIDE</h1>

<p align="center">
  <b>Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.28418">📄 Paper (arXiv)</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/EIT-NLP/LLM-Pruning/tree/main/DynamicPruningKernels">⚡ Inference Kernel Library</a>
</p>

**WIDE** is a differentiable dynamic-width pruning framework for LLMs. At every Transformer layer, lightweight routers let each token independently select attention-head groups and FFN-channel groups, enabling adaptive computation for both prefill and decode.

## ✨ Highlights

| Capability | Description |
| --- | --- |
| **Token-wise dynamic width** | Allocates computation below the layer and sublayer level. |
| **Two-stage recovery** | Router-only calibration followed by optional LoRA tuning. |
| **Pruning-kernel co-design** | Combines mask reordering with block- and intra-block skipping. |

## 🧩 Method

| Router | Pruned unit | Affected operators |
| --- | --- | --- |
| Attention | Query-head group | Q projection, attention, and O projection |
| FFN | Contiguous intermediate-channel group | Up/Gate and Down projections |

K/V projections remain dense to avoid KV-cache eviction. Each router uses a small bottleneck and produces hard binary decisions with Gumbel-Softmax during training and deterministic routing during inference.

The training pipeline contains:

1. **Router calibration:** freeze the backbone and optimize only the routers.
2. **LoRA recovery:** load stage 1 and optionally tune LoRA adapters together with the routers.

## 🛠️ Environment

The complete reference environment is pinned in [`requirements.txt`](./requirements.txt). Core versions include:

| Package | Version |
| --- | --- |
| PyTorch | 2.8.0 |
| Transformers | 4.57.1 |
| Lightning | 2.6.0 |
| PEFT | 0.18.0 |
| FlashAttention | 2.8.3 |
| Liger Kernel | 0.6.4 |
| Datasets | 4.4.2 |
| LM Evaluation Harness | 0.4.10.dev0 |

From the `LLM-Pruning` repository root:

```bash
conda create -n wide python=3.11 -y
conda activate wide

pip install torch==2.8.0
pip install -r WIDE/requirements.txt
```

If FlashAttention needs to compile locally, install PyTorch first as above and rerun the second command with `--no-build-isolation`. Training uses BF16 and supports single-GPU or multi-GPU FSDP2 execution.

## 📚 Data

`main.py` accepts Hugging Face dataset IDs or local directories loadable by `datasets.load_dataset`. Training data needs a `train` split with a `text` column; evaluation data needs a `test` split. When the same dataset is used for both and has no test split, WIDE creates one automatically.

An optional utility creates a local text-only subset from a streaming dataset:

```bash
python WIDE/datasets/stream_sample.py \
  ZengXiangyu/RedPajama-Data-1T-Sample \
  --num-samples 200000 \
  --test-ratio 0.001 \
  --output-dir /path/to/redpajama_subset
```

Tokenized data is cached under `WIDE/datasets/` after the first run.

## ⚡ Quick Start

[`run_pipeline.sh`](./run_pipeline.sh) runs router calibration, stage-1 evaluation, LoRA recovery, and stage-2 evaluation.

Before running it, update:

- `base_path`, model path, and dataset paths;
- `HF_HOME`, `HF_TOKEN`, and ModelScope cache if needed;
- SwanLab logging directory and API key.

Then launch the full pipeline:

```bash
cd LLM-Pruning

SPARSITY=0.5 ATTN_G=4 FFN_G=128 \
  bash WIDE/run_pipeline.sh
```

`ATTN_G` is the number of query heads controlled by one routing decision and must divide the query-head count. `FFN_G` is the number of intermediate channels per group and must divide the FFN intermediate size. Larger groups are generally more kernel-friendly; smaller groups allow more flexible routing.

Main options:

| Argument | Description |
| --- | --- |
| `--sparsity` | Shared target sparsity |
| `--sparsity_attention`, `--sparsity_ffn` | Separate targets; set both when used |
| `--sparse_target` | `all`, `attention`, or `ffn` |
| `--attn_group_size`, `--ffn_group_size` | Attention and FFN routing granularity |
| `--route_rank_size` | Router bottleneck rank |

Training Llama3.1-8B with `ATTN_G=4`, `FFN_G=32` in an A100-SMX4-40G x 4 node requires 9h for stage1 and 13h (cpu-offload) for stage2.

## 📊 Evaluation

The pipeline evaluates WikiText-2 perplexity and zero-shot accuracy on ARC-Easy, ARC-Challenge, BoolQ, WinoGrande, PIQA, OpenBookQA, and HellaSwag using `lm-evaluation-harness`.

Evaluate an existing router checkpoint with:

```bash
python WIDE/main.py \
  --benchmark \
  --benchmark_tasks wikitext arc_easy arc_challenge boolq winogrande piqa openbookqa hellaswag \
  --benchmark_batch_size auto \
  --benchmark_max_batch_size 16 \
  --benchmark_max_length 4096 \
  --model_name Llama-3.1-8B \
  --model_path /path/to/Llama-3.1-8B \
  --router_ckpt_path /path/to/router.pth \
  --project_name WIDE-Bench \
  --wrapper_type attn_group_ffn_block \
  --sparse_target all \
  --sparsity 0.5 \
  --attn_group_size 4 \
  --ffn_group_size 128 \
  --route_rank_size 32
```

For stage 2, additionally pass `--lora_ckpt_path /path/to/lora.pth`.

```text
<save_dir>/
├── router_stage1/router.pth
└── router_stage2/
    ├── router.pth
    └── lora.pth
```

## ⚙️ Inference Kernel Library

This directory focuses on training and quality evaluation. Fused `gemm_mn`, `gemm_k`, and routed-attention implementations are maintained in [`DynamicPruningKernels`](../DynamicPruningKernels), an installable library with lazy backend dispatch across CuTe JIT and reference implementations. See its README for the latest installation, supported backends, and kernel variants.

## 📦 Repository Structure

```text
WIDE/
├── main.py                    # training and lm-eval entrypoint
├── mainloop.py                # data, optimization, and checkpointing
├── run_pipeline.sh            # stage 1 -> eval -> stage 2 -> eval
├── transformer_fsdp_hook.py   # FSDP2 sharding policy
├── requirements.txt           # pinned reference environment
├── datasets/stream_sample.py  # streaming dataset subset utility
└── wrapper/                   # routers and model wrappers
```

## 📝 Citation

```bibtex
@article{hu2026wide,
  title         = {WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning},
  author        = {Hu, Haozhe and Wu, Hao and Yin, Peiran and Han, Chao and Ma, Yunpu and Shen, Xiaoyu},
  year          = {2026},
  eprint        = {2607.28418},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI}
}
```
