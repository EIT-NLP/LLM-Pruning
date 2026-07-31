<h1 align="center"><b>Repository of 🚀 LLM-Pruning</b></h1>

**🚀LLM-Pruning** is a unified repository which collects the works of [EIT-NLP Lab](https://idt.eitech.edu.cn/nlp/#/) on LLM pruning, including calibration datasets, pruning methods, training & evaluation code, and their inference implementation.

## ✨ Overview

LLM pruning aims to reduce the computational & memory cost of LLMs by removing, sparsifying, or approximating redundant components while preserving model quality. This repository focuses on a broad range of LLM pruning directions, e.g., *unstructured pruning*, *semi-structured pruning*, *structured pruning*, *low-rank approximation*, *attention pruning (sparse attention and KV pruning)*, as well as their combinations with *quantization*, *distillation*, and so on.

## 📰 Latest News
- **Coming soon**: More pruning projects, papers, checkpoints, and evaluation scripts...

## 🗺️ Contents

| Work | Status | Code | Paper / Project | What it provides |
|---|---|---|---|---|
| **PruningInferSim** | [arxiv](https://arxiv.org/abs/2606.09080) | [`PruningInferSim/`](./PruningInferSim) | *Beyond FLOPs: Benchmarking Real Inference Acceleration of LLM Pruning under a GEMM-Centric Taxonomy* | A pruning-wrapper and profiling framework for simulating TTFT/TPOT speedups under a GEMM-centric taxonomy. |
| **InformedRouting** | [ICML2026](https://icml.cc/virtual/2026/poster/64064) | [`LFF/`](./LFF) | Bring Future Vision: Dynamic Computation Allocation Guided by Lightweight Feature Forecaster | Adding lightweight feature forecaster for Attn/FFN Module. |
| **WIDE** | [arxiv](https://arxiv.org/abs/2607.28418) | [`WIDE/`](./WIDE) | *WIDE: Boosting Adaptive LLM Inference via Token-level Dynamic Width Pruning* | A differentiable framework for training token-wise dynamic-width LLMs with routed KV head groups and FFN-channel groups. |
| **DynamicPruningKernels** | -- | [`DynamicPruningKernels/`](./DynamicPruningKernels) | *Unified CuTe/Python DSL Kernels for Token-wise Dynamic Structured Pruning* | An kernel library with fused GEMM & attention operators, and SGL JIT-based CuTe autotune interface. |
| More works | Coming soon | - | - | Future releases in the LLM pruning series. |
