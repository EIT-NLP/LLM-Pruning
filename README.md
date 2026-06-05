<h1 align="center"><b>Repository of 🚀 LLM-Pruning</b></h1>

**🚀LLM-Pruning** is a unified repository which collects the works of [EIT-NLP Lab](https://idt.eitech.edu.cn/nlp/#/) on LLM pruning, including calibration datasets, pruning methods, training & evaluation code, and their inference implementation.

## ✨ Overview

LLM pruning aims to reduce the computational & memory cost of LLMs by removing, sparsifying, or approximating redundant components while preserving model quality. This repository focuses on a broad range of LLM pruning directions, e.g., *unstructured pruning*, *semi-structured pruning*, *structured pruning*, *low-rank approximation*, *attention pruning (sparse attention and KV pruning)*, as well as their combinations with *quantization*, *distillation*, and so on.

## 📰 Latest News
- **Coming soon**: More pruning projects, papers, checkpoints, and evaluation scripts...

## 🗺️ Contents

| Work | Status | Code | Paper / Project | What it provides |
|---|---|---|---|---|
| **PruningInferSim** | Building | [`PruningInferSim/`](./PruningInferSim) | *From Theory to Practice: Benchmarking LLM Pruning Inference Acceleration under GEMM-Centric Taxonomy* | A pruning-wrapper and profiling framework for simulating TTFT/TPOT speedups under a GEMM-centric taxonomy. |
| More works | Coming soon | - | - | Future releases in the LLM pruning series. |
