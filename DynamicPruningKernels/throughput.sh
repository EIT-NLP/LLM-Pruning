#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
KERNEL_FAMILY="${KERNEL_FAMILY:-gemm}"
KERNEL="${KERNEL:-gemm_mn}"
KERNEL_VERSION="${KERNEL_VERSION:-1}"
ARCH="${ARCH:-}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"

M_VALUES="${M_VALUES:-128 256 512}"
N="${N:-4096}"
K="${K:-4096}"
G="${G:-128}"

B_VALUES="${B_VALUES:-1}"
TQ_VALUES="${TQ_VALUES:-1}"
TK_VALUES="${TK_VALUES:-4096}"
HQ="${HQ:-32}"
HK="${HK:-8}"
D="${D:-128}"
NG="${NG:-8}"
IS_CAUSAL="${IS_CAUSAL:-true}"

SPARSITY_VALUES="${SPARSITY_VALUES:-0.5}"
AUTOTUNE="${AUTOTUNE:-true}"
CUDAGRAPH="${CUDAGRAPH:-true}"
CHECK_PRECISION="${CHECK_PRECISION:-true}"
RUN_TYPE="${RUN_TYPE:-run_throughput}"

export JIT_AUTOTUNE_VERBOSE="${JIT_AUTOTUNE_VERBOSE:-1}"
export JIT_AUTOTUNE_COMPILE_WORKERS="${JIT_AUTOTUNE_COMPILE_WORKERS:-8}"
export JIT_AUTOTUNE_FORCE_TUNE="${JIT_AUTOTUNE_FORCE_TUNE:-0}"
export JIT_AUTOTUNE_FORCE_RECOMPILE="${JIT_AUTOTUNE_FORCE_RECOMPILE:-0}"
export TILELANG_AUTO_TUNING_MAX_CPU_COUNT="${TILELANG_AUTO_TUNING_MAX_CPU_COUNT:-12}"

read -r -a KERNEL_VERSIONS <<< "${KERNEL_VERSION}"
read -r -a M_ARGS <<< "${M_VALUES}"
read -r -a B_ARGS <<< "${B_VALUES}"
read -r -a TQ_ARGS <<< "${TQ_VALUES}"
read -r -a TK_ARGS <<< "${TK_VALUES}"
read -r -a SPARSITY_ARGS <<< "${SPARSITY_VALUES}"

ARGS=(
  --kernel-family "${KERNEL_FAMILY}"
  --kernel "${KERNEL}"
  --kernel-version "${KERNEL_VERSIONS[@]}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --sparsity "${SPARSITY_ARGS[@]}"
  --M "${M_ARGS[@]}"
  --N "${N}"
  --K "${K}"
  --G "${G}"
  --B "${B_ARGS[@]}"
  --Tq "${TQ_ARGS[@]}"
  --Tk "${TK_ARGS[@]}"
  --Hq "${HQ}"
  --Hk "${HK}"
  --D "${D}"
  --NG "${NG}"
  --is_causal "${IS_CAUSAL}"
  --check-precision "${CHECK_PRECISION}"
  --run-type "${RUN_TYPE}"
  --autotune "${AUTOTUNE}"
  --cudagraph "${CUDAGRAPH}"
)

if [[ -n "${ARCH}" ]]; then
  ARGS+=(--arch "${ARCH}")
fi

"${PYTHON_BIN}" run.py "${ARGS[@]}"
