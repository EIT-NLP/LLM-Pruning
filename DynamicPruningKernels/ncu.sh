#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

command -v ncu >/dev/null 2>&1 || {
  echo "ncu was not found; install NVIDIA Nsight Compute or add it to PATH." >&2
  exit 1
}

PYTHON_BIN="${PYTHON_BIN:-python}"
KERNEL_FAMILY="${KERNEL_FAMILY:-gemm}"
KERNEL="${KERNEL:-gemm_mn}"
KERNEL_VERSION="${KERNEL_VERSION:-1}"
ARCH="${ARCH:-sm12x}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"

M="${M:-128}"
N="${N:-4096}"
K="${K:-4096}"
G="${G:-128}"

B="${B:-1}"
TQ="${TQ:-1}"
TK="${TK:-4096}"
HQ="${HQ:-32}"
HK="${HK:-8}"
D="${D:-128}"
NG="${NG:-8}"
IS_CAUSAL="${IS_CAUSAL:-true}"

SPARSITY="${SPARSITY:-0.5}"
AUTOTUNE="${AUTOTUNE:-true}"
CHECK_PRECISION="${CHECK_PRECISION:-true}"
NCU_SET="${NCU_SET:-full}"

export CUTE_JIT_CUDA_CFLAGS="${CUTE_JIT_CUDA_CFLAGS:-} -lineinfo --generate-line-info"
export JIT_AUTOTUNE_VERBOSE="${JIT_AUTOTUNE_VERBOSE:-1}"
export JIT_AUTOTUNE_COMPILE_WORKERS="${JIT_AUTOTUNE_COMPILE_WORKERS:-8}"
export JIT_AUTOTUNE_FORCE_TUNE="${JIT_AUTOTUNE_FORCE_TUNE:-0}"
export JIT_AUTOTUNE_FORCE_RECOMPILE="${JIT_AUTOTUNE_FORCE_RECOMPILE:-0}"

TIMESTAMP="$(date +%Y%m%d%H%M)"
if [[ "${KERNEL_FAMILY}" == "attention" ]]; then
  NCU_NAME="${KERNEL}_${B}_${TQ}_${TK}_${HQ}_${HK}_${D}_${NG}_${SPARSITY}_ts${TIMESTAMP}"
else
  NCU_NAME="${KERNEL}_${M}_${N}_${K}_${G}_${SPARSITY}_ts${TIMESTAMP}"
fi

NCU_PATH="${ROOT_DIR}/ncu_profile/${KERNEL}/v${KERNEL_VERSION}/${ARCH}"
KERNEL_REGEX="${KERNEL_REGEX:-regex:^((?!elementwise|reduce|sort|Sort|index|indices).)*$}"
mkdir -p "${NCU_PATH}"

NCU_PROFILE_AFTER_WARMUP=1 \
ncu \
  --set "${NCU_SET}" \
  --target-processes all \
  --profile-from-start no \
  --import-source yes \
  --source-folders "${ROOT_DIR}" \
  --kernel-name "${KERNEL_REGEX}" \
  -f -o "${NCU_PATH}/${NCU_NAME}" \
  "${PYTHON_BIN}" ./run.py \
    --kernel-family "${KERNEL_FAMILY}" \
    --kernel "${KERNEL}" \
    --kernel-version "${KERNEL_VERSION}" \
    --arch "${ARCH}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --sparsity "${SPARSITY}" \
    --M "${M}" \
    --N "${N}" \
    --K "${K}" \
    --G "${G}" \
    --B "${B}" \
    --Tq "${TQ}" \
    --Tk "${TK}" \
    --Hq "${HQ}" \
    --Hk "${HK}" \
    --D "${D}" \
    --NG "${NG}" \
    --is_causal "${IS_CAUSAL}" \
    --autotune "${AUTOTUNE}" \
    --check-precision "${CHECK_PRECISION}"
