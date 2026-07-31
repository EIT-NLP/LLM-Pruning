import os
import itertools
from pathlib import Path
from typing import Any, Optional

import torch


def _configure_cuda_home() -> None:
    """Prefer a complete CUDA toolkit when TileLang would find nvcc-only wheels."""
    if os.environ.get("CUDA_HOME"):
        return
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except (ImportError, OSError):
        return
    if CUDA_HOME is None:
        return
    cuda_home = Path(CUDA_HOME)
    if (cuda_home / "bin" / "nvcc").is_file():
        os.environ["CUDA_HOME"] = str(cuda_home)


_configure_cuda_home()

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune, set_autotune_inputs


def _sync_tilelang_cuda_home() -> None:
    """Update TileLang if another module imported it before this backend."""
    cuda_home = os.environ.get("CUDA_HOME")
    if not cuda_home:
        return

    import tilelang.env as tilelang_env
    import tilelang.contrib.nvcc as tilelang_nvcc

    tilelang_env.CUDA_HOME = cuda_home
    tilelang_nvcc.CUDA_HOME = cuda_home


_sync_tilelang_cuda_home()

PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

def _cuda_cccl_compile_flags() -> list[str]:
    """Locate libcudacxx headers omitted by some pip CUDA toolkits."""
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home is None:
        try:
            from torch.utils.cpp_extension import CUDA_HOME

            cuda_home = CUDA_HOME
        except (ImportError, OSError):
            cuda_home = None

    if cuda_home is None:
        return []

    root = Path(cuda_home)
    candidates = [root / "include" / "cccl", root / "include"]
    candidates.extend(root.glob("targets/*/include/cccl"))
    for include_dir in candidates:
        if (include_dir / "cuda" / "atomic").is_file():
            return [f"-I{include_dir}"]
    return []


def decoding_head_space(*_: Any, **__: Any) -> list[dict[str, int]]:
    return [
        {
            "BN": bn,
            "SplitKV": split_kv,
            "num_stages": pipeline,
            "threads": num_thread,
        }
        for bn, split_kv, pipeline, num_thread in itertools.product(
            [32, 64],
            [1, 2, 3, 4, 5, 6, 7, 8],
            [1, 2],
            [128],
        )
    ]

@autotune(
    configs=decoding_head_space,
    cache_input_tensors=False,
)
@tilelang.jit(pass_configs=PASS_CFG, compile_flags=_cuda_cccl_compile_flags())
def decoding_head_device(
    B: int, BM: int, Tk: int, Hq: int, Hk: int, D: int, NG: int, pSplit: int,
    scale: float, is_leftpad: bool, dtype: T.DType,
    BN: int=64,
    SplitKV: int=1,
    num_stages: int=2,
    threads: int=128,
):
    scale *= 1.44269504 # 1/log(2)
    kv_group_size = Hq // Hk
    group_size = Hq // NG
    valid_BM = min(group_size, kv_group_size)
    head_blocks = max(Hk, NG)
    total_k_tiles = T.ceildiv(Tk, BN)
    accum_dtype = T.float32

    @T.prim_func
    def decoding_split(Q, K, V, O, O_partial, Mask, Leftpad):
        Q: T.Tensor[[B, 1, Hq, D], dtype]
        K: T.Tensor[[B, Tk, Hk, D], dtype]
        V: T.Tensor[[B, Tk, Hk, D], dtype]
        O: T.Tensor[[B, 1, Hq, D], dtype]
        O_partial: T.Tensor[[B, Hq, pSplit, D], dtype]
        Mask: T.Tensor[[B, 1, NG], T.bool]
        Leftpad: T.Tensor[[B], T.int32]

        LSE: T.Tensor = T.alloc_buffer([B, Hq, pSplit], accum_dtype)

        with T.Kernel(SplitKV, head_blocks, B, threads=threads) as (bidx, bidy, bidz):
            sQ = T.alloc_shared([BM, D], dtype)
            sK = T.alloc_shared([BN, D], dtype)
            sV = T.alloc_shared([BN, D], dtype)
            sP = T.alloc_shared([BM, BN], dtype)
            sO = T.alloc_shared([BM, D], dtype)

            rLeftpad = T.alloc_fragment([1], T.int32)
            rMask = T.alloc_fragment([1], T.bool)

            rMax = T.alloc_fragment([BM], accum_dtype)
            rMax_tmp = T.alloc_fragment([BM], accum_dtype)
            rScale = T.alloc_fragment([BM], accum_dtype)
            rSum = T.alloc_fragment([BM], accum_dtype)
            rLogsum = T.alloc_fragment([BM], accum_dtype)
            rAcc = T.alloc_fragment([BM, D], accum_dtype)
            rPc = T.alloc_fragment([BM, BN], accum_dtype)

            query_head = valid_BM * bidy
            key_head = query_head // kv_group_size
            group_id = query_head // (Hq // NG)
            T.copy(Mask[bidz, 0, group_id], rMask)
            if rMask[0] != 0:
                T.fill(rMax, -T.infinity(accum_dtype))
                T.fill(rSum, 0.0)
                T.fill(rAcc, 0.0)

                rLeftpad[0] = 0
                if is_leftpad: T.copy(Leftpad[bidz], rLeftpad)

                T.copy(Q[bidz, 0, query_head:query_head + BM, :], sQ, disable_tma=True)

                tiles_per_split = T.floordiv(total_k_tiles, SplitKV)
                remaining_tiles = T.floormod(total_k_tiles, SplitKV)
                split_k_tiles = tiles_per_split + T.if_then_else(
                    bidx < remaining_tiles, 1, 0
                )
                base_k_tile = tiles_per_split * bidx + T.min(
                    bidx, remaining_tiles
                )
                kv_start = T.max(base_k_tile * BN, rLeftpad[0])
                kv_end = T.min(Tk, (base_k_tile + split_k_tiles) * BN)
                iter_num = T.max(0, T.ceildiv(kv_end - kv_start, BN))

                for it in T.Pipelined(iter_num, num_stages=num_stages):
                    iter_kv_start = kv_start + it * BN
                    iter_kv_end = iter_kv_start + BN

                    T.copy(K[bidz, iter_kv_start:iter_kv_end, key_head, :], sK)
                    T.copy(V[bidz, iter_kv_start:iter_kv_end, key_head, :], sV)
                    T.clear(rPc)
                    T.gemm(sQ, sK, rPc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(BM, BN):
                        rPc[i, j] = T.if_then_else(
                            iter_kv_start + j < kv_end,
                            rPc[i, j] * scale,
                            -T.infinity(accum_dtype)
                        )
                    T.fill(rMax_tmp, -T.infinity(accum_dtype))
                    T.reduce_max(rPc, rMax_tmp, clear=False)
                    for i in T.Parallel(BM):
                        rMax_tmp[i] = T.max(rMax_tmp[i], rMax[i])
                        rScale[i] = T.exp2(rMax[i] - rMax_tmp[i])
                        rSum[i] *= rScale[i]
                    for i, j in T.Parallel(BM, BN):
                        rPc[i, j] = T.exp2(rPc[i, j] - rMax_tmp[i])
                    T.reduce_sum(rPc, rSum, clear=False)
                    for i, j in T.Parallel(BM, D):
                        rAcc[i, j] *= rScale[i]

                    T.copy(rPc, sP)
                    T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)
                    T.copy(rMax_tmp, rMax)

                for i, j in T.Parallel(BM, D):
                    rAcc[i, j] /= rSum[i]
                T.copy(rAcc, sO)

                for i in T.Parallel(BM):
                    if i < valid_BM:
                        rLogsum[i] = T.log2(rSum[i]) + rMax[i]
                        LSE[bidz, query_head + i, bidx] = rLogsum[i]

                for i, j in T.Parallel(BM, D):
                    if i < valid_BM:
                        O_partial[bidz, query_head + i, bidx, j] = sO[i, j]

        with T.Kernel(Hq, B) as (bidx, bidy):
            rO = T.alloc_fragment([pSplit, D], accum_dtype)
            rO_final = T.alloc_fragment([D], accum_dtype)
            rO_final_cast = T.alloc_fragment([D], dtype)
            rLSE = T.alloc_fragment([pSplit], accum_dtype)
            rMax = T.alloc_fragment([1], accum_dtype)
            rScale = T.alloc_fragment([pSplit], accum_dtype)
            rSum = T.alloc_fragment([1], accum_dtype)
            rMask = T.alloc_fragment([1], T.bool)

            group_id = bidx // (Hq // NG)
            T.copy(Mask[bidy, 0, group_id], rMask)
            if rMask[0] != 0:
                T.copy(LSE[bidy, bidx, :], rLSE)
                T.copy(O_partial[bidy, bidx, :, :], rO)
                for i, j in T.Parallel(pSplit, D):
                    rO[i, j] = T.if_then_else(i < SplitKV, rO[i, j], 0)
                for i in T.Parallel(pSplit):
                    rLSE[i] = T.if_then_else(i < SplitKV, rLSE[i], -T.infinity(accum_dtype))

                T.fill(rMax, -T.infinity(accum_dtype))
                T.reduce_max(rLSE, rMax, clear=False)
                for i in T.Parallel(pSplit):
                    rScale[i] = T.if_then_else(i < SplitKV, T.exp2(rLSE[i] - rMax[0]), 0)
                T.reduce_sum(rScale, rSum)
                for i, j in T.Parallel(pSplit, D):
                    rO[i, j] *= rScale[i]
                T.reduce_sum(rO, rO_final, dim=0)
                for i in T.Parallel(D):
                    rO_final[i] /= rSum[0]

                T.copy(rO_final, rO_final_cast)
                T.copy(rO_final_cast, O[bidy, 0, bidx, :])

    @T.prim_func
    def decoding_merge(Q, K, V, O, O_partial, Mask, Leftpad):
        Q: T.Tensor[[B, 1, Hq, D], dtype]
        K: T.Tensor[[B, Tk, Hk, D], dtype]
        V: T.Tensor[[B, Tk, Hk, D], dtype]
        O: T.Tensor[[B, 1, Hq, D], dtype]
        O_partial: T.Tensor[[B, Hq, pSplit, D], dtype]
        Mask: T.Tensor[[B, 1, NG], T.bool]
        Leftpad: T.Tensor[[B], T.int32]

        with T.Kernel(SplitKV, head_blocks, B, threads=threads) as (bidx, bidy, bidz):
            sQ = T.alloc_shared([BM, D], dtype)
            sK = T.alloc_shared([BN, D], dtype)
            sV = T.alloc_shared([BN, D], dtype)
            sP = T.alloc_shared([BM, BN], dtype)
            sO = T.alloc_shared([BM, D], dtype)

            rLeftpad = T.alloc_fragment([1], T.int32)
            rMask = T.alloc_fragment([1], T.bool)

            rMax = T.alloc_fragment([BM], accum_dtype)
            rMax_tmp = T.alloc_fragment([BM], accum_dtype)
            rScale = T.alloc_fragment([BM], accum_dtype)
            rSum = T.alloc_fragment([BM], accum_dtype)
            rAcc = T.alloc_fragment([BM, D], accum_dtype)
            rPc = T.alloc_fragment([BM, BN], accum_dtype)

            query_head = valid_BM * bidy
            key_head = query_head // kv_group_size
            group_id = query_head // (Hq // NG)
            T.copy(Mask[bidz, 0, group_id], rMask)
            if rMask[0] != 0:
                T.fill(rMax, -T.infinity(accum_dtype))
                T.fill(rSum, 0.0)
                T.fill(rAcc, 0.0)

                rLeftpad[0] = 0
                if is_leftpad: T.copy(Leftpad[bidz], rLeftpad)

                T.copy(Q[bidz, 0, query_head:query_head + BM, :], sQ, disable_tma=True)
                kv_start = rLeftpad[0]
                kv_end = Tk
                iter_num = T.max(0, T.ceildiv(kv_end - kv_start, BN))

                for it in T.Pipelined(iter_num, num_stages=num_stages):
                    iter_kv_start = kv_start + it * BN
                    iter_kv_end = iter_kv_start + BN

                    T.copy(K[bidz, iter_kv_start:iter_kv_end, key_head, :], sK)
                    T.copy(V[bidz, iter_kv_start:iter_kv_end, key_head, :], sV)
                    T.clear(rPc)
                    T.gemm(sQ, sK, rPc, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    for i, j in T.Parallel(BM, BN):
                        rPc[i, j] = T.if_then_else(
                            iter_kv_start + j < kv_end,
                            rPc[i, j] * scale,
                            -T.infinity(accum_dtype)
                        )
                    T.fill(rMax_tmp, -T.infinity(accum_dtype))
                    T.reduce_max(rPc, rMax_tmp, dim=-1, clear=False)
                    for i in T.Parallel(BM):
                        rMax_tmp[i] = T.max(rMax_tmp[i], rMax[i])
                        rScale[i] = T.exp2(rMax[i] - rMax_tmp[i])
                        rSum[i] *= rScale[i]
                    for i, j in T.Parallel(BM, BN):
                        rPc[i, j] = T.exp2(rPc[i, j] - rMax_tmp[i])
                    T.reduce_sum(rPc, rSum, clear=False)
                    for i, j in T.Parallel(BM, D):
                        rAcc[i, j] *= rScale[i]

                    T.copy(rPc, sP)
                    T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)
                    T.copy(rMax_tmp, rMax)

                for i, j in T.Parallel(BM, D):
                    rAcc[i, j] /= rSum[i]

                T.copy(rAcc, sO)
                for i, j in T.Parallel(BM, D):
                    if i < valid_BM:
                        O[bidz, 0, query_head + i, j] = rAcc[i, j]

    if SplitKV > 1:
        return decoding_split
    else:
        return decoding_merge


def decoding_head_host(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    Leftpad: Optional[torch.Tensor]=None,
    O: Optional[torch.Tensor]=None,
    scale: Optional[float]=None,
    estimate_sparsity: Optional[float]=0.5,
    autotune: bool=True,
    **kwargs,
):
    assert Q.dim() == 4 and K.dim() ==4 and V.dim() == 4
    assert Q.shape[1] == 1
    assert K.shape == V.shape
    assert Mask.dim() == 3
    assert estimate_sparsity <= 1
    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity

    B, Tq, Hq, D = Q.shape
    _, Tk, Hk, _ = K.shape
    _, _, NG = Mask.shape
    if Hq % Hk or Hq % NG:
        raise ValueError(
            "query heads must be divisible by KV heads and route groups"
        )

    if O is None: O = torch.zeros((B, Tq, Hq, D), dtype=Q.dtype, device=Q.device)

    is_leftpad = True
    if Leftpad is None:
        Leftpad = torch.empty((B,), dtype=torch.int32, device=Q.device)
        is_leftpad = False
    else:
        Leftpad = Leftpad.to(
            device=Q.device,
            dtype=torch.int32,
        ).contiguous()

    if scale is None: scale = D ** -0.5
    BM = max(16, tilelang.next_power_of_2(Hq // Hk))
    pSplit = 8
    tl_dtype = getattr(T, str(Q.dtype).split('.')[-1])

    O_partial = torch.empty((B, Hq, pSplit, D), dtype=Q.dtype, device=Q.device)

    compile_kwargs = dict(
        B=B,
        BM=BM,
        Tk=Tk,
        Hq=Hq,
        Hk=Hk,
        D=D,
        NG=NG,
        pSplit=pSplit,
        scale=scale,
        is_leftpad=is_leftpad,
        dtype=tl_dtype,
    )

    if not autotune:
        compile_kwargs.update(
            BN=int(kwargs.get("BN", 64)),
            SplitKV=int(kwargs.get("SplitKV", 1)),
            num_stages=int(kwargs.get("num_stages", 2)),
            threads=int(kwargs.get("threads", 128)),
        )

    with set_autotune_inputs(Q, K, V, O, O_partial, Mask, Leftpad):
        kernel = decoding_head_device(**compile_kwargs)

    kernel(Q, K, V, O, O_partial, Mask, Leftpad)
    return O
