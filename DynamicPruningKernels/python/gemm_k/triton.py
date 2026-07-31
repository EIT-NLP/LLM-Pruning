import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

gemm_configs = [
    triton.Config(
        {'BM': bm, "BN": bn, "SplitK": split_k},
        num_warps=4,
        num_stages=3,
    )
    for bm in [16, 32, 64, 128, 256]
    for bn in [16, 32, 64, 128, 256]
    for split_k in [1, 2, 4, 8, 16]
]
gemv_configs = [
    triton.Config(
        {"BN": bn, "SplitK": split_k},
        num_warps=4,
        num_stages=3,
    )
    for bn in [16, 32, 64, 128, 256]
    for split_k in [1, 2, 4, 8, 16]
]

def prune_gemm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    M = args["M"]
    N = args["N"]
    K = args["K"]
    G = args["G"]
    NG = K // G

    max_bm = min(256, max(16, triton.next_power_of_2(int(M))))
    max_bn = min(256, max(16, triton.next_power_of_2(int(N))))
    bm_space = [max_bm, max_bm // 2]
    bn_space = [max_bn, max_bn // 2]

    return [
        cfg for cfg in configs
        if cfg.kwargs['BM'] in bm_space and cfg.kwargs['BN'] in bn_space and (cfg.kwargs["SplitK"] <= NG and (K // cfg.kwargs["SplitK"]) % (K // NG) == 0)
    ]

def prune_gemv_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    K = args["K"]
    G = args["G"]
    NG = K // G

    return [
        cfg for cfg in configs
        if (cfg.kwargs["SplitK"] <= NG and (K // cfg.kwargs["SplitK"]) % (K // NG) == 0)
    ]

@triton.autotune(
    configs=gemm_configs,
    key=["M", "N", "K", "G"],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    restore_value=["D"],
)
@triton.jit
def gemm_k_device(
    A: tl.tensor,
    B: tl.tensor,
    D: tl.tensor,
    Mask: tl.tensor, # [NG, M]
    Index: tl.tensor, # [NG, M]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SplitK: tl.constexpr,
    L2Group: tl.constexpr,
):
    bidx, bidy, bidz = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gdimx, gdimy = tl.num_programs(0), tl.num_programs(1)
    bidx, bidy = tl.swizzle2d(bidx, bidy, gdimx, gdimy, L2Group)

    NG: tl.constexpr = K // G
    G_iter: tl.constexpr = G // BK

    k_size_per_group = K // SplitK
    iter_in_group = k_size_per_group // (BK * G_iter)

    for iter_k in tl.range(0, iter_in_group, warp_specialize=False):
        k_start = k_size_per_group * bidz + iter_k * G_iter * BK
        k_iter_id = bidz * iter_in_group + iter_k

        acc = tl.zeros((BM, BN), dtype=tl.float32)

        rMask = tl.load(
            Mask + k_iter_id * M + bidx * BM + tl.arange(0, BM),
            mask=bidx * BM + tl.arange(0, BM) < M,
            other=0,
        )
        execute_cta = tl.reduce_or(rMask, axis=-1)
        
        if execute_cta == 1:
            rIndex = tl.load(
                Index + k_iter_id * M + bidx * BM + tl.arange(0, BM),
                mask=rMask > 0,
                other=0,
            )
        
            bm_offset = rIndex * K
            bn_offset = bidy * BN + tl.arange(0, BN)
            bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BK), BK)
            bk_offset = k_start + tl.arange(0, BK)[None, :]
            for i in tl.range(0, G_iter):
                sA = tl.load(A + bm_offset[:, None] + bk_offset)
                sB = tl.load(B + bn_offset[:, None] * K + bk_offset)

                acc = tl.dot(sA, sB.T, acc=acc)
                bk_offset += BK
            
            tl.atomic_add(
                D + rIndex[:, None] * N + bn_offset[None, :],
                acc.to(A.dtype.element_ty),
                mask=(rMask > 0)[:, None],
                sem='relaxed',
            )


@triton.autotune(
    configs=gemv_configs,
    key=["M", "N", "K", "NG"],
    prune_configs_by={"early_config_prune": prune_gemv_configs},
    restore_value=["D"],
)
@triton.jit
def gemv_k_device(
    A: tl.tensor,
    B: tl.tensor,
    D: tl.tensor,
    Mask: tl.tensor, # [M, NG]
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SplitK: tl.constexpr,
    L2Group: tl.constexpr,
):
    bidx, bidy, bidz = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gdimx, gdimy = tl.num_programs(0), tl.num_programs(1)
    bidx, bidy = tl.swizzle2d(bidx, bidy, gdimx, gdimy, L2Group)

    NG: tl.constexpr = K // G
    G_iter: tl.constexpr = G // BK

    k_size_per_group = K // SplitK
    iter_in_group = k_size_per_group // (BK * G_iter)

    for iter_k in tl.range(0, iter_in_group, warp_specialize=False):
        k_start = k_size_per_group * bidz + iter_k * G_iter * BK
        k_iter_id = bidz * iter_in_group + iter_k

        acc = tl.zeros((BN,), dtype=tl.float32)
        execute_cta = tl.load(Mask + bidx * NG + k_iter_id)
        
        if execute_cta == 1:
            bn_offset = bidy * BN + tl.arange(0, BN)
            bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BK), BK)
            bk_offset = k_start + tl.arange(0, BK)
            for i in tl.range(0, G_iter):
                sA = tl.load(A + bidx * K + bk_offset)
                sB = tl.load(B + bn_offset[:, None] * K + bk_offset[None, :])
                acc += tl.sum((sA[None, :] * sB).to(tl.float32), axis=-1)
                bk_offset += BK
            
            tl.atomic_add(
                D + bidx * N + bn_offset,
                acc.to(A.dtype.element_ty),
                sem='relaxed',
            )


def gemm_k_host(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor]=None,
    estimate_sparsity: Optional[float]=0.5,
    **kwargs,
):
    assert A.dim() == 3 and B.dim() == 2
    assert Mask.dim() in (2, 3)
    assert estimate_sparsity <= 1
    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity

    Bsz, T, K = A.shape
    A_f = A.flatten(0, 1)

    M, _ = A_f.shape
    N, _ = B.shape
    if D is None: D = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    if Mask.dim() == 2: Mask = Mask.unsqueeze(-1) # [B, T, NG]

    if M > 1:
        Mask_t = Mask.flatten(0, 1).transpose(0, 1).contiguous()
        Mask_st, Index = torch.sort(Mask_t, dim=-1, descending=True, stable=False)
        Mask_st = Mask_st.to(torch.uint8)
        Index = Index.to(torch.uint32)
        NG = Mask_st.shape[0]
        G = K // NG

        grid = lambda meta: (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']), meta['SplitK'])

        gemm_k_device[grid](
            A_f, B, D, Mask_st, Index,
            M, N, K, G,
            BK=32,
            L2Group=4,
        )
    else: # gemv
        Mask_t = Mask.flatten(0, 1).to(torch.uint8)
        NG = Mask_t.shape[1]
        G = K // NG
        
        grid = lambda meta: (M, triton.cdiv(N, meta['BN']), meta['SplitK'])
        gemv_k_device[grid](
            A_f, B, D, Mask_t,
            M, N, K, G,
            BK=32,
            L2Group=4,
        )
        
    return D.reshape(Bsz, T, N)
