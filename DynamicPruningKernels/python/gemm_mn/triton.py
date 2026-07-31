import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

from ..utils import (
    ACTIVATION,
)

gemm_configs = [
    triton.Config(
        {'BM': bm, "BN": bn, "BK": bk, "SplitK": split_k},
        num_warps=4,
        num_stages=3,
    )
    for bm in [16, 32, 64, 128, 256]
    for bn in [16, 32, 64, 128, 256]
    for bk in [32, 64]
    for split_k in [1, 2, 4, 8]
]
gemv_configs = [
    triton.Config(
        {"BN": bn, "BK": bk, "SplitK": split_k},
        num_warps=4,
        num_stages=3,
    )
    for bn in [16, 32, 64, 128, 256]
    for bk in [32, 64]
    for split_k in [1, 2, 4, 8]
]

def prune_gemm_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    M = args["M"]
    N = args["N"]
    K = args["K"]
    G = args["G"]

    max_bm = min(256, max(16, triton.next_power_of_2(int(M))))
    max_bn = min(256, max(16, triton.next_power_of_2(int(G))))
    bm_space = [max_bm, max_bm // 2]
    bn_space = [max_bn, max_bn // 2]

    return [
        cfg for cfg in configs
        if cfg.kwargs['BM'] in bm_space and cfg.kwargs['BN'] in bn_space and cfg.kwargs['BN'] <= G and K % cfg.kwargs['SplitK'] == 0
    ]

def prune_gemv_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    G = args["G"]
    K = args["K"]

    return [
        cfg for cfg in configs
        if cfg.kwargs['BN'] <= G and K % cfg.kwargs['SplitK'] == 0
    ]

@triton.autotune(
    configs=gemm_configs,
    key=['M', 'N', 'K', 'G'],
    prune_configs_by={"early_config_prune": prune_gemm_configs},
    restore_value=['D'],
)
@triton.jit
def gemm_mn_device(
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
    Activation: tl.constexpr,
):
    bidx, bidy, bidz = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gdimx, gdimy = tl.num_programs(0), tl.num_programs(1)
    bidx, bidy = tl.swizzle2d(bidx, bidy, gdimx, gdimy, L2Group)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    NGIter: tl.constexpr = G // BN

    base_off_m = bidx * BM
    base_off_n = bidy * BN + tl.arange(0, BN)
    base_off_k = bidz * BK

    # load CTA-level skip
    rMask = tl.load(
        Mask + (bidy // NGIter) * M + base_off_m + tl.arange(0, BM),
        base_off_m + tl.arange(0, BM) < M, 0
    )
    execute_cta = tl.reduce_or(rMask, axis=-1)

    if execute_cta == 1:
        rIndex = tl.load(
            Index + (bidy // NGIter) * M + base_off_m + tl.arange(0, BM),
            rMask > 0, 0
        )

        split_k_size = tl.cdiv(K, SplitK)
        off_k = base_off_k[None, :] + tl.arange(0, BK)
        for i in tl.range(0, tl.cdiv(split_k_size, BK)):
            sA = tl.load(A + rIndex[:, None] * K + off_k)
            sB = tl.load(B + base_off_n[:, None] * K + off_k)
            acc = tl.dot(sA, sB.T, acc=acc)
            off_k += BK * SplitK
        
        if SplitK == 1:
            if Activation == 1: # relu
                acc = tl.maximum(acc, 0)
            if Activation == 2: # silu
                acc *= tl.sigmoid(acc)
            
            tl.store(
                D + rIndex[:, None] * N + base_off_n[None, :],
                acc.to(A.dtype.element_ty),
                (rMask > 0)[:, None]
            )
        else:
            tl.atomic_add(
                D + rIndex[:, None] * N + base_off_n[None, :],
                acc.to(A.dtype.element_ty),
                (rMask > 0)[:, None],
                sem='relaxed',
            )

@triton.autotune(
    configs=gemv_configs,
    key=['M', 'N', 'K', 'G'],
    prune_configs_by={"early_config_prune": prune_gemv_configs},
    restore_value=['D'],
)
@triton.jit
def gemv_mn_device(
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
    Activation: tl.constexpr,
):
    bidx, bidy, bidz = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gdimx, gdimy = tl.num_programs(0), tl.num_programs(1)
    bidx, bidy = tl.swizzle2d(bidx, bidy, gdimx, gdimy, L2Group)

    acc = tl.zeros((BN,), dtype=tl.float32)
    NG: tl.constexpr = N // G
    NGIter: tl.constexpr = G // BN

    base_off_m = bidx
    base_off_n = bidy * BN + tl.arange(0, BN)
    base_off_k = bidz * BK

    # load CTA-level skip
    execute_cta = tl.load(Mask + base_off_m * NG + (bidy // NGIter))

    if execute_cta == 1:
        split_k_size = tl.cdiv(K, SplitK)
        off_k = base_off_k + tl.arange(0, BK)
        for i in tl.range(0, tl.cdiv(split_k_size, BK)):
            sA = tl.load(A + base_off_m * K + off_k)
            sB = tl.load(B + base_off_n[:, None] * K + off_k[None, :])
            acc += tl.sum((sA[None, :] * sB).to(tl.float32), axis=-1)
            off_k += BK * SplitK
        
        if SplitK == 1:
            if Activation == 1: # relu
                acc = tl.maximum(acc, 0)
            if Activation == 2: # silu
                acc *= tl.sigmoid(acc)
            
            tl.store(
                D + base_off_m * N + base_off_n,
                acc.to(A.dtype.element_ty),
            )
        else:
            tl.atomic_add(
                D + base_off_m * N + base_off_n,
                acc.to(A.dtype.element_ty),
                sem='relaxed',
            )


def gemm_mn_host(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor]=None,
    activation: Optional[str]="identity",
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
        G = N // NG
        
        grid = lambda meta: (triton.cdiv(M, meta['BM']), triton.cdiv(N, meta['BN']), meta['SplitK'])
        gemm_mn_device[grid](
            A_f, B, D, Mask_st, Index,
            M, N, K, G,
            L2Group=4,
            Activation=ACTIVATION.get(activation, 0),
        )
    
    else: # gemv
        Mask_t = Mask.flatten(0, 1).to(torch.uint8)
        NG = Mask_t.shape[1]
        G = N // NG
        
        grid = lambda meta: (M, triton.cdiv(N, meta['BN']), meta['SplitK'])
        gemv_mn_device[grid](
            A_f, B, D, Mask_t,
            M, N, K, G,
            L2Group=4,
            Activation=ACTIVATION.get(activation, 0),
        )

    return D.reshape(Bsz, T, N)