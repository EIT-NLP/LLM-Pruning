import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

attention_configs = [
    triton.Config(
        {'BM': bm, "BN": bn},
        num_warps=4,
        num_stages=pipeline,
    )
    for bm in [16, 32, 64, 128]
    for bn in [16, 32, 64, 128]
    for pipeline in [1, 2]
]

def prune_attention_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    M = args["Tq"]
    N = args["Tk"]

    max_bm = min(128, max(16, triton.next_power_of_2(int(M))))
    max_bn = min(128, max(16, triton.next_power_of_2(int(N))))
    bm_space = [max_bm, max(16, max_bm // 2), max(16, max_bm // 4)]
    bn_space = [max_bn, max(16, max_bn // 2), max(16, max_bn // 4)]

    return [
        cfg for cfg in configs
        if cfg.kwargs['BM'] in bm_space and cfg.kwargs['BN'] in bn_space
    ]

@triton.autotune(
    configs=attention_configs,
    key=["B", "Tq", "Tk", "Hq", "Hk", "D", "NG"],
    prune_configs_by={"early_config_prune": prune_attention_configs},
    restore_value=["O"],
)
@triton.jit
def attention_head_device(
    Q: tl.tensor,
    K: tl.tensor,
    V: tl.tensor,
    O: tl.tensor,
    Mask: tl.tensor, # [B, Tq, NG]
    Index: tl.tensor, # [B, Tq, NG]
    Leftpad: tl.tensor, # [B,]
    B: tl.constexpr,
    Tq: tl.constexpr,
    Tk: tl.constexpr,
    Hq: tl.constexpr,
    Hk: tl.constexpr,
    D: tl.constexpr,
    NG: tl.constexpr,
    scale: tl.float32,
    is_causal: tl.constexpr,
    is_leftpad: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr
):
    # grid: [cdiv(Tq, BM), Hq, B]
    bidx, bidy, bidz = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    gdimx = tl.num_programs(0)

    rMax = tl.full([BM], -float('inf'), dtype=tl.float32)
    rSum = tl.zeros([BM], dtype=tl.float32)
    rO = tl.zeros([BM, D], dtype=tl.float32)

    scale *= 1.44269504 # 1/log(2)
    query_head = bidy
    key_head = query_head // (Hq // Hk)
    group_id = query_head // (Hq // NG)

    m_base_off = bidx * BM

    rMask = tl.load(
        Mask + bidz * (Tq * NG) + (m_base_off + tl.arange(0, BM)) * NG + group_id,
        mask=(m_base_off + tl.arange(0, BM)) < Tq,
        other=0,
    )
    execute_cta = tl.reduce_or(rMask, axis=-1)

    if execute_cta == 1:
        b_base_off_q = bidz * (Tq * Hq * D)
        b_base_off_k = bidz * (Tk * Hk * D)
        h_base_off_q = query_head * D
        h_base_off_k = key_head * D

        mask_q = rMask > 0
        rIndex = tl.load(
            Index + bidz * (Tq * NG) + (m_base_off + tl.arange(0, BM)) * NG + group_id,
            mask=mask_q,
            other=0,
        )
        sQ = tl.load(
            Q + b_base_off_q + rIndex[:, None] * (Hq * D) + h_base_off_q + tl.arange(0, D)[None, :],
            mask=mask_q[:, None],
            other=0,
        )

        kv_start = tl.load(Leftpad + bidz) if is_leftpad else 0
        kv_end = tl.minimum(Tk, tl.max(rIndex) + 1) if is_causal else Tk
        for kv_pos in tl.range(kv_start, kv_end, BN):
            mask_kv = kv_pos + tl.arange(0, BN) < Tk
            sK = tl.load(
                K + b_base_off_k + (kv_pos + tl.arange(0, BN)[:, None]) * (Hk * D) + h_base_off_k + tl.arange(0, D)[None, :],
                mask=mask_kv[:, None],
                other=0,
            )
            sV = tl.load(
                V + b_base_off_k + (kv_pos + tl.arange(0, BN)[:, None]) * (Hk * D) + h_base_off_k + tl.arange(0, D)[None, :],
                mask=mask_kv[:, None],
                other=0,
            )

            rP = tl.dot(sQ, sK.T) * scale
            rP = tl.where(mask_kv[None, :], rP, -float('inf'))
            if is_causal:
                rP = tl.where(
                    rIndex[:, None] >= (kv_pos + tl.arange(0, BN))[None, :],
                    rP, -float('inf')
                )
            
            rMax_new = tl.maximum(rMax, tl.max(rP, axis=-1))
            rScale = tl.exp2(rMax - rMax_new)
            rP = tl.exp2(rP - rMax_new[:, None])
            rSum = rSum * rScale + tl.sum(rP, axis=-1)
            rO = rO * rScale[:, None] + tl.dot(rP.to(Q.dtype.element_ty), sV)
            rMax = rMax_new

        rO /= rSum[:, None]
        tl.store(
            O + b_base_off_q + rIndex[:, None] * (Hq * D) + h_base_off_q + tl.arange(0, D)[None, :],
            rO.to(Q.dtype.element_ty),
            mask=mask_q[:, None],
        )

def attention_head_host(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    Leftpad: Optional[torch.Tensor]=None,
    O: Optional[torch.Tensor]=None,
    scale: Optional[float]=None,
    is_causal: Optional[bool]=True,
    estimate_sparsity: Optional[float]=0.5,
    **kwargs,
):
    assert Q.dim() == 4 and K.dim() ==4 and V.dim() == 4
    assert K.shape == V.shape
    assert Mask.dim() == 3
    assert estimate_sparsity <= 1
    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity

    B, Tq, Hq, D = Q.shape
    _, Tk, Hk, _ = K.shape
    _, _, NG = Mask.shape

    if O is None: O = torch.zeros((B, Tq, Hq, D), dtype=Q.dtype, device=Q.device)

    is_leftpad = True
    if Leftpad is None:
        Leftpad = torch.empty((B,), dtype=Q.dtype, device=Q.device)
        is_leftpad = False

    if scale is None: scale = D ** -0.5

    Mask_s, Index = torch.sort(Mask, dim=1, descending=True, stable=False)
    Mask_s = Mask_s.to(torch.uint8)
    Index = Index.to(torch.uint32)

    grid = lambda meta: (triton.cdiv(Tq, meta['BM']), Hq, B)
    attention_head_device[grid](
        Q, K, V, O, Mask_s, Index, Leftpad,
        B, Tq, Tk, Hq, Hk, D, NG,
        scale, is_causal, is_leftpad,
    )
    return O
