import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any
from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn

##############################################################
#                        BLASST
##############################################################

def blasst_prefill_impl(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    pad_offset: tl.tensor, # [B]
    execute_block: tl.tensor, # Optional [LK // BLOCK_N]
    out: tl.tensor,
    LQ: tl.int64,
    LK: tl.int64,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    qk_scale: tl.float32,
    threshold: tl.float32,
    predefined_skip: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    score_scale = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale *= 1.44269504 # 1/log(2)
    key_head_id = query_head_id // G

    query_batch_offset = batch_id * LQ * HQ * D
    query_seq_range = query_id * BLOCK_M + tl.arange(0, BLOCK_M)
    query_head_offset = query_head_id * D

    key_batch_offset = batch_id * LK * HK * D
    key_head_offset = key_head_id * D

    pad_offset_kv = tl.load(pad_offset + batch_id)
    key_length = LK - pad_offset_kv

    query_range_mask = query_seq_range < LQ
    query_data = tl.load(
        q + query_batch_offset + query_seq_range[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0,
    )

    num_kv_iter = tl.cdiv((query_id + 1) * BLOCK_M - pad_offset_kv + 1, BLOCK_N)
    for tile_kv in tl.range(0, num_kv_iter):
        is_greater_than_threshold = 1
        if predefined_skip:
            need_execute = tl.load(execute_block + tile_kv)
            is_greater_than_threshold &= (need_execute == 1)

        key_seq_range = tile_kv * BLOCK_N + tl.arange(0, BLOCK_N) + pad_offset_kv
        key_range_mask = key_seq_range < key_length
        key_data = tl.load(
            k + (key_batch_offset + key_seq_range[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
            mask=key_range_mask[:, None],
            other=0.0,
        )

        qk = tl.dot(query_data, key_data.T) * qk_scale
        # causal mask for each query pos
        causal_mask = query_seq_range[:, None] >= key_seq_range[None, :]
        qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

        score_local_max = tl.max(qk, 1)
        score_max_new = tl.maximum(score_max, score_local_max)
        is_greater_than_threshold &= tl.reduce_or(tl.exp2(score_max_new - score_local_max) * score_scale > threshold, axis=0)

        if is_greater_than_threshold: # continue load V on smem and compute PV
            value_data = tl.load(
                v + (key_batch_offset + key_seq_range[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                mask=key_range_mask[:, None],
                other=0.0,
            )

            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        
    acc /= score_sum[:, None]
    tl.store(
        out + query_batch_offset + query_seq_range[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :],
        acc.to(q.dtype.element_ty),
        mask=query_range_mask[:, None],
    )

##############################################################
#                     Seer Attention
##############################################################
def seer_prefill_impl(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    pad_offset: tl.tensor, # [B]
    execute_block: tl.tensor, # [B, HK, cdiv(LQ, BLOCK_M), cdiv(LK, BLOCK_N)]
    out: tl.tensor,
    LQ: tl.int64,
    LK: tl.int64,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    qk_scale: tl.float32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    query_id, query_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query_block_num, key_block_num = tl.cdiv(LQ, BLOCK_M), tl.cdiv(LK, BLOCK_N)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    score_scale = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    qk_scale *= 1.44269504 # 1/log(2)
    key_head_id = query_head_id // G

    query_batch_offset = batch_id * LQ * HQ * D
    query_seq_range = query_id * BLOCK_M + tl.arange(0, BLOCK_M)
    query_head_offset = query_head_id * D

    key_batch_offset = batch_id * LK * HK * D
    key_head_offset = key_head_id * D

    pad_offset_kv = tl.load(pad_offset + batch_id)
    key_length = LK - pad_offset_kv

    mask_offset = (batch_id * HK + key_head_id) * query_block_num * key_block_num + query_id * key_block_num

    query_range_mask = query_seq_range < LQ
    query_data = tl.load(
        q + query_batch_offset + query_seq_range[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0,
    )

    num_kv_iter = tl.cdiv((query_id + 1) * BLOCK_M - pad_offset_kv + 1, BLOCK_N)
    for tile_kv in tl.range(0, num_kv_iter):
        is_execute = tl.load(execute_block + mask_offset + (tile_kv * BLOCK_N + pad_offset_kv) // BLOCK_N)
        if is_execute:
            key_seq_range = tile_kv * BLOCK_N + tl.arange(0, BLOCK_N) + pad_offset_kv
            key_range_mask = key_seq_range < key_length
            key_data = tl.load(
                k + (key_batch_offset + key_seq_range[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                mask=key_range_mask[:, None],
                other=0.0,
            )

            qk = tl.dot(query_data, key_data.T) * qk_scale
            # causal mask for each query pos
            causal_mask = query_seq_range[:, None] >= key_seq_range[None, :]
            qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))

            score_local_max = tl.max(qk, 1)
            score_max_new = tl.maximum(score_max, score_local_max)

            value_data = tl.load(
                v + (key_batch_offset + key_seq_range[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]),
                mask=key_range_mask[:, None],
                other=0.0,
            )

            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        
    acc /= score_sum[:, None]
    tl.store(
        out + query_batch_offset + query_seq_range[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :],
        acc.to(q.dtype.element_ty),
        mask=query_range_mask[:, None],
    )
    

class SparseAttentionPrefill:

    support_kernel = [
        'blasst',
        'seer',
    ]

    @classmethod
    def _blasst_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        threshold: float,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):  
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.empty_like(q)
        grid = lambda meta: (triton.cdiv(LQ, meta['BLOCK_M']), HQ, B)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            blasst_prefill_impl,
            enable_autotune=True,
            config=config,
            keys=['LQ', 'LK', 'HQ', 'HK', 'D'],
            do_not_specialize=['qk_scale', 'threshold']
        )
        kernel[grid](
            q, k, v, pad_offset, execute_block, out,
            LQ, LK, HQ, HK, D, G, D**-0.5, threshold,
            predefined_skip=execute_block is not None,
        )
        return out
    
    @classmethod
    def _seer_kernel(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):  
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        # compute qk score
        q_block = rearrange(q, 'b (nk k) h d -> b h nk k d', k=BLOCK_M).contiguous().mean(-2) # [b h lq // BLOCK_M, d]
        q_block = rearrange(q_block, 'b (h g) k d -> b h g k d', g=G).mean(2)
        k_block = rearrange(k, 'b (nk k) h d -> b h nk k d', k=BLOCK_N).contiguous().mean(-2) # [b h lk // BLOCK_N, d]
        qk_score = q_block @ (k_block.transpose(-1, -2)) # [B, HK, LQ', LK']
        qk_score = torch.softmax(qk_score * D**-0.5, dim=-1)

        out = torch.empty_like(q)
        grid = lambda meta: (triton.cdiv(LQ, meta['BLOCK_M']), HQ, B)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'num_stages'],
            BLOCK_M=BLOCK_N, # equal to BLOCK_N
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            seer_prefill_impl,
            enable_autotune=True,
            config=config,
            keys=['LQ', 'LK', 'HQ', 'HK', 'D'],
            do_not_specialize=['qk_scale']
        )
        kernel[grid](
            q, k, v, pad_offset, execute_block, out,
            LQ, LK, HQ, HK, D, G, D**-0.5,
        )
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for sparse attention (prefill)\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q, k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            skip_block: Optional[torch.Tensor] with shape [LK // 16], predefined skip decision for each kv block
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        q = kwargs.get('q')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 2
        num_warps = 4

        BLOCK_M = triton.next_power_of_2(LQ)
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LQ)))
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        block_size = kwargs.pop('block_size', -1)
        if block_size > 0:
            BLOCK_N = block_size

        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_M > 32: BLOCK_M >>= 1
            elif num_stages > 2: num_stages -= 1
            else: break
            
        while HQ * B * triton.cdiv(LQ, BLOCK_M) < num_sm:
            if BLOCK_M > 16: BLOCK_M >>= 1
            else: break
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            num_warps=num_warps,
            pad_offset=pad_offset,
            BLOCK_M_list=[16, 32, 64],
            num_stages_list=[2, 3],
            **kwargs
        )