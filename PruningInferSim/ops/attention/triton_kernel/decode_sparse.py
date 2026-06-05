import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any
from ops.utils import get_autotune_config, get_autotune_cache

##############################################################
#                        BLASST
##############################################################

def blasst_decode_impl(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    pad_offset: tl.tensor, # [B]
    execute_block: tl.tensor, # Optional [LK // BLOCK_N]
    out: tl.tensor, # [B, 1, HQ, D] or [B, 1, HQ, splitk, D]
    metadata: tl.tensor, # Optional [B, 1, HQ, 2, splitk]
    LK: tl.int64,
    LBK: tl.int64, # size of each split k block, set to 0 if no split
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    NBK: tl.int64, # num of split k for each seq
    qk_scale: tl.float32,
    threshold: tl.float32,
    kv_split: tl.constexpr, # whether to using flash decoding
    predefined_skip: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_id, key_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qk_scale *= 1.44269504 # 1/log(2)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    score_scale = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    start_q, end_q = batch_id, batch_id + 1
    seq_offset_q = start_q

    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k = batch_id * LK + split_id * LBK + pad_offset_k
    end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
    key_length = end_k - start_k

    query_range_mask = tl.arange(0, BLOCK_M) < G
    query_data = tl.load(
        q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0
    )

    split_n_range = tl.cdiv(key_length, BLOCK_N)
    for tile_n in tl.range(0, split_n_range):
        is_greater_than_threshold = 1
        if predefined_skip:
            need_execute = tl.load(execute_block + tile_n)
            is_greater_than_threshold &= (need_execute == 1)
        
        key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
        key_data = tl.load(
            k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
            mask=key_range_mask[:, None],
            other=0.0
        )

        # qk size [BLOCK_M, BLOCK_N]
        qk = tl.dot(query_data, key_data.T) * qk_scale
        qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

        # scalar
        score_local_max = tl.max(qk, 1)
        score_max_new = tl.maximum(score_max, score_local_max)
        is_greater_than_threshold &= tl.reduce_or(tl.exp2(score_max_new - score_local_max) * score_scale > threshold, axis=0)
        
        if is_greater_than_threshold:
            value_data = tl.load(
                v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=key_range_mask[:, None],
                other=0.0
            )

            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        
    if not kv_split: acc /= score_sum[:, None]
    
    tl.store(
        out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :],
        acc.to(out.dtype.element_ty),
        mask=query_range_mask[:, None],
    )
    
    if kv_split:
        tl.store(
            metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id,
            score_max.to(metadata.dtype.element_ty),
            mask=query_range_mask,
        )
        tl.store(
            metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id,
            score_sum.to(metadata.dtype.element_ty),
            mask=query_range_mask,
        )

##############################################################
#                     Seer Attention
##############################################################
def seer_decode_impl(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    pad_offset: tl.tensor, # [B]
    execute_block: tl.tensor, # [B, HK, cdiv(LQ, BLOCK_M), cdiv(LK, BLOCK_N)]
    out: tl.tensor, # [B, 1, HQ, D] or [B, 1, HQ, splitk, D]
    metadata: tl.tensor, # Optional [B, 1, HQ, 2, splitk]
    LK: tl.int64,
    LBK: tl.int64, # size of each split k block, set to 0 if no split
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    NBK: tl.int64, # num of split k for each seq
    qk_scale: tl.float32,
    kv_split: tl.constexpr, # whether to using flash decoding
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    split_id, key_head_id, batch_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    query_block_num, key_block_num = 1, tl.cdiv(LK, BLOCK_N)
    qk_scale *= 1.44269504 # 1/log(2)

    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    score_scale = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

    start_q, end_q = batch_id, batch_id + 1
    seq_offset_q = start_q

    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k = batch_id * LK + split_id * LBK + pad_offset_k
    end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
    key_length = end_k - start_k

    mask_offset = (batch_id * HK + key_head_id) * query_block_num * key_block_num

    query_range_mask = tl.arange(0, BLOCK_M) < G
    query_data = tl.load(
        q + ((seq_offset_q * HQ * D) + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :],
        mask=query_range_mask[:, None],
        other=0.0
    )

    split_n_range = tl.cdiv(key_length, BLOCK_N)
    for tile_n in tl.range(0, split_n_range):
        is_execute = tl.load(execute_block + mask_offset + (start_k + tile_n * BLOCK_N) // BLOCK_N)
        if is_execute:
            key_range_mask = ((tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) < key_length)
            key_data = tl.load(
                k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=key_range_mask[:, None],
                other=0.0
            )

            # qk size [BLOCK_M, BLOCK_N]
            qk = tl.dot(query_data, key_data.T) * qk_scale
            qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))

            # scalar
            score_local_max = tl.max(qk, 1)
            score_max_new = tl.maximum(score_max, score_local_max)
            
            value_data = tl.load(
                v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :],
                mask=key_range_mask[:, None],
                other=0.0
            )

            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        
    if not kv_split: acc /= score_sum[:, None]
    
    tl.store(
        out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :],
        acc.to(out.dtype.element_ty),
        mask=query_range_mask[:, None],
    )
    
    if kv_split:
        tl.store(
            metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id,
            score_max.to(metadata.dtype.element_ty),
            mask=query_range_mask,
        )
        tl.store(
            metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id,
            score_sum.to(metadata.dtype.element_ty),
            mask=query_range_mask,
        )

def merge_impl(
    out_tmp: tl.tensor, # [B, 1, HQ, splitk, D]
    metadata: tl.tensor, # [B, 1, HQ, 2, splitk] for local max and local sum
    out: tl.tensor, 
    HQ: tl.constexpr,
    D: tl.constexpr,
    NBK: tl.int64, # num of split k for each seq
    NBK2: tl.constexpr, # next_pow_of_2(NBK)
):
    query_head_id, batch_id = tl.program_id(0), tl.program_id(1)

    NBK_mask = tl.arange(0, NBK2) < NBK
    split_range = tl.arange(0, NBK2)
    acc = tl.load(
        out_tmp + (batch_id * HQ + query_head_id) * NBK * D + (split_range * D)[:, None] + tl.arange(0, D)[None, :],
        mask=NBK_mask[:, None],
        other=0.0
    )
    score_max = tl.load(
        metadata + (batch_id * HQ + query_head_id) * 2 * NBK + split_range,
        mask=NBK_mask,
        other=-float('inf')
    )
    score_sum = tl.load(
        metadata + (batch_id * HQ + query_head_id) * 2 * NBK + NBK + split_range,
        mask=NBK_mask,
        other=0.0
    )

    global_max = tl.max(score_max)
    global_scale = tl.exp2(score_max - global_max)
    score_sum *= global_scale
    global_sum = tl.sum(score_sum)

    acc = tl.sum(acc * global_scale[:, None], axis=0)
    acc /= global_sum

    tl.store(
        out + (batch_id * HQ + query_head_id) * D + tl.arange(0, D),
        acc.to(out.dtype.element_ty)
    )
    

class SparseAttentionDecode:

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
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        split_size: Optional[int]=0,
        num_split: Optional[int]=1,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.empty_like(q)
        grid = lambda meta: (num_split, HK, B)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            blasst_decode_impl,
            enable_autotune=True,
            config=config,
            keys=['LK', 'LBK', 'HQ', 'HK', 'D'],
            do_not_specialize=['NBK', 'qk_scale', 'threshold']
        )

        if split_size == 0:
            kernel[grid](
                q, k, v, pad_offset, execute_block, out, None,
                LK, split_size, HQ, HK, D, G, 1, D**-0.5, threshold,
                kv_split=False,
                predefined_skip=execute_block is not None,
            )
        
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)

            merge_kernel = get_autotune_cache(
                merge_impl,
                enable_autotune=True,
                config=[],
                keys=[],
                do_not_specialize=['NBK'],
            )

            kernel[grid](
                q, k, v, pad_offset, execute_block, out_local, metadata,
                LK, split_size, HQ, HK, D, G, num_split, D**-0.5, threshold,
                kv_split=True,
                predefined_skip=execute_block is not None,
            )
            grid = lambda meta: (HQ, B)
            merge_kernel[grid](
                out_local, metadata, out,
                HQ, D, num_split, triton.next_power_of_2(num_split),
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
        BLOCK_M: Optional[int]=16,
        BLOCK_N: Optional[int]=32,
        split_size: Optional[int]=0,
        num_split: Optional[int]=1,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LK', 'qk_scale'],
        **kwargs
    ):
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        # compute qk score
        # q_block = rearrange(q, 'b k (h g) d -> b h g k d', g=G).contiguous().mean(2)
        # k_block = rearrange(k, 'b (nk k) h d -> b h nk k d', k=BLOCK_N).contiguous().mean(-2) # [b h lk // BLOCK_N, d]
        # qk_score = q_block.permute(0, 2, 1, 3) @ (k_block.permute(0, 2, 3, 1)) # [B, HK, LQ', LK']
        # qk_score = torch.softmax(qk_score * D**-0.5, dim=-1)

        out = torch.empty_like(q)
        grid = lambda meta: (num_split, HK, B)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'num_stages'],
            BLOCK_M=BLOCK_N, # same with BLOCK_N
            BLOCK_N=BLOCK_N,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            seer_decode_impl,
            enable_autotune=True,
            config=config,
            keys=['LK', 'LBK', 'HQ', 'HK', 'D'],
            do_not_specialize=['NBK', 'qk_scale', 'threshold']
        )

        if split_size == 0:
            kernel[grid](
                q, k, v, pad_offset, execute_block, out, None,
                LK, split_size, HQ, HK, D, G, 1, D**-0.5,
                kv_split=False,
            )
        
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)

            merge_kernel = get_autotune_cache(
                merge_impl,
                enable_autotune=True,
                config=[],
                keys=[],
                do_not_specialize=['NBK'],
            )

            kernel[grid](
                q, k, v, pad_offset, execute_block, out_local, metadata,
                LK, split_size, HQ, HK, D, G, num_split, D**-0.5,
                kv_split=True,
            )
            grid = lambda meta: (HQ, B)
            merge_kernel[grid](
                out_local, metadata, out,
                HQ, D, num_split, triton.next_power_of_2(num_split),
            )
        
        return out
    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for sparse attention (decode)\\
        Return [batch_size, query_length, num_query_heads, head_dim]
        Args:
            q, k, v: torch.Tensor with shape [batch_size, seqlen, num_heads, head_dim]
            pad_offset: Optional[torch.Tensor] with shape [batch_size,], left padding offset for key and value
            skip_block: Optional[torch.Tensor] with shape [LK // 16], predefined skip decision for each kv block
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]

        if impl == 'auto': impl = 'split'

        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK

        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None: pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)

        BLOCK_M = max(16, triton.next_power_of_2(G))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
        BLOCK_N = kwargs.pop('block_size', BLOCK_N)

        block_size = kwargs.pop('block_size', -1)
        if block_size > 0:
            BLOCK_N = block_size

        num_stages = 2
        num_warps = 4

        num_split = 1
        while num_split * B * HK < num_sm * 2 and int(LK / num_split) > max(BLOCK_N, 64):
            num_split += 1
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        split_size = triton.cdiv(LK, num_split) if num_split > 1 else 0
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            split_size=split_size,
            num_split=num_split,
            num_stages=num_stages,
            num_warps=num_warps,
            pad_offset=pad_offset,
            num_stages_list=[2, 3],
            **kwargs
        )