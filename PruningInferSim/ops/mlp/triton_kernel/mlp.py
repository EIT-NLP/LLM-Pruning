import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange

from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_gemm

#########################
# Kernel Implementation
#########################
def bm_sort_mlp_gemv_impl(
    x: tl.tensor, # [M, K]
    route_mask: tl.tensor, # [M]
    w: tl.tensor, # [N, K]
    b: tl.tensor,
    out: tl.tensor,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    tmid, tnid, kid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

    # compute indices in groups
    # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
    mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    query_mask = tl.load(route_mask + mid)
    if query_mask != 0:
        split_k_size = tl.cdiv(K, SPLIT_K)
        bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
        bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
        bk_offset = kid * BLOCK_K + tl.arange(0, BLOCK_K)
        for i in tl.range(0, tl.cdiv(split_k_size, BLOCK_K)):
            x_data = tl.load(
                x + mid * K + bk_offset,
                mask=bk_offset < K,
                other=0,
            )
            w_data = tl.load(
                w + bn_offset[:, None] * K + bk_offset[None, :],
                mask=bk_offset[None, :] < K,
                other=0,
            )

            acc += tl.sum((x_data[None, :] * w_data).to(tl.float32), axis=1)
            bk_offset += BLOCK_K * SPLIT_K
        
        if SPLIT_K == 1:
            if HAS_BIAS:
                acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))
            
            if ACTIVATION == 'silu':
                acc *= tl.sigmoid(acc)
            if ACTIVATION == 'relu':
                acc = tl.maximum(acc, 0)
            
            tl.store(
                out + mid * N + bn_offset,
                acc.to(x.dtype.element_ty),
                mask=(bn_offset < N),
            )
        else:
            tl.atomic_add(
                out + mid * N + bn_offset,
                acc.to(x.dtype.element_ty),
                mask=(bn_offset < N),
                sem='relaxed',
            )


def bm_sort_mlp_impl(
    x: tl.tensor, # [M, K]
    route_mask: tl.tensor, # [M] or [cdiv(M, BLOCK_M)]
    route_indices: tl.tensor, # [M]
    w: tl.tensor, # [N, K]
    b: tl.tensor,
    out: tl.tensor,
    M: tl.int64,
    N: tl.int64,
    K: tl.int64,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    IS_OFFLINE: tl.constexpr,
):
    tmid, tnid, kid = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    BLOCK_NUM_M, BLOCK_NUM_N = tl.num_programs(0), tl.num_programs(1)

    # compute indices in groups
    # Group 1: B[0:BN], B[BN:2BN], ... B[(G-1)*BN:G*BN] -> A[0:BM]
    mid, nid = tl.swizzle2d(tmid, tnid, BLOCK_NUM_M, BLOCK_NUM_N, GROUP_SIZE)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if IS_OFFLINE:
        skip_flag = tl.load(route_mask + mid)
    else:
        query_mask = tl.load(
            route_mask + mid * BLOCK_M + tl.arange(0, BLOCK_M),
            mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
            other=0,
        )
        skip_flag = tl.reduce_or(query_mask, axis=-1)

    if skip_flag != 0:
        if IS_OFFLINE:
            bm_indices = tl.load(
                route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=mid * BLOCK_M + tl.arange(0, BLOCK_M) < M,
                other=-1,
            )
            bm_mask = bm_indices >= 0
            bm_indices = tl.where(bm_mask, bm_indices, 0)
        else:
            bm_mask = query_mask > 0
            bm_indices = tl.load(
                route_indices + mid * BLOCK_M + tl.arange(0, BLOCK_M),
                mask=bm_mask,
                other=0,
            )

        split_k_size = tl.cdiv(K, SPLIT_K)

        bm_offset = bm_indices * K
        bn_offset = nid * BLOCK_N + tl.arange(0, BLOCK_N)
        bn_offset = tl.max_contiguous(tl.multiple_of(bn_offset, BLOCK_N), BLOCK_N)
        bk_offset = kid * BLOCK_K + tl.arange(0, BLOCK_K)[None, :]
        for i in tl.range(0, tl.cdiv(split_k_size, BLOCK_K)):
            x_data = tl.load(
                x + bm_offset[:, None] + bk_offset,
                mask=bk_offset < K,
                other=0,
            )
            w_data = tl.load(
                w + bn_offset[:, None] * K + bk_offset,
                mask=bk_offset < K,
                other=0,
            )

            acc = tl.dot(x_data, w_data.T, acc=acc)
            bk_offset += BLOCK_K * SPLIT_K
        
        if SPLIT_K == 1:
            if HAS_BIAS:
                acc += (tl.load(b + bn_offset, mask=bn_offset < N, other=0).to(tl.float32))[None, :]
            
            if ACTIVATION == 'silu':
                acc *= tl.sigmoid(acc)
            if ACTIVATION == 'relu':
                acc = tl.maximum(acc, 0)
            
            tl.store(
                out + bm_indices[:, None] * N + bn_offset[None, :],
                acc.to(x.dtype.element_ty),
                mask=bm_mask[:, None] & (bn_offset < N)[None, :],
            )
        else:
            tl.atomic_add(
                out + bm_indices[:, None] * N + bn_offset[None, :],
                acc.to(x.dtype.element_ty),
                mask=bm_mask[:, None] & (bn_offset < N)[None, :],
                sem='relaxed',
            )

#########################
# Host Implementation
#########################
class BMSparseMLP:

    support_kernel = [
        'dense',
        'indexing',
        'sort_online',
        'sort_offline',
    ]

    @classmethod
    def _dense_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        out = torch.nn.functional.linear(x, w, b).masked_fill(route_mask.logical_not()[:, :, None], 0)
        return out
    
    @classmethod
    def _indexing_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        **kwargs
    ):
        indices = kwargs.get('indices', torch.nonzero(route_mask.flatten()).flatten())
        res = torch.zeros((x.shape[0] * x.shape[1], w.shape[0]), dtype=x.dtype, device=x.device)
        tmp = x.flatten(0, 1).index_select(dim=0, index=indices)
        out = torch.nn.functional.linear(tmp, w, b)
        res[indices] = out
        return rearrange(res, '(b l) d -> b l d', b=x.shape[0]), dict(
            indices=indices,
        )

    @classmethod
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        b: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        SPLIT_K: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
        use_gemv: Optional[bool]=False,
        num_stages: Optional[int]=3,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['M'],
        **kwargs
    ):
        B, L, D = x.shape

        M = B * L
        N = w.shape[0]
        K = D

        x_flat = x.reshape((M, D))
        m_sort = kwargs.get('m_sort', None)
        m_sort_indices = kwargs.get('m_sort_indices', None)
        if not use_gemv:
            if m_sort_indices is None:
                m_flat = route_mask.flatten(0, 1)
                m_sort, m_sort_indices = torch.sort(m_flat, descending=True, stable=False)

                if kwargs.get('is_offline', False):
                    # offline calculate skipping
                    m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
                    m_sort = torch.nn.functional.pad(m_sort, (0, BLOCK_M - M % BLOCK_M), value=0).reshape(-1, BLOCK_M)
                    m_sort = m_sort.any(dim=-1)
            
            out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']), meta['SPLIT_K'])
            config = get_autotune_config(
                params=['BLOCK_N', 'BLOCK_K', 'SPLIT_K', 'GROUP_SIZE', 'num_stages'],
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                SPLIT_K=SPLIT_K,
                GROUP_SIZE=GROUP_SIZE,
                num_stages=num_stages,
                **kwargs,
            )
            kernel = get_autotune_cache(
                bm_sort_mlp_impl,
                enable_autotune=True,
                config=config,
                keys=['M', 'N', 'K'],
            )
            kernel[grid](
                x_flat, m_sort, m_sort_indices,
                w, b, out,
                M, N, K,
                BLOCK_M=BLOCK_M,
                HAS_BIAS=b is not None,
                ACTIVATION=kwargs.get('activation', 'identity'),
                IS_OFFLINE=kwargs.get('is_offline', False),
            )
        else:
            m_flat = route_mask.flatten(0, 1)
            out = torch.zeros((M, N), dtype=x.dtype, device=x.device)
            grid = lambda meta: (M, triton.cdiv(N, meta['BLOCK_N']), meta['SPLIT_K'])
            config = get_autotune_config(
                params=['BLOCK_N', 'BLOCK_K', 'SPLIT_K', 'GROUP_SIZE', 'num_stages'],
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                SPLIT_K=SPLIT_K,
                GROUP_SIZE=GROUP_SIZE,
                num_stages=num_stages,
                **kwargs,
            )
            kernel = get_autotune_cache(
                bm_sort_mlp_gemv_impl,
                enable_autotune=True,
                config=config,
                keys=['M', 'N', 'K'],
            )
            kernel[grid](
                x_flat, m_flat,
                w, b, out,
                M, N, K,
                HAS_BIAS=b is not None,
                ACTIVATION=kwargs.get('activation', 'identity'),
            )

        return rearrange(out, '(B L) N -> B L N', B=B, N=N), dict(
            m_sort=m_sort,
            m_sort_indices=m_sort_indices,
        )

    
    @classmethod
    def kernel(
        cls,
        impl: str,
        **kwargs
    ):
        """
        Get kernel for token sparse MLP, where x = [M, K], weight = [N, K]\\
        **sort:**\\
        Return the same shape as input, with additional sort prologue for better skipping
        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            route_mask: torch.Tensor with shape [batch_size, seqlen], 1 for active token
            w: torch.Tensor with shape [intermediate_size, hidden_size], weight
            b: torch.Tensor with shape [hidden_size], bias
            sorted_mask: optional, torch.Tensor with shape [batch_size, seqlen], 1 for active token, sorted
            sorted_indices: optinal, torch.Tensor with shape [batch_size, seqlen], index of sorted route_mask
            estimated_sparsity: optinal, float, estimated sparsity of route_mask
            impl: str, impl of kernel
            do_not_specialize: List, omit kernel re-compile for these variables
        """        
        x = kwargs.get('x')
        w = kwargs.get('w')
        dtype = x.dtype
        device = x.device

        B, L, D = x.shape
        M, N, K = B * L, w.shape[0], D
        route_mask = kwargs.pop('route_mask', None)
        
        # get tiling
        num_sm = torch.cuda.get_device_properties("cuda").multi_processor_count
        cc = torch.cuda.get_device_capability("cuda")[0]
        num_stages = 3
        num_warps = 4
        group_size = 4

        estimated_sparsity = kwargs.pop('estimated_sparsity', 1)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        
        if estimated_sparsity == 0: estimated_sparsity = 1
        
        if impl == 'auto': # auto dispatch
            if route_mask is None: impl = 'dense'
            else: impl = 'sort_offline'

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M >= int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(64, triton.next_power_of_2(N)))
            BLOCK_K = min(64, max(32, triton.next_power_of_2(K)))

            SPLIT_K = 1
            SPLIT_K_list = [1]
            use_gemv = M == 1

            while not check_shared_memory_gemm(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, dtype.itemsize):
                if BLOCK_K > 32: BLOCK_K >>= 1
                elif BLOCK_N > 64: BLOCK_N >>= 1
                elif BLOCK_M > 32: BLOCK_M >>= 1
                elif num_stages > 2: num_stages -= 1
                else: break
            
            # split-k active
            if triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) < num_sm and not use_gemv:
                SPLIT_K_list = [2, 3, 4, 5, 6]
                min_waste = 1.0
                best_split_k = 2
                for split_k in SPLIT_K_list:
                    waste = float(num_sm - ((triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * split_k) % num_sm)) / float(num_sm)
                    if (min_waste > 0 and waste < min_waste):
                        min_waste = waste
                        best_split_k = split_k
                
                SPLIT_K = best_split_k
            
            elif M * triton.cdiv(N, BLOCK_N) < num_sm and use_gemv:
                SPLIT_K_list = [2, 3, 4, 5, 6]
                min_waste = 1.0
                best_split_k = 2
                for split_k in SPLIT_K_list:
                    waste = float(num_sm - ((M * triton.cdiv(N, BLOCK_N) * split_k) % num_sm)) / float(num_sm)
                    if (min_waste > 0 and waste < min_waste):
                        min_waste = waste
                        best_split_k = split_k
                
                SPLIT_K = best_split_k
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
            BLOCK_K = kwargs.pop('BLOCK_K', BLOCK_K)
            
        else:
            BLOCK_M = -1
            BLOCK_N = -1
            BLOCK_K = -1
            SPLIT_K = -1
            use_gemv = False
            SPLIT_K_list = [1]
        
        if impl not in cls.support_kernel:
            raise ValueError(f"{impl} is not supported")
        
        is_offline = impl == 'sort_offline'
        impl = impl.split('_')[0]
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=SPLIT_K,
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            is_offline=is_offline,
            use_gemv=use_gemv,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[64, 128],
            BLOCK_K_list=[32, 64],
            SPLIT_K_list=SPLIT_K_list,
            num_stages_list=[2, 3],
            **kwargs
        )