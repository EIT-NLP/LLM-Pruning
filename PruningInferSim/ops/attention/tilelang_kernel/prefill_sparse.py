import torch
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange
from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn

##############################################################
#                        BLASST
##############################################################
@tilelang.jit
def blasst_prefill_impl(
    mQ, mK, mV, mO,
    mPad, mBSMask,
    BM: int, BN: int,
    sm_scale: float,
    threshold: float,
    dtype: T.DType=T.float16,
    accum_dtype: T.DType=T.float32,
    num_stages: int=2,
):
    Hq, Hk, C = T.const('Hq, Hk, C')
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    BSK = T.dynamic('BSK')

    BTq, BTk = T.ceildiv(Tq, BM), T.ceildiv(Tk, BN)

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk

    with T.Kernel(B, Hq, BTq, threads=128) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)

        rPad = T.alloc_fragment([1], T.int32)
        rBSMask = T.alloc_fragment([1], T.int8)

        rMax = T.alloc_fragment([BM], accum_dtype)
        rMax_local = T.alloc_fragment([BM], accum_dtype)
        rSkip = T.alloc_fragment([BM], T.int8)
        rSkip_reduce = T.alloc_fragment([1], T.int8)
        rMax_tmp = T.alloc_fragment([BM], accum_dtype)
        rScale = T.alloc_fragment([BM], accum_dtype)
        rSum = T.alloc_fragment([BM], accum_dtype)
        rLogsum = T.alloc_fragment([BM], accum_dtype)
        rAcc = T.alloc_fragment([BM, C], accum_dtype)
        rAcc_tmp = T.alloc_fragment([BM, BN], accum_dtype)
        rP =  T.alloc_fragment([BM, BN], dtype)

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, bidz * BM:(bidz + 1) * BM, bidy, :], sQ, disable_tma=True)

        iter_num = T.ceildiv((bidz + 1) * BM - rPad[0] + 1, BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[iter], rBSMask)
            T.copy(mK[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy // G, :], sK, disable_tma=True)
            T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
            for i, j in T.Parallel(BM, BN):
                rAcc_tmp[i, j] = T.if_then_else(
                    (bidz * BM + i < Tq) & (iter * BN + j + rPad[0] < Tk) & (bidz * BM + i >= iter * BN + j + rPad[0]),
                    rAcc_tmp[i, j] * qk_scale,
                    -T.infinity(accum_dtype)
                )
            
            T.reduce_max(rAcc_tmp, rMax_local)
            for i in T.Parallel(BM):
                rMax_tmp[i] = T.max(rMax_local[i], rMax[i])
                rSkip[i] = T.exp2(rMax_tmp[i] - rMax_local[i]) * rScale[i] > threshold
            T.reduce_bitor(rSkip, rSkip_reduce)
            rBSMask[0] &= rSkip_reduce[0]
            if rBSMask[0] != 0:
                for i in T.Parallel(BM):
                    rScale[i] = T.exp2(rMax[i] - rMax_tmp[i])
                for i, j in T.Parallel(BM, BN):
                    rAcc_tmp[i, j] = T.exp2(rAcc_tmp[i, j] - rMax_tmp[i])
                T.reduce_sum(rAcc_tmp, rSum)
                for i in T.Parallel(BM):
                    rLogsum[i] = rLogsum[i] * rScale[i] + rSum[i]
                for i, j in T.Parallel(BM, C):
                    rAcc[i, j] *= rScale[i]
                
                T.copy(rAcc_tmp, rP)
                T.copy(mV[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy // G, :], sV, disable_tma=True)
                T.gemm(rP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(BM, C):
            rAcc[i, j] /= rLogsum[i]
        
        T.copy(rAcc, mO[bidx, bidz * BM:(bidz + 1) * BM, bidy, :], disable_tma=True)



##############################################################
#                     Seer Attention
##############################################################
@tilelang.jit
def seer_prefill_impl(
    mQ, mK, mV, mO,
    mPad, mBSMask,
    BM: int, BN: int,
    sm_scale: float,
    dtype: T.DType=T.float16,
    accum_dtype: T.DType=T.float32,
    num_stages: int=2,
):
    Hq, Hk, C = T.const('Hq, Hk, C')
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    BSQ, BSK = T.dynamic('BSQ, BSK')

    BTq, BTk = T.ceildiv(Tq, BM), T.ceildiv(Tk, BN)

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[B, Hk, BSQ, BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk

    with T.Kernel(B, Hq, BTq, threads=128) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)

        rPad = T.alloc_fragment([1], T.int32)
        rBSMask = T.alloc_fragment([1], T.int8)

        rMax = T.alloc_fragment([BM], accum_dtype)
        rMax_tmp = T.alloc_fragment([BM], accum_dtype)
        rScale = T.alloc_fragment([BM], accum_dtype)
        rSum = T.alloc_fragment([BM], accum_dtype)
        rLogsum = T.alloc_fragment([BM], accum_dtype)
        rAcc = T.alloc_fragment([BM, C], accum_dtype)
        rAcc_tmp = T.alloc_fragment([BM, BN], accum_dtype)
        rP =  T.alloc_fragment([BM, BN], dtype)

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, bidz * BM:(bidz + 1) * BM, bidy, :], sQ, disable_tma=True)

        iter_num = T.ceildiv((bidz + 1) * BM - rPad[0] + 1, BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[bidx, bidy // G, bidz, iter], rBSMask)
            if rBSMask[0] != 0:
                T.copy(mK[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy // G, :], sK, disable_tma=True)
                T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(BM, BN):
                    rAcc_tmp[i, j] = T.if_then_else(
                        (bidz * BM + i < Tq) & (iter * BN + j + rPad[0] < Tk) & (bidz * BM + i >= iter * BN + j + rPad[0]),
                        rAcc_tmp[i, j] * qk_scale,
                        -T.infinity(accum_dtype)
                    )
                
                T.reduce_max(rAcc_tmp, rMax_tmp)
                for i in T.Parallel(BM):
                    rMax_tmp[i] = T.max(rMax_tmp[i], rMax[i])
                    rScale[i] = T.exp2(rMax[i] - rMax_tmp[i])
                for i, j in T.Parallel(BM, BN):
                    rAcc_tmp[i, j] = T.exp2(rAcc_tmp[i, j] - rMax_tmp[i])
                T.reduce_sum(rAcc_tmp, rSum)
                for i in T.Parallel(BM):
                    rLogsum[i] = rLogsum[i] * rScale[i] + rSum[i]
                for i, j in T.Parallel(BM, C):
                    rAcc[i, j] *= rScale[i]
                
                T.copy(rAcc_tmp, rP)
                T.copy(mV[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy // G, :], sV, disable_tma=True)
                T.gemm(rP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(BM, C):
            rAcc[i, j] /= rLogsum[i]
        
        T.copy(rAcc, mO[bidx, bidz * BM:(bidz + 1) * BM, bidy, :], disable_tma=True)


class SparseAttentionTilelangPrefill:

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
        blasst_prefill_impl(
            q, k, v, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
            BLOCK_M, BLOCK_N, D ** -0.5, threshold, dtype=getattr(T, str(q.dtype).split('.')[-1])
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
        seer_prefill_impl(
            q, k, v, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
            BLOCK_M, BLOCK_N, D ** -0.5, dtype=getattr(T, str(q.dtype).split('.')[-1])
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

        BLOCK_M = tilelang.next_power_of_2(LQ)
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(64, max(16, tilelang.next_power_of_2(LQ)))
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)

        block_size = kwargs.pop('block_size', -1)
        if block_size > 0:
            BLOCK_N = block_size

        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_M > 32: BLOCK_M >>= 1
            elif num_stages > 2: num_stages -= 1
            else: break
            
        while HQ * B * tilelang.cdiv(LQ, BLOCK_M) < num_sm:
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
            **kwargs
        )