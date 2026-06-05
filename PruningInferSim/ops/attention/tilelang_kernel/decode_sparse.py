import torch
import tilelang
import tilelang.language as T

from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange
from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn

PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
}

##############################################################
#                        BLASST
##############################################################
@tilelang.jit(pass_configs=PASS_CFG)
def blasst_decode_fuse_impl(
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

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk

    with T.Kernel(B, Hk, threads=128) as (bidx, bidy):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)
        sP = T.alloc_shared([BM, BN], dtype)

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

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, 0, bidy * G:bidy * G + BM, :], sQ, disable_tma=True)

        iter_num = T.ceildiv(Tk - rPad[0], BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[iter], rBSMask)
            T.copy(mK[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy, :], sK, disable_tma=True)
            T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
            for i, j in T.Parallel(BM, BN):
                rAcc_tmp[i, j] = T.if_then_else(
                    (iter * BN + j + rPad[0] < Tk),
                    rAcc_tmp[i, j] * qk_scale,
                    -T.infinity(accum_dtype)
                )
            
            T.reduce_max(rAcc_tmp, rMax_tmp)
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
                
                T.copy(rAcc_tmp, sP)
                T.copy(mV[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy, :], sV, disable_tma=True)
                T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(BM, C):
            rAcc[i, j] /= rLogsum[i]

        for i, j in T.Parallel(G, C):
            mO[bidx, 0, bidy * G + i, j] = rAcc[i, j]

@tilelang.jit(pass_configs=PASS_CFG)
def blasst_decode_split_impl(
    mQ, mK, mV, mO_tmp, mO_meta, mO,
    mPad, mBSMask,
    BM: int, BN: int, SplitK: int,
    sm_scale: float,
    threshold: float,
    dtype: T.DType=T.float16,
    accum_dtype: T.DType=T.float32,
    num_stages: int=0,
):
    Hq, Hk, C = T.const('Hq, Hk, C')
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    BSK = T.dynamic('BSK')

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mO_tmp: T.Tensor[[B, Tq, Hq, SplitK, C], accum_dtype]
    mO_meta: T.Tensor[[B, Tq, Hq, 2, SplitK], accum_dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk
    split_size = T.ceildiv(Tk, SplitK)

    with T.Kernel(B, Hk, SplitK, threads=128) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)
        sP = T.alloc_shared([BM, BN], dtype)

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

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, 0, bidy * G:bidy * G + BM, :], sQ, disable_tma=True)

        start_k = bidz * split_size + rPad[0]
        end_k = T.min(start_k + split_size, Tk)
        iter_num = T.ceildiv(end_k - start_k, BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[(start_k - rPad[0]) // BN + iter], rBSMask)
            T.copy(mK[bidx, start_k + iter * BN:start_k + (iter + 1) * BN, bidy, :], sK, disable_tma=True)
            T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
            for i, j in T.Parallel(BM, BN):
                rAcc_tmp[i, j] = T.if_then_else(
                    (start_k + (iter * BN) + j < end_k),
                    rAcc_tmp[i, j] * qk_scale,
                    -T.infinity(accum_dtype)
                )
            
            T.reduce_max(rAcc_tmp, rMax_tmp)
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
                
                T.copy(rAcc_tmp, sP)
                T.copy(mV[bidx, start_k + iter * BN:start_k + (iter + 1) * BN, bidy, :], sV, disable_tma=True)
                T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(G, C):
            mO_tmp[bidx, 0, bidy * G + i, bidz, j] = rAcc[i, j]
        for i in T.Parallel(G):
            mO_meta[bidx, 0, bidy * G + i, 0, bidz] = rMax[i]
            mO_meta[bidx, 0, bidy * G + i, 1, bidz] = rLogsum[i]
    
    # merge
    with T.Kernel(B, Hq, threads=128) as (bidx, bidy):
        rAcc = T.alloc_fragment([SplitK, C], accum_dtype)
        rMax = T.alloc_fragment([SplitK], accum_dtype)
        rLogsum = T.alloc_fragment([SplitK], accum_dtype)

        rMax_global = T.alloc_fragment([1], accum_dtype)
        rScale = T.alloc_fragment([SplitK], accum_dtype)
        rSum_reduce = T.alloc_fragment([1], accum_dtype)
        rAcc_reduce = T.alloc_fragment([C], accum_dtype)

        for i in T.Serial(SplitK):
            rMax[i] = mO_meta[bidx, 0, bidy, 0, i]
            rLogsum[i] = mO_meta[bidx, 0, bidy, 1, i]
            for j in T.Parallel(C):
                rAcc[i, j] = mO_tmp[bidx, 0, bidy, i, j]

        T.reduce_max(rMax, rMax_global)
        for i in T.Serial(SplitK):
            rScale[i] = T.exp2(rMax[i] - rMax_global[0])
            rLogsum[i] *= rScale[i]
        T.reduce_sum(rLogsum, rSum_reduce)
        for i in T.Serial(SplitK):
            for j in T.Parallel(C):
                rAcc[i, j] = rAcc[i, j] * rScale[i]
        T.reduce_sum(rAcc, rAcc_reduce, dim=0)
        for i in T.Parallel(C):
            mO[bidx, 0, bidy, i] = rAcc_reduce[i] / rSum_reduce[0]

##############################################################
#                     Seer Attention
##############################################################
@tilelang.jit(pass_configs=PASS_CFG)
def seer_decode_fuse_impl(
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

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[B, Hk, BSQ, BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk

    with T.Kernel(B, Hk, threads=128) as (bidx, bidy):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)
        sP = T.alloc_shared([BM, BN], dtype)

        rPad = T.alloc_fragment([1], T.int32)
        rBSMask = T.alloc_fragment([1], T.int8)

        rMax = T.alloc_fragment([BM], accum_dtype)
        rMax_tmp = T.alloc_fragment([BM], accum_dtype)
        rScale = T.alloc_fragment([BM], accum_dtype)
        rSum = T.alloc_fragment([BM], accum_dtype)
        rLogsum = T.alloc_fragment([BM], accum_dtype)
        rAcc = T.alloc_fragment([BM, C], accum_dtype)
        rAcc_tmp = T.alloc_fragment([BM, BN], accum_dtype)

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, 0, bidy * G:bidy * G + BM, :], sQ, disable_tma=True)

        iter_num = T.ceildiv(Tk - rPad[0], BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[bidx, bidy, 0, iter], rBSMask)
            if rBSMask[0] != 0:
                T.copy(mK[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy, :], sK, disable_tma=True)
                T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(BM, BN):
                    rAcc_tmp[i, j] = T.if_then_else(
                        (iter * BN + j + rPad[0] < Tk),
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
                
                T.copy(rAcc_tmp, sP)
                T.copy(mV[bidx, iter * BN + rPad[0]:(iter + 1) * BN + rPad[0], bidy, :], sV, disable_tma=True)
                T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(BM, C):
            rAcc[i, j] /= rLogsum[i]

        for i, j in T.Parallel(G, C):
            mO[bidx, 0, bidy * G + i, j] = rAcc[i, j]


@tilelang.jit(pass_configs=PASS_CFG)
def seer_decode_split_impl(
    mQ, mK, mV, mO_tmp, mO_meta, mO,
    mPad, mBSMask,
    BM: int, BN: int, SplitK: int,
    sm_scale: float,
    dtype: T.DType=T.float16,
    accum_dtype: T.DType=T.float32,
    num_stages: int=0,
):
    Hq, Hk, C = T.const('Hq, Hk, C')
    B, Tq, Tk = T.dynamic('B, Tq, Tk')
    BSQ, BSK = T.dynamic('BSQ, BSK')

    mQ: T.Tensor[[B, Tq, Hq, C], dtype]
    mK: T.Tensor[[B, Tk, Hk, C], dtype]
    mV: T.Tensor[[B, Tk, Hk, C], dtype]
    mO: T.Tensor[[B, Tq, Hq, C], dtype]
    mO_tmp: T.Tensor[[B, Tq, Hq, SplitK, C], accum_dtype]
    mO_meta: T.Tensor[[B, Tq, Hq, 2, SplitK], accum_dtype]
    mPad: T.Tensor[[B], T.int32]
    mBSMask: T.Tensor[[B, Hk, BSQ, BSK], T.int8]

    qk_scale = sm_scale * 1.44269504
    G = Hq // Hk
    split_size = T.ceildiv(Tk, SplitK)

    with T.Kernel(B, Hk, SplitK, threads=128) as (bidx, bidy, bidz):
        sQ = T.alloc_shared([BM, C], dtype)
        sK = T.alloc_shared([BN, C], dtype)
        sV = T.alloc_shared([BN, C], dtype)
        sP = T.alloc_shared([BM, BN], dtype)

        rPad = T.alloc_fragment([1], T.int32)
        rBSMask = T.alloc_fragment([1], T.int8)

        rMax = T.alloc_fragment([BM], accum_dtype)
        rMax_tmp = T.alloc_fragment([BM], accum_dtype)
        rScale = T.alloc_fragment([BM], accum_dtype)
        rSum = T.alloc_fragment([BM], accum_dtype)
        rLogsum = T.alloc_fragment([BM], accum_dtype)
        rAcc = T.alloc_fragment([BM, C], accum_dtype)
        rAcc_tmp = T.alloc_fragment([BM, BN], accum_dtype)

        T.fill(rMax, -T.infinity(accum_dtype))
        T.fill(rLogsum, 0.0)
        T.fill(rAcc, 0.0)

        T.copy(mPad[bidx], rPad)
        T.copy(mQ[bidx, 0, bidy * G:bidy * G + BM, :], sQ, disable_tma=True)

        start_k = bidz * split_size + rPad[0]
        end_k = T.min(start_k + split_size, Tk)
        iter_num = T.ceildiv(end_k - start_k, BN)
        for iter in T.Pipelined(iter_num, num_stages=num_stages):
            T.copy(mBSMask[bidx, bidy, 0, (start_k - rPad[0]) // BN + iter], rBSMask)
            if rBSMask[0] != 0:
                T.copy(mK[bidx, start_k + iter * BN:start_k + (iter + 1) * BN, bidy, :], sK, disable_tma=True)
                T.gemm(sQ, sK, rAcc_tmp, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
                for i, j in T.Parallel(BM, BN):
                    rAcc_tmp[i, j] = T.if_then_else(
                        (start_k + (iter * BN) + j < end_k),
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
                
                T.copy(rAcc_tmp, sP)
                T.copy(mV[bidx, start_k + iter * BN:start_k + (iter + 1) * BN, bidy, :], sV, disable_tma=True)
                T.gemm(sP, sV, rAcc, policy=T.GemmWarpPolicy.FullRow, clear_accum=False)

                T.copy(rMax_tmp, rMax)
        
        for i, j in T.Parallel(G, C):
            mO_tmp[bidx, 0, bidy * G + i, bidz, j] = rAcc[i, j]
        for i in T.Parallel(G):
            mO_meta[bidx, 0, bidy * G + i, 0, bidz] = rMax[i]
            mO_meta[bidx, 0, bidy * G + i, 1, bidz] = rLogsum[i]
    
    # merge
    with T.Kernel(B, Hq, threads=128) as (bidx, bidy):
        rAcc = T.alloc_fragment([SplitK, C], accum_dtype)
        rMax = T.alloc_fragment([SplitK], accum_dtype)
        rLogsum = T.alloc_fragment([SplitK], accum_dtype)

        rMax_global = T.alloc_fragment([1], accum_dtype)
        rScale = T.alloc_fragment([SplitK], accum_dtype)
        rSum_reduce = T.alloc_fragment([1], accum_dtype)
        rAcc_reduce = T.alloc_fragment([C], accum_dtype)

        for i in T.Serial(SplitK):
            rMax[i] = mO_meta[bidx, 0, bidy, 0, i]
            rLogsum[i] = mO_meta[bidx, 0, bidy, 1, i]
            for j in T.Parallel(C):
                rAcc[i, j] = mO_tmp[bidx, 0, bidy, i, j]

        T.reduce_max(rMax, rMax_global)
        for i in T.Serial(SplitK):
            rScale[i] = T.exp2(rMax[i] - rMax_global[0])
            rLogsum[i] *= rScale[i]
        T.reduce_sum(rLogsum, rSum_reduce)
        for i in T.Serial(SplitK):
            for j in T.Parallel(C):
                rAcc[i, j] = rAcc[i, j] * rScale[i]
        T.reduce_sum(rAcc, rAcc_reduce, dim=0)
        for i in T.Parallel(C):
            mO[bidx, 0, bidy, i] = rAcc_reduce[i] / rSum_reduce[0]

class SparseAttentionTilelangDecode:

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
        num_split: Optional[int]=1,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
        **kwargs
    ):
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4

        _, LK, HK, _ = k.shape
        G = HQ // HK

        out = torch.empty_like(q)
        if num_split == 1:
            blasst_decode_fuse_impl(
                q, k, v, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
                BLOCK_M, BLOCK_N, D ** -0.5, threshold, dtype=getattr(T, str(q.dtype).split('.')[-1])
            )
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
            blasst_decode_split_impl(
                q, k, v, out_local, metadata, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
                BLOCK_M, BLOCK_N, num_split, D ** -0.5, threshold, dtype=getattr(T, str(q.dtype).split('.')[-1])
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
        num_split: Optional[int]=1,
        num_stages: Optional[int]=2,
        num_warps: Optional[int]=4,
        do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'],
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
        if num_split == 1:
            seer_decode_fuse_impl(
                q, k, v, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
                BLOCK_M, BLOCK_N, D ** -0.5, dtype=getattr(T, str(q.dtype).split('.')[-1])
            )
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
            seer_decode_split_impl(
                q, k, v, out_local, metadata, out, pad_offset.to(torch.int32), execute_block.to(torch.int8),
                BLOCK_M, BLOCK_N, num_split, D ** -0.5, dtype=getattr(T, str(q.dtype).split('.')[-1])
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

        BLOCK_M = max(16, tilelang.next_power_of_2(G))
        BLOCK_N = min(64, max(16, tilelang.next_power_of_2(LK)))
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
        
        return getattr(cls, f"_{impl}_kernel")(
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_split=num_split,
            num_stages=num_stages,
            num_warps=num_warps,
            pad_offset=pad_offset,
            **kwargs
        )