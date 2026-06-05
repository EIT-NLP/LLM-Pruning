import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.ampere import mma_v2
from triton.experimental.gluon.language.nvidia.blackwell import (tma, mbarrier, fence_async_shared)

from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_gemm

# SM120 gather & scatter fuse gemm, with TMA gather4 load A, TMA load B, and sm80 mma

@aggregate
class GEMMConfig:
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    BLOCK_K: gl.constexpr
    SPLIT_K: gl.constexpr
    GROUP_SIZE: gl.constexpr
    num_stages: gl.constexpr
    num_warps: gl.constexpr

    mma_layout: gl.constexpr
    rA_layout: gl.constexpr
    rB_layout: gl.constexpr

    sA_layout: gl.constexpr
    sB_layout: gl.constexpr

    dtype: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, GROUP_SIZE, num_stages, num_warps, dtype):
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)
        self.BLOCK_K = gl.constexpr(BLOCK_K)
        self.SPLIT_K = gl.constexpr(SPLIT_K)
        self.GROUP_SIZE = gl.constexpr(GROUP_SIZE)
        self.num_stages = gl.constexpr(num_stages)
        self.num_warps = gl.constexpr(num_warps)
        self.dtype = gl.constexpr(dtype)
        
        mma_col: gl.constexpr = 2 if self.BLOCK_K == 32 else 4
        mma_row: gl.constexpr = self.num_warps // mma_col

        self.mma_layout = gl.constexpr(gl.NVMMADistributedLayout(
            version=[2, 0],
            warps_per_cta=[mma_row, mma_col],
            instr_shape=[16, 8],
        ))
        self.rA_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=0,
            parent=self.mma_layout,
            k_width=2,
        ))
        self.rB_layout = gl.constexpr(gl.DotOperandLayout(
            operand_index=1,
            parent=self.mma_layout,
            k_width=2,
        ))

        self.sA_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([self.BLOCK_M, self.BLOCK_K], self.dtype))
        self.sB_layout = gl.constexpr(gl.NVMMASharedLayout.get_default_for([self.BLOCK_N, self.BLOCK_K], self.dtype))


@gluon.jit
def swizzle_l2(i, j, size_i, size_j, size_g):
    # "unrolled index in array"
    ij = i * size_j + j
    # number of elements in `size_g` groups
    # of `size_j` columns
    size_gj = size_g * size_j
    # index of the group in which (i,j) is
    group_id = ij // size_gj
    # row-index of the first element of this group
    off_i = group_id * size_g
    # last group may have fewer rows
    size_g = gl.minimum(size_i - off_i, size_g)
    # linear index with respect to the first element in this group
    ij = ij % size_gj
    # new row and column indices
    new_i = off_i + ij % size_g
    new_j = ij // size_g
    return new_i, new_j

@gluon.jit
def issue_load_AB(
    producer: gl.int64,
    mA_desc: tma.tensor_descriptor,
    mB_desc: tma.tensor_descriptor,
    sA: gl.shared_memory_descriptor,
    sB: gl.shared_memory_descriptor,
    bars: gl.shared_memory_descriptor,
    n_base,
    k_offset,
    mA_idx: gl.tensor,
    BLOCK_M: gl.constexpr,
    num_stages: gl.constexpr,
    pred=True,
):
    index = producer % num_stages
    producer += 1
    bar = bars.index(index)

    mbarrier.expect(bar, BLOCK_M * mA_desc.block_type.nbytes + mB_desc.block_type.nbytes)
    tma.async_gather(
        mA_desc, mA_idx, k_offset,
        barrier=bar,
        result=sA.index(index),
        pred=pred,
    )
    tma.async_copy_global_to_shared(
        mB_desc, [n_base, k_offset],
        barrier=bar,
        result=sB.index(index),
        pred=pred,
    )
    return producer

@gluon.jit
def issue_mma(
    consumer: gl.int64,
    sA: gl.shared_memory_descriptor,
    sB: gl.shared_memory_descriptor,
    bars: gl.shared_memory_descriptor,
    rA_layout: gl.constexpr,
    rB_layout: gl.constexpr,
    rD: gl.tensor,
    num_stages: gl.constexpr,
):
    index = consumer % num_stages
    phase = consumer // num_stages & 1
    consumer += 1
    bar = bars.index(index)

    mbarrier.wait(bar, phase)
    # load shared to register
    rA = sA.index(index).load(rA_layout)
    rB = sB.index(index).permute((1, 0)).load(rB_layout)

    rD = mma_v2(rA, rB, rD)
    return consumer, rD

def bm_sort_sm120_impl(
    mA_desc: tma.tensor_descriptor,
    mB_desc: tma.tensor_descriptor,
    mD,
    route_mask: gl.tensor, # [M] / [cdiv(M, BLOCK_M)]
    route_indices: gl.tensor, # [M]
    M: gl.int64,
    N: gl.int64,
    K: gl.int64,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    IS_OFFLINE: gl.constexpr,
    SPLIT_K: gl.constexpr,
    ACTIVATION: gl.constexpr,
    num_stages: gl.constexpr,
    num_warps: gl.constexpr,
):
    bidx, bidy, bidz = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    bdimx, bdimy = gl.num_programs(0), gl.num_programs(1)

    bidx, bidy = swizzle_l2(bidx, bidy, bdimx, bdimy, GROUP_SIZE)
    cfg = GEMMConfig(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, GROUP_SIZE, num_stages, num_warps, mA_desc.dtype)

    sA = gl.allocate_shared_memory(cfg.dtype, [num_stages, BLOCK_M, BLOCK_K], cfg.sA_layout)
    sB = gl.allocate_shared_memory(cfg.dtype, [num_stages, BLOCK_N, BLOCK_K], cfg.sB_layout)

    bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_stages):
        mbarrier.init(bars.index(i), count=1)
    
    m_base = bidx * BLOCK_M
    n_base = bidy * BLOCK_N
    k_base = bidz * BLOCK_K
    split_k_size = gl.cdiv(K, SPLIT_K)
    num_k_tile = gl.cdiv(split_k_size, BLOCK_K)

    producer: gl.int64 = 0
    consumer: gl.int64 = 0
    
    # mma register layout per warp & thread
    rD = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=cfg.mma_layout)

    # issue mask & index
    mn_layout: gl.constexpr = gl.BlockedLayout([4, 1], [1, 32], [num_warps, 1], [1, 0])
    indices_layout: gl.constexpr = gl.SliceLayout(1, mn_layout)

    if IS_OFFLINE:
        rSkip = gl.load(route_mask + bidx)
    else:
        rMask = gl.load(
            route_mask + m_base + gl.arange(0, BLOCK_M, layout=indices_layout),
            mask=m_base + gl.arange(0, BLOCK_M, layout=indices_layout) < M,
            other=0,
        )
        rSkip = gl.reduce_or(rMask, axis=-1)
    
    if rSkip > 0:
        if IS_OFFLINE:
            rIndices = gl.load(
                route_indices + m_base + gl.arange(0, BLOCK_M, layout=indices_layout),
                mask=m_base + gl.arange(0, BLOCK_M, layout=indices_layout) < M,
                other=M,
            )
        else:
            rIndices = gl.load(
                route_indices + m_base + gl.arange(0, BLOCK_M, layout=indices_layout),
                mask=rMask > 0,
                other=M,
            )

        # prologue, issues first pipe - 1 loading
        for k in gl.static_range(0, num_stages - 1):
            producer = issue_load_AB(
                producer,
                mA_desc, mB_desc, sA, sB, bars,
                n_base, k_base + producer * BLOCK_K * SPLIT_K, rIndices,
                BLOCK_M, num_stages,
            )
        
        # mainloop, issue wmma
        for k in range(0, num_k_tile - (num_stages - 1)):
            producer = issue_load_AB(
                producer,
                mA_desc, mB_desc, sA, sB, bars,
                n_base, k_base + producer * BLOCK_K * SPLIT_K, rIndices,
                BLOCK_M, num_stages,
            )

            consumer, rD = issue_mma(
                consumer,
                sA, sB, bars,
                cfg.rA_layout, cfg.rB_layout, rD,
                num_stages,
            )
        
        # final (num_stages - 1) mma
        for k in gl.static_range(0, num_stages - 1):
            consumer, rD = issue_mma(
                consumer,
                sA, sB, bars,
                cfg.rA_layout, cfg.rB_layout, rD,
                num_stages,
            )

        for i in gl.static_range(num_stages):
            mbarrier.invalidate(bars.index(i))
        
        # epilogue
        store_col: gl.constexpr = 1 if BLOCK_K == 32 else 2
        store_row: gl.constexpr = num_warps // store_col
        mn_r2g_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 4], [store_row, store_col], [1, 0])
        rIndices = gl.convert_layout(rIndices, gl.SliceLayout(1, mn_r2g_layout), assert_trivial=False)

        if SPLIT_K == 1:
            if ACTIVATION == 'silu': rD *= (1 / (1 + gl.exp(-rD)))
            if ACTIVATION == 'relu': rD = gl.maximum(rD)

            rD_new = gl.convert_layout(rD, mn_r2g_layout, assert_trivial=False)
            gl.store(
                mD + rIndices[:, None] * N + n_base + gl.arange(0, BLOCK_N)[None, :],
                rD_new,
                mask=(rIndices < M)[:, None] & (n_base + gl.arange(0, BLOCK_N) < N)[None, :],
            )
        
        else:
            rD_new = gl.convert_layout(rD, mn_r2g_layout, assert_trivial=False)
            gl.atomic_add(
                mD + rIndices[:, None] * N + n_base + gl.arange(0, BLOCK_N)[None, :],
                rD_new,
                mask=(rIndices < M)[:, None] & (n_base + gl.arange(0, BLOCK_N) < N)[None, :],
                sem='relaxed',
            )

class BMSparseMLP:
    support_kernel = [
        'sort_online',
        'sort_offline',
    ]

    @classmethod
    def _sort_kernel(
        cls,
        x: torch.Tensor,
        route_mask: torch.Tensor,
        w: torch.Tensor,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
        SPLIT_K: Optional[int]=1,
        GROUP_SIZE: Optional[int]=4,
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

        # tma descriptor
        gl_dtype = getattr(gl, str(x_flat.dtype).split('.')[1])
        A_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl_dtype)
        B_desc_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl_dtype)

        A_desc = TensorDescriptor.from_tensor(x_flat, [1, BLOCK_K], A_desc_layout)
        B_desc = TensorDescriptor.from_tensor(w, [BLOCK_N, BLOCK_K], B_desc_layout)

        config = get_autotune_config(
            params=['BLOCK_M', 'BLOCK_N', 'BLOCK_K', 'SPLIT_K', 'GROUP_SIZE', 'num_stages'],
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            SPLIT_K=SPLIT_K,
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            bm_sort_sm120_impl,
            enable_autotune=True,
            config=config,
            keys=['M', 'N', 'K'],
            is_gluon=True
        )
        kernel[grid](
            A_desc, B_desc, out,
            m_sort, m_sort_indices.to(torch.int32),
            M, N, K,
            ACTIVATION=kwargs.get('activation', 'identity'),
            IS_OFFLINE=kwargs.get('is_offline', False),
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
            else: impl = 'sort_online'

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M >= int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(64, triton.next_power_of_2(N)))
            BLOCK_K = min(64, max(32, triton.next_power_of_2(K)))

            SPLIT_K = 1
            SPLIT_K_list = [1]

            while not check_shared_memory_gemm(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, dtype.itemsize):
                if BLOCK_K > 32: BLOCK_K >>= 1
                elif BLOCK_N > 64: BLOCK_N >>= 1
                elif BLOCK_M > 32: BLOCK_M >>= 1
                else: num_stages -= 1
            
            # split-k active
            if triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) < num_sm:
                SPLIT_K_list = [2, 3, 4, 5, 6]
                min_waste = 1.0
                best_split_k = 2
                for split_k in SPLIT_K_list:
                    waste = float(num_sm - ((triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * split_k) % num_sm)) / float(num_sm)
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
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[64, 128],
            BLOCK_K_list=[32, 64],
            SPLIT_K_list=SPLIT_K_list,
            num_stages_list=[2, 3],
            **kwargs
        )