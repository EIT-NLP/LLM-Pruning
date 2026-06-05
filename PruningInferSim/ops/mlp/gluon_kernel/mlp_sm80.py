import torch
import triton
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

from typing import Optional, Tuple, Dict, List, Any
from einops import rearrange
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.ampere import (async_copy, mbarrier, mma_v2)

from ops.utils import get_autotune_config, get_autotune_cache

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
    producer: gl.uint32,
    mA: gl.tensor,
    mB: gl.tensor,
    sA: gl.shared_memory_descriptor,
    sB: gl.shared_memory_descriptor,
    nk_layout: gl.constexpr,
    n_base,
    mA_idx: gl.tensor,
    M: gl.int64,
    N: gl.int64,
    K: gl.int64,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_stages: gl.constexpr,
):
    index = producer % num_stages
    k_base = producer * BLOCK_K

    mA_mask = (mA_idx < M)[:, None]
    mB_mask = (n_base + gl.arange(0, BLOCK_N) < N)[:, None]

    async_copy.async_copy_global_to_shared(sA.index(index), mA + mA_idx[:, None] * K + k_base + gl.arange(0, BLOCK_K)[None, :], mask=mA_mask)
    async_copy.async_copy_global_to_shared(sB.index(index), mB + (n_base + gl.arange(0, BLOCK_N, layout=nk_layout))[:, None] * K + k_base + gl.arange(0, BLOCK_K)[None, :], mask=mB_mask)
    async_copy.commit_group()

    return producer + 1

@gluon.jit
def issue_mma(
    consumer: gl.uint32,
    sA: gl.shared_memory_descriptor,
    sB: gl.shared_memory_descriptor,
    rA_layout: gl.constexpr,
    rB_layout: gl.constexpr,
    rD: gl.tensor,
    num_stages: gl.constexpr,
):
    index = consumer % num_stages

    # load shared to register
    rA = sA.index(index).load(rA_layout)
    rB = sB.index(index).permute((1, 0)).load(rB_layout)

    rD = mma_v2(rA, rB, rD)
    return consumer + 1, rD

# SM80 gather & scatter fuse gemm.

def bm_sort_sm80_impl(
    mA: gl.tensor,
    mB: gl.tensor,
    mD: gl.tensor,
    route_mask: gl.tensor, # [M] / [cdiv(M, BLOCK_M)]
    route_indices: gl.tensor, # [M]
    M: gl.int64,
    N: gl.int64,
    K: gl.int64,
    sA_layout: gl.constexpr,
    sB_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE: gl.constexpr,
    IS_OFFLINE: gl.constexpr,
    num_stages: gl.constexpr,
    num_warps: gl.constexpr,
):
    bidx, bidy = gl.program_id(0), gl.program_id(1)
    bdimx, bdimy = gl.num_programs(0), gl.num_programs(1)

    bidx, bidy = swizzle_l2(bidx, bidy, bdimx, bdimy, GROUP_SIZE)

    dtype: gl.constexpr = mA.dtype.element_ty
    sA = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_M, BLOCK_K], sA_layout)
    sB = gl.allocate_shared_memory(dtype, [num_stages, BLOCK_N, BLOCK_K], sB_layout)
    
    m_base = bidx * BLOCK_M
    n_base = bidy * BLOCK_N
    num_k_tile = gl.cdiv(K, BLOCK_K)

    producer: gl.uint32 = 0
    consumer: gl.uint32 = 0
    
    # mma register layout per warp & thread
    rD_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0],
        warps_per_cta=[1, num_warps],
        instr_shape=[16, 8] # m16n8k8
    )
    rA_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=rD_layout,
        k_width=32 // 16,
    )
    rB_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1,
        parent=rD_layout,
        k_width=32 // 16,
    )
    rD = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, layout=rD_layout)

    # issue mask & index
    mn_layout: gl.constexpr = gl.BlockedLayout([1, 8], [32 // (BLOCK_K // 8), BLOCK_K // 8], [num_warps, 1], [1, 0])
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

        nk_layout: gl.constexpr = gl.SliceLayout(1, mn_layout)
        # prologue, issues first pipe - 1 loading
        for k in gl.static_range(0, num_stages - 1):
            producer = issue_load_AB(
                producer,
                mA, mB, sA, sB,
                nk_layout,
                n_base, rIndices,
                M, N, K,
                BLOCK_N, BLOCK_K, num_stages,
            )
        
        # mainloop, issue wmma
        for k in range(0, num_k_tile - (num_stages - 1)):
            producer = issue_load_AB(
                producer,
                mA, mB, sA, sB,
                nk_layout,
                n_base, rIndices,
                M, N, K,
                BLOCK_N, BLOCK_K, num_stages,
            )
            async_copy.wait_group(num_stages - 1)
            consumer, rD = issue_mma(
                consumer,
                sA, sB,
                rA_layout, rB_layout, rD,
                num_stages,
            )
        
        # final (num_stages - 1) mma
        for k in gl.static_range(0, num_stages - 1):
            async_copy.wait_group(num_stages - 2 - k)
            consumer, rD = issue_mma(
                consumer,
                sA, sB,
                rA_layout, rB_layout, rD,
                num_stages,
            )
        
        # epilogue

        rD_new = gl.convert_layout(rD, mn_layout, assert_trivial=False)
        gl.store(
            mD + rIndices[:, None] * N + n_base + gl.arange(0, BLOCK_N)[None, :],
            rD_new,
            mask=(rIndices < M)[:, None] & (n_base + gl.arange(0, BLOCK_N) < N)[None, :],
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
        b: Optional[torch.Tensor]=None,
        BLOCK_M: Optional[int]=64,
        BLOCK_N: Optional[int]=32,
        BLOCK_K: Optional[int]=32,
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
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        gl_dtype = getattr(gl, str(x_flat.dtype).split('.')[1])
        sA_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl_dtype)
        sB_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_N, BLOCK_K], gl_dtype)

        config = get_autotune_config(
            params=['GROUP_SIZE', 'num_stages'],
            GROUP_SIZE=GROUP_SIZE,
            num_stages=num_stages,
            **kwargs,
        )
        kernel = get_autotune_cache(
            bm_sort_sm80_impl,
            enable_autotune=True,
            config=config,
            keys=['M', 'N', 'K'],
            is_gluon=True
        )
        kernel[grid](
            x_flat, w, out,
            m_sort, m_sort_indices.to(torch.uint32),
            M, N, K,
            sA_layout, sB_layout,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
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

        estimated_sparsity = kwargs.pop('estimated_sparsity', 0)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, L), dtype=torch.bool, device=device)
        
        if impl == 'auto': # auto dispatch
            if route_mask is None: impl = 'dense'
            else: impl = 'sort_online'

        if impl in ['sort_offline', 'sort_online']:
            BLOCK_M = triton.next_power_of_2(int(M * estimated_sparsity))
            if BLOCK_M >= int(M * estimated_sparsity): BLOCK_M = BLOCK_M >> 1

            BLOCK_M = min(128, max(16, BLOCK_M))
            BLOCK_N = min(128, max(16, triton.next_power_of_2(N)))
            
            BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
            BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
            BLOCK_K = min(64, max(32, triton.next_power_of_2(K)))

            while BLOCK_N * BLOCK_K > 128 * 128 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1

            while BLOCK_M * BLOCK_N * BLOCK_K > 128 * 64 * 64 and BLOCK_K > 32:
                BLOCK_K = BLOCK_K >> 1
            
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
            GROUP_SIZE=group_size,
            num_stages=num_stages,
            num_warps=num_warps,
            route_mask=route_mask,
            is_offline=is_offline,
            BLOCK_M_list=[16, 32, 64, 128],
            BLOCK_N_list=[32, 64, 128],
            BLOCK_K_list=[32, 64],
            num_stages_list=[2, 3],
            **kwargs
        )