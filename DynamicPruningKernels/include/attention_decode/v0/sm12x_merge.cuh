#pragma once
#include "attention_decode/v0/traits.cuh"

// main kernel func
template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG,
    uint32_t kBM, uint32_t kBN,
    uint32_t kSplitRound, class MergeTraits,
    typename DType, uint32_t kWarps=4
>
__global__ __launch_bounds__(256) void attention_head_v0_merge(const __grid_constant__ MergeParams params) {
    using namespace cute;
    const auto& [Mask, O, pO, LSE, B, Tk, Split, scale] = params;
    constexpr uint32_t Hq = kHq;
    constexpr uint32_t Hk = kHk;
    constexpr uint32_t D = kD;
    constexpr uint32_t NG = kNG;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t SplitRound = kSplitRound;

    constexpr uint32_t ElemPerThr = MergeTraits::ElemPerThr;
    using sOLayout = typename MergeTraits::sPOLayout;
    using g2sO_tiled_copy = typename MergeTraits::g2sO_tiled_copy;
    using s2rO_tiled_copy = typename MergeTraits::s2rO_tiled_copy;

    // grid: (1, Hq, B)
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    constexpr uint32_t group_size = Hq / NG;
    const uint32_t group_id = bidy / group_size;

    //
    // prepare tensors and smem layout
    //
    Tensor mO = make_tensor(
        make_gmem_ptr(static_cast<DType*>(O)),
        make_ordered_layout(
            make_shape(_1{}, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mpO = make_tensor(
        make_gmem_ptr(static_cast<DType*>(pO)),
        make_ordered_layout(
            make_shape(Int<SplitRound>{}, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mLSE = make_tensor(
        make_gmem_ptr(static_cast<float*>(LSE)),
        make_ordered_layout(
            make_shape(Int<Hq>{}, Int<SplitRound>{}, B),
            make_step(_1{}, _0{}, _2{})
        )
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_layout(make_shape(B, _1{}, Int<NG>{}), LayoutRight{})
    );

    extern __shared__ uint8_t shared_memory[];
    using SharedStorageO = SharedMemoryPO<DType, sOLayout, SplitRound>;
    SharedStorageO &smem_o = *reinterpret_cast<SharedStorageO*>(shared_memory);
    Tensor sO = make_tensor(
        make_smem_ptr(smem_o.PO.begin()),
        sOLayout{}
    );

    // load mask
    if (mMask(make_coord(bidz, 0, group_id)) == 0) return;

    Tensor gpO = local_tile(
        mpO,
        make_tile(Int<SplitRound>{}, Int<D>{}),
        make_coord(0, 0, bidy, bidz)
    );
    auto g2sO_thr_copy = g2sO_tiled_copy{}.get_slice(tidx);
    auto pSgO = g2sO_thr_copy.partition_S(gpO);
    auto pDsO = g2sO_thr_copy.partition_D(sO);
    copy(g2sO_tiled_copy{}, pSgO, pDsO);
    cp_async_fence();

    auto s2rO_thr_copy = s2rO_tiled_copy{}.get_slice(tidx);
    auto pSsO = s2rO_thr_copy.partition_S(sO);
    auto pDrO = make_fragment_like<DType>(pSsO);

    auto rO_final = make_fragment_like<float>(pDrO(make_coord(_, _), _0{}, _0{}));
    clear(rO_final);

    // handle global scale when pO on the fly
    float rL = -MAXFLOAT;
    float rM = -MAXFLOAT;
    float rScale = 0;
    float rSum = 0;
    if (warp_id == 0 && lane_id < Split) {
        rL = mLSE(make_coord(bidy, lane_id, bidz));
        rM = rL;
    }
    
    CUTE_UNROLL
    for (int32_t off=16; off > 0; off >>= 1) {
        rM = cute::max(rM, __shfl_xor_sync(0xffffffff, rM, off));
    }
    if (warp_id == 0 and lane_id < Split) {
        rScale = exp2f(rL - rM);
        rSum = rScale;
    }
    CUTE_UNROLL
    for (int32_t off=16; off > 0; off >>= 1) {
        rSum += __shfl_down_sync(0xffffffff, rSum, off);
    }

    if (warp_id == 0 && lane_id < Split) {
        smem_o.global_scale[lane_id] = rScale;
        if (lane_id == 0) smem_o.global_scale[Split] = __frcp_rn(rSum);
    }
    
    cp_async_wait<0>();
    __syncthreads();

    CUTE_NO_UNROLL
    for (uint32_t iter=0; iter < Split; ++iter) {
        auto trO = pDrO(make_coord(_, _), iter, _0{});
        auto trO_cast = make_fragment_like<float>(trO);

        copy(s2rO_tiled_copy{}, pSsO(make_coord(_, _), iter, _0{}), trO);
        copy(trO, trO_cast);

        CUTE_UNROLL
        for (uint32_t i=0; i < size(trO_cast); ++i) {
            trO_cast(i) *= smem_o.global_scale[iter];
            rO_final(i) += trO_cast(i);
        }
    }

    float final_scale = smem_o.global_scale[Split];
    CUTE_UNROLL
    for (uint32_t i=0; i < size(rO_final); ++i) {
        rO_final(i) *= final_scale;
    }

    // write back
    auto rO_final_cast = make_tensor_like<DType>(rO_final);
    auto gO = local_tile(
        mO,
        make_tile(_1{}, Int<D>{}),
        make_coord(_0{}, _0{}, bidy, bidz)
    );
    auto pDgO = s2rO_thr_copy.partition_D(gO);

    copy(rO_final, rO_final_cast);
    copy(s2rO_tiled_copy{}, rO_final_cast, pDgO(make_coord(_, _), _0{}, _0{}));
}