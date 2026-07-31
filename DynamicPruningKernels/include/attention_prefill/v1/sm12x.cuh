#pragma once
#include "utils.cuh"
// sglang jit plugin
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
// cute
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <type_traits>

namespace {

#define DEBUG_COND (tidx == 0 && bidx + bidy + bidz == 0)

// layout for smem
template <typename DType, class LayoutQ, uint32_t kWarps=4>
struct SharedMemoryQ {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutQ>> Q;

    alignas(16) uint32_t cta_reduce_cache[kWarps * 2 + 2];
};
template <typename DType, class LayoutK, class LayoutV, uint32_t PipelineK, uint32_t PipelineV>
struct SharedMemoryKV {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutK>> K;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutV>> V;

    alignas(16) uint64_t tma_k_barrier[PipelineK];
    alignas(16) uint64_t tma_v_barrier[PipelineV];
};
template <typename DType, class LayoutO>
struct SharedMemoryO {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutO>> O;
};

// helper
template <uint32_t kMMAIter, bool kWarpSkipG2S, bool kWarpSkipMMA>
struct SkipHelper;

template <uint32_t kMMAIter>
struct SkipHelper<kMMAIter, false, false> {
    uint8_t execute_cta;
    uint8_t rMask[kMMAIter*2];
    uint32_t rIndex[kMMAIter*2];
};

template <uint32_t kMMAIter>
struct SkipHelper<kMMAIter, true, false> {
    uint8_t execute_cta;
    uint8_t rMask[kMMAIter*2];
    uint32_t rIndex[kMMAIter*2];

    uint8_t execute_g2s_warp[kMMAIter*2];
};

template <uint32_t kMMAIter>
struct SkipHelper<kMMAIter, false, true> {
    uint8_t execute_cta;
    uint8_t rMask[kMMAIter*2];
    uint32_t rIndex[kMMAIter*2];

    uint8_t execute_mma_warp[kMMAIter];
};

template <uint32_t kMMAIter>
struct SkipHelper<kMMAIter, true, true> {
    uint8_t execute_cta;
    uint8_t rMask[kMMAIter*2];
    uint32_t rIndex[kMMAIter*2];

    uint8_t execute_g2s_warp[kMMAIter*2];
    uint8_t execute_mma_warp[kMMAIter];
};

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kWarps=4, uint32_t kBytes=2>
struct WarpLayoutTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;

    // mma warp layout
    // mma perm at least to be (MMARow * 16, MMACol * 8 * 2), enable s2rB copy use LDSMx4
    // using (4, 1) layout for row-wise warp shuffle
    static constexpr uint32_t MMACol = 1;
    static constexpr uint32_t MMARow = kWarps;
    static constexpr uint32_t MMAIter = cute::ceil_div(kBM, (16 * MMARow));

    // infer the G2SQ layout
    // due to causal masking is based on mma layout, the G2SQ loading row should also align with mma's row layout
    // (8, 8) with FullRow mma layout, each BN is owner by 4 lane
    static constexpr uint32_t G2SQColPerCTA = 4;
};

template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kPipelineReg,
    class WarpLayoutTraits_, typename DType, uint32_t kWarps=4
>
struct MMATraits {
    static constexpr uint32_t BM = kBM;
    static constexpr uint32_t BN = kBN;
    static constexpr uint32_t BK = kBK;
    static constexpr uint32_t PipelineK = kPipelineK;
    static constexpr uint32_t PipelineV = kPipelineV;

    static constexpr uint32_t G2SQColPerCTA = WarpLayoutTraits_::G2SQColPerCTA;
    static constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    static constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    static constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;

    // mma atom and layout
    using mma_op = std::conditional_t<
        std::is_same_v<DType, __nv_bfloat16>,
        cute::SM80_16x8x16_F32BF16BF16F32_TN,
        cute::SM80_16x8x16_F32F16F16F32_TN
    >;
    using mma_atom = cute::MMA_Atom<mma_op>;

    // mma QK layout
    using mma_qk_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<MMARow>{}, cute::Int<MMACol>{}, cute::_1{}),
        cute::make_step(cute::_2{}, cute::_1{}, cute::_0{})
    ));
    using mma_qk_permutations = cute::Tile<cute::Int<BM>, cute::Int<BN>, cute::_16>;

    using tiled_mma_qk = decltype(cute::make_tiled_mma(
        mma_atom{},
        mma_qk_layout{},
        mma_qk_permutations{}
    ));

    // mma PV layout
    using mma_pv_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<MMARow>{}, cute::Int<MMACol>{}, cute::_1{}),
        cute::make_step(cute::_2{}, cute::_1{}, cute::_0{})
    ));
    using mma_pv_permutations = cute::Tile<cute::Int<BM>, cute::Int<BK>, cute::_16>;

    using tiled_mma_pv = decltype(cute::make_tiled_mma(
        mma_atom{},
        mma_pv_layout{},
        mma_pv_permutations{}
    ));

    // smem layout Q, for gmem -> smem -> RF, smem -> RF ldmatrix for mma.sync layout
    using SwizzleLayoutAtomQ = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sQLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomQ{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));

    // smem layout K&V, head dim (BK) in [32, 64, 128] with 2-byte data type
    using SwizzleLayoutAtomK = std::conditional_t<
        BK == 32,
        decltype(cute::GMMA::Layout_K_SW64_Atom<DType>{}),
        decltype(cute::GMMA::Layout_K_SW128_Atom<DType>{})
    >;
    using sKLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomK{},
        cute::make_shape(cute::Int<BN>{}, cute::Int<BK>{}, cute::Int<PipelineK>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));

    using SwizzleLayoutAtomV = std::conditional_t<
        BK == 32,
        decltype(cute::GMMA::Layout_MN_SW64_Atom<DType>{}),
        decltype(cute::GMMA::Layout_MN_SW128_Atom<DType>{})
    >;
    using sVLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomV{},
        cute::make_shape(cute::Int<BK>{}, cute::Int<BN>{}, cute::Int<PipelineV>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));

    // smem layout O
    using SwizzleLayoutAtomO = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sOLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomO{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));

    using SharedStorageQ = SharedMemoryQ<DType, sQLayout, kWarps>;
    using SharedStorageKV = SharedMemoryKV<DType, sKLayout, sVLayout, PipelineK, PipelineV>;
    using SharedStorageO = SharedMemoryO<DType, sOLayout>;
    static constexpr size_t smem_size = cute::max(sizeof(SharedStorageQ), sizeof(SharedStorageKV), sizeof(SharedStorageO));

    // Query G2SQ copy with TV layout
    using g2sQ_copy_type = typename CopyWidthToType<16>::type;
    using g2sQ_copy_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2sQ_copy_type>, DType>;
    using g2sQ_tiled_copy_s = decltype(cute::make_tiled_copy(
        g2sQ_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<1>{}, cute::Int<G2SQColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using g2sQ_tiled_copy_g = decltype(cute::make_tiled_copy(
        g2sQ_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<G2SQColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));

    // Query S2RQ copy
    using s2rQ_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using s2rQ_tiled_copy = decltype(cute::make_tiled_copy_A(s2rQ_atom{}, tiled_mma_qk{}));

    // Key & Value G2S TMA copy box
    static constexpr uint32_t tma_k_iter = BK <= 64 ? 1 : BK / 64;
    static constexpr uint32_t tma_k_atom = BK <= 64 ? BK : 64;
    using g2sK_copy_box = decltype(cute::take<0, 2>(sKLayout{}));
    using g2sV_copy_box = decltype(cute::take<0, 2>(sVLayout{}));

    // Key & Value S2R ldmatrix copy and its tv layout
    using s2rK_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using s2rV_atom = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, DType>;
    using s2rK_tiled_copy = decltype(cute::make_tiled_copy_B(s2rK_atom{}, tiled_mma_qk{}));
    using s2rV_tiled_copy = decltype(cute::make_tiled_copy_B(s2rV_atom{}, tiled_mma_pv{}));

    // R2S copy and its tv layout
    using r2sO_atom = cute::Copy_Atom<cute::SM120_U32x4_STSM_N, DType>;
    using r2sO_tiled_copy = decltype(cute::make_tiled_copy_C(r2sO_atom{}, tiled_mma_pv{}));

    // S2G copy
    using s2gO_copy_type = typename CopyWidthToType<16>::type;
    using s2gO_copy_atom = cute::Copy_Atom<cute::UniversalCopy<s2gO_copy_type>, DType>;
    using s2gO_tiled_copy_s = decltype(cute::make_tiled_copy(
        s2gO_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<1>{}, cute::Int<G2SQColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using s2gO_tiled_copy_g = decltype(cute::make_tiled_copy(
        s2gO_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<1>{}, cute::Int<G2SQColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
};

template <typename TMADescK, typename TMADescV>
struct AttentionParams {
    const void* Q;
    const void* K;
    const void* V;
    const void* Mask;
    const void* Index;
    const void* Leftpad;
    void* O;
    uint32_t B;
    uint32_t Tq;
    uint32_t Tk;
    float scale;
    TMADescK tma_desc_k;
    TMADescV tma_desc_v;
};

// Skip metadata update func
template <
    bool kWarpSkipG2S, bool kWarpSkipMMA,
    uint32_t kMMARow, uint32_t kMMACol, uint32_t kMMAIter,
    class MaskTensor, class IndexTensor
>
__device__ __forceinline__ void update_skip_metadata(
    SkipHelper<kMMAIter, kWarpSkipG2S, kWarpSkipMMA>& skip_helper,
    const MaskTensor& mMask,
    const IndexTensor& mIndex,
    const uint32_t B,
    const uint32_t Tq,
    const uint32_t max_off_m,
    const uint32_t base_off_m,
    const uint32_t group_id,
    const uint32_t warp_id,
    const uint32_t lane_id,
    const uint32_t batch_id
) {
    using namespace cute;

    // Mask is sorted descending for each K-group. If the first row in this CTA
    // is inactive, every later row in the CTA is inactive as well.
    const uint8_t execute_cta = base_off_m < max_off_m
        ? static_cast<uint8_t>(mMask(make_coord(batch_id, base_off_m, group_id)) != 0)
        : 0;
    skip_helper.execute_cta = execute_cta;
    if (execute_cta == 0) {
        CUTE_UNROLL
        for (uint32_t i=0; i < kMMAIter * 2; ++i) {
            skip_helper.rMask[i] = 0;
            skip_helper.rIndex[i] = 0;
            if constexpr (kWarpSkipG2S) {
                skip_helper.execute_g2s_warp[i] = 0;
            }
        }
        if constexpr (kWarpSkipMMA) {
            CUTE_UNROLL
            for (uint32_t i=0; i < kMMAIter; ++i) {
                skip_helper.execute_mma_warp[i] = 0;
            }
        }
        return;
    }

    // The mask is sorted for each K-group. If the last row covered by this CTA
    // is active, every row in the CTA is active and per-row mask loads are
    // redundant. Keep the epilogue path unchanged by filling rMask with 1s.
    constexpr uint32_t kRowsCovered = kMMAIter * kMMARow * 16;
    const uint8_t full_tile = static_cast<uint8_t>((max_off_m - base_off_m) == kRowsCovered);
    const uint8_t full_cta = full_tile
        ? static_cast<uint8_t>(mMask(make_coord(batch_id, max_off_m - 1, group_id)) != 0)
        : 0;
    if (full_cta) {
        const uint8_t group_all_active = static_cast<uint8_t>(mMask(make_coord(batch_id, Tq - 1, group_id)) != 0);

        CUTE_UNROLL
        for (
            uint32_t i=0,
            off=base_off_m + (warp_id / kMMACol) * 16 + lane_id / 4;
            i < kMMAIter;
            ++i, off+=16 * kMMARow
        ) {
            skip_helper.rMask[i*2] = 1;
            skip_helper.rIndex[i*2] = group_all_active ? off : static_cast<uint32_t>(mIndex(make_coord(batch_id, off, group_id)));
            skip_helper.rMask[i*2+1] = 1;
            skip_helper.rIndex[i*2+1] = group_all_active ? off + 8 : static_cast<uint32_t>(mIndex(make_coord(batch_id, off + 8, group_id)));

            skip_helper.rIndex[i*2] = off < Tq ? skip_helper.rIndex[i*2] : 0;
            skip_helper.rIndex[i*2+1] = off + 8 < Tq ? skip_helper.rIndex[i*2+1] : 0;

            if constexpr (kWarpSkipG2S) {
                skip_helper.execute_g2s_warp[i*2] = 1;
                skip_helper.execute_g2s_warp[i*2+1] = 1;
            }
        }

        if constexpr (kWarpSkipMMA) {
            CUTE_UNROLL
            for (uint32_t i=0; i < kMMAIter; ++i) {
                skip_helper.execute_mma_warp[i] = 1;
            }
        }
        return;
    }

    CUTE_UNROLL
    for (
        uint32_t i=0,
        off=base_off_m + (warp_id / kMMACol) * 16 + lane_id / 4;
        i < kMMAIter;
        ++i, off+=16 * kMMARow
    ) {
        const uint8_t mask1 = off < max_off_m ? mMask(make_coord(batch_id, off, group_id)) : 0;
        skip_helper.rMask[i*2] = mask1;
        skip_helper.rIndex[i*2] = mask1 ? static_cast<uint32_t>(mIndex(make_coord(batch_id, off, group_id))) : 0;

        const uint8_t mask2 = off + 8 < max_off_m ? mMask(make_coord(batch_id, off + 8, group_id)) : 0;
        skip_helper.rMask[i*2+1] = mask2;
        skip_helper.rIndex[i*2+1] = mask2 ? static_cast<uint32_t>(mIndex(make_coord(batch_id, off + 8, group_id))) : 0;

        if constexpr (kWarpSkipG2S) {
            skip_helper.execute_g2s_warp[i*2] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask[i*2] > 0)));
            skip_helper.execute_g2s_warp[i*2+1] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask[i*2+1] > 0)));
        }
    }
    
    if constexpr (kWarpSkipMMA) {
        CUTE_UNROLL
        for (uint32_t i=0; i < kMMAIter; ++i) {
            skip_helper.execute_mma_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>((skip_helper.rMask[i*2] != 0) || (skip_helper.rMask[i*2+1] != 0))));
        }
    }
}

// QK fragment C -> PV fragment A convert
template <class PFrag, class AFrag, typename DType>
__device__ __forceinline__ void convert_QKrC_to_PVrA(PFrag const& rP, AFrag& rPlA) {
    using namespace cute;
    CUTE_UNROLL
    for (uint32_t m_rest = 0; m_rest < size<1>(rP); ++m_rest) {
        CUTE_UNROLL
        for (uint32_t m_atom = 0; m_atom < size<0,1>(rP); ++m_atom) {
            CUTE_UNROLL
            for (uint32_t n_hi = 0; n_hi < size<2>(rP); ++n_hi) {
                uint32_t a_k_atom = n_hi & 1;
                uint32_t k_rest = n_hi >> 1;

                auto& dst = rPlA(make_coord(_0{}, m_atom, a_k_atom), m_rest, k_rest);
                *reinterpret_cast<uint32_t*>(&dst) = Float22Half2<DType>()(
                    rP(make_coord(_0{}, m_atom), m_rest, n_hi),
                    rP(make_coord(_1{}, m_atom), m_rest, n_hi)
                );
            }
        }
    }
}

// main kernel func
template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG, bool kIsCausal, bool kIsLeftpad,
    uint32_t kBM, uint32_t kBN, uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kPipelineReg,
    typename TMADescK, typename TMADescV, class WarpLayoutTraits_, class MMATraits_,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA, typename DType, uint32_t kWarps=4
>
__global__ void attention_head_v1_kernel(const __grid_constant__ AttentionParams<TMADescK, TMADescV> params) {
    using namespace cute;
    const auto& [Q, K, V, Mask, Index, Leftpad, O, B, Tq, Tk, scale, tma_desc_k, tma_desc_v] = params;
    constexpr uint32_t Hq = kHq;
    constexpr uint32_t Hk = kHk;
    constexpr uint32_t D = kD;
    constexpr uint32_t NG = kNG;
    constexpr bool IsCausal = kIsCausal;
    constexpr bool IsLeftpad = kIsLeftpad;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t PipelineK = kPipelineK;
    constexpr uint32_t PipelineV = kPipelineV;
    constexpr uint32_t PipelineReg = kPipelineReg;

    constexpr uint32_t G2SQColPerCTA = WarpLayoutTraits_::G2SQColPerCTA;
    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;
    constexpr uint32_t MMAIter = WarpLayoutTraits_::MMAIter;

    using tiled_mma_qk = typename MMATraits_::tiled_mma_qk;
    using tiled_mma_pv = typename MMATraits_::tiled_mma_pv;
    using sQLayout = typename MMATraits_::sQLayout;
    using sKLayout = typename MMATraits_::sKLayout;
    using sVLayout = typename MMATraits_::sVLayout;
    using sOLayout = typename MMATraits_::sOLayout;
    using g2sQ_tiled_copy_s = typename MMATraits_::g2sQ_tiled_copy_s;
    using g2sQ_tiled_copy_g = typename MMATraits_::g2sQ_tiled_copy_g;
    using s2rQ_tiled_copy = typename MMATraits_::s2rQ_tiled_copy;
    using s2rK_tiled_copy = typename MMATraits_::s2rK_tiled_copy;
    using s2rV_tiled_copy = typename MMATraits_::s2rV_tiled_copy;
    using r2sO_tiled_copy = typename MMATraits_::r2sO_tiled_copy;
    using s2gO_tiled_copy_s = typename MMATraits_::s2gO_tiled_copy_s;
    using s2gO_tiled_copy_g = typename MMATraits_::s2gO_tiled_copy_g;

    // grid: (cdiv(Tq, BM), Hq, B)
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    constexpr uint32_t kv_group_size = Hq / Hk;
    constexpr uint32_t group_size = Hq / NG;
    const uint32_t query_head = bidy;
    const uint32_t key_head = bidy / kv_group_size;
    const uint32_t group_id = query_head / group_size;
    const float scale_ln2 = scale * 1.44269504;
    constexpr float lazy_rescale_threshold = 8;

    const uint32_t base_off_m = bidx * BM;
    const uint32_t thread_off_m_1 = (warp_id / MMACol) * 16 + lane_id / 4;
    const uint32_t thread_off_m_2 = thread_off_m_1 + 8; // CTA-local row, aligned with mma layout
    const uint32_t max_off_m = cute::min(Tq, base_off_m + BM);
    constexpr uint32_t MMARowPerCTA = MMARow * 16;

    //
    // prepare tensors and smem layout
    //
    Tensor mQ = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(Q)),
        make_ordered_layout(
            make_shape(Tq, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mK = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(K)),
        make_ordered_layout(
            make_shape(Tk, Int<D>{}, Int<Hk>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mV = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(V)),
        make_ordered_layout(
            make_shape(Tk, Int<D>{}, Int<Hk>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mO = make_tensor(
        make_gmem_ptr(static_cast<DType*>(O)),
        make_ordered_layout(
            make_shape(Tq, Int<D>{}, Int<Hq>{}, B),
            make_step(_2{}, _0{}, _1{}, _3{})
        )
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_layout(make_shape(B, Tq, Int<NG>{}), LayoutRight{})
    );
    Tensor mIndex = make_tensor(
        make_gmem_ptr(static_cast<const int64_t*>(Index)),
        make_layout(make_shape(B, Tq, Int<NG>{}), LayoutRight{})
    );
    Tensor mLeftpad = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(Leftpad)),
        make_layout(make_shape(B), LayoutRight{})
    );

    extern __shared__ uint8_t shared_memory[];
    using SharedStorageKV = SharedMemoryKV<DType, sKLayout, sVLayout, PipelineK, PipelineV>;
    SharedStorageKV &smem_kv = *reinterpret_cast<SharedStorageKV*>(shared_memory);

    using SharedStorageQ = SharedMemoryQ<DType, sQLayout, kWarps>;
    using SharedStorageO = SharedMemoryO<DType, sOLayout>;
    SharedStorageQ &smem_q = *reinterpret_cast<SharedStorageQ*>(shared_memory);
    SharedStorageO &smem_o = *reinterpret_cast<SharedStorageO*>(shared_memory);

    Tensor sK = make_tensor(
        make_smem_ptr(smem_kv.K.begin()),
        sKLayout{}
    );
    Tensor sV = make_tensor(
        make_smem_ptr(smem_kv.V.begin()),
        sVLayout{}
    );
    Tensor sQ = make_tensor(
        make_smem_ptr(smem_q.Q.begin()),
        sQLayout{}
    );
    Tensor sO = make_tensor(
        make_smem_ptr(smem_o.O.begin()),
        sOLayout{}
    );

    // init barrier
    if (tidx == 0) {
        CUTE_UNROLL
        for (uint32_t i=0; i < PipelineK; ++i) {
            cute::initialize_barrier(smem_kv.tma_k_barrier[i]);
        }
        CUTE_UNROLL
        for (uint32_t i=0; i < PipelineV; ++i) {
            cute::initialize_barrier(smem_kv.tma_v_barrier[i]);
        }
    }
    __syncthreads();

    // load mask & index
    SkipHelper<MMAIter, kWarpSkipG2S, kWarpSkipMMA> skip_helper;
    update_skip_metadata<kWarpSkipG2S, kWarpSkipMMA, MMARow, MMACol, MMAIter, decltype(mMask), decltype(mIndex)>(
        skip_helper, mMask, mIndex, B, Tq, max_off_m, base_off_m, group_id, warp_id, lane_id, bidz
    );
    if (skip_helper.execute_cta == 0) return;

    uint32_t producer_k = 0;
    uint32_t producer_v = 0;
    uint32_t consumer = 0;
    uint32_t producer_stage_k = 0;
    uint32_t producer_stage_v = 0;
    uint32_t consumer_stage_k = 0;
    uint32_t consumer_stage_v = 0;

    //
    // prologue, load Q from gmem -> smem -> RF
    //
    Tensor gQ = local_tile(
        mQ,
        make_tile(_1{}, Int<D>{}),
        make_coord(_, 0, query_head, bidz)
    );
    auto g2sQ_thr_copy_g = g2sQ_tiled_copy_g{}.get_slice(tidx % G2SQColPerCTA);
    auto g2sQ_thr_copy_s = g2sQ_tiled_copy_s{}.get_slice(tidx % G2SQColPerCTA);
    auto pSgQ = g2sQ_thr_copy_g.partition_S(gQ);
    auto pDsQ = g2sQ_thr_copy_s.partition_D(sQ);

    CUTE_UNROLL
    for (uint32_t i=0, off1=thread_off_m_1, off2=thread_off_m_2; i < MMAIter; ++i, off1+=MMARowPerCTA, off2+=MMARowPerCTA) {
        if constexpr (kThrSkipG2S) {
            if (skip_helper.rMask[i*2]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            if (skip_helper.rMask[i*2+1]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
        else if constexpr (kWarpSkipG2S) {
            if (skip_helper.execute_g2s_warp[i*2]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            if (skip_helper.execute_g2s_warp[i*2+1]) copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
        else {
            copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2]), pDsQ(_, off1, _));
            copy(g2sQ_tiled_copy_g{}, pSgQ(_, _0{}, _, skip_helper.rIndex[i*2+1]), pDsQ(_, off2, _));
        }
    }
    cp_async_fence();

    // fill the pipeline for K & V
    // 1. get Tk start and end
    uint32_t key_start = 0;
    uint32_t key_end = Tk;
    if constexpr (IsLeftpad) {key_start = mLeftpad(make_coord(bidz));}
    uint32_t cta_min_query = 0;
    if constexpr (IsCausal) {
        // Count CTA query range from the real sorted row indices. Tiles that
        // end before cta_min_query are fully causal-valid for every active row.
        uint32_t local_max_query = 0;
        uint32_t local_min_query = Tk;
        CUTE_UNROLL
        for (uint32_t i=0; i < MMAIter*2; ++i) {
            if (skip_helper.rMask[i]) {
                local_max_query = cute::max(local_max_query, skip_helper.rIndex[i]);
                local_min_query = cute::min(local_min_query, skip_helper.rIndex[i]);
            }
        }
        uint32_t warp_max_query = __reduce_max_sync(0xffffffff, local_max_query);
        uint32_t warp_min_query = __reduce_min_sync(0xffffffff, local_min_query);

        if (lane_id == 0) {
            smem_q.cta_reduce_cache[warp_id] = warp_max_query;
            smem_q.cta_reduce_cache[kWarps + warp_id] = warp_min_query;
        }
        __syncthreads();

        uint32_t x_max = 0;
        uint32_t x_min = Tk;
        if (warp_id == 0 && lane_id < kWarps) {
            x_max = smem_q.cta_reduce_cache[lane_id];
            x_min = smem_q.cta_reduce_cache[kWarps + lane_id];
        }
        if (warp_id == 0) {
            x_max = __reduce_max_sync(0xffffffff, x_max);
            x_min = __reduce_min_sync(0xffffffff, x_min);
            if (lane_id == 0) {
                smem_q.cta_reduce_cache[kWarps * 2] = x_max;
                smem_q.cta_reduce_cache[kWarps * 2 + 1] = x_min;
            }
        }
        __syncthreads();
        const uint32_t cur_max_query = smem_q.cta_reduce_cache[kWarps * 2];
        cta_min_query = smem_q.cta_reduce_cache[kWarps * 2 + 1];
        key_end = cute::min(Tk, cur_max_query + 1);
    }

    // K & V load helper
    auto tma_mK = tma_desc_k.get_tma_tensor(make_shape(Tk, Int<D>{}, Int<Hk>{}, B));
    auto tma_mV = tma_desc_v.get_tma_tensor(make_shape(Int<D>{}, Tk, Int<Hk>{}, B));

    auto tma_k_thr = tma_desc_k.get_slice(_0{});
    auto tma_v_thr = tma_desc_v.get_slice(_0{});

    uint32_t k_phase[PipelineK] = {};
    uint32_t v_phase[PipelineV] = {};

    auto gK = local_tile(
        tma_mK,
        make_tile(Int<BN>{}, Int<D>{}),
        make_coord(_, _0{}, key_head, bidz)
    );
    auto gV = local_tile(
        tma_mV,
        make_tile(Int<D>{}, Int<BN>{}),
        make_coord(_0{}, _, key_head, bidz)
    );
    auto pSgK = tma_k_thr.partition_S(gK);
    auto pDsK = tma_k_thr.partition_D(sK);
    auto pSgV = tma_v_thr.partition_S(gV);
    auto pDsV = tma_v_thr.partition_D(sV);

    const uint32_t key_start_tile = key_start / BN;
    const uint32_t key_end_tile = cute::ceil_div(key_end, BN);
    const uint32_t key_tile_num = key_end_tile - key_start_tile;
    constexpr size_t kv_load_bytes = BN * D * sizeof(DType);

    auto tma_issue_k = [&](uint32_t tile_pos) {
        if (warp_id == 0 && cute::elect_one_sync()) {
            set_barrier_transaction_bytes(
                smem_kv.tma_k_barrier[producer_stage_k],
                kv_load_bytes
            );
            copy(
                tma_desc_k.with(smem_kv.tma_k_barrier[producer_stage_k]),
                pSgK(_, _0{}, _0{}, tile_pos), pDsK(_, _0{}, _0{}, producer_stage_k)
            );
        }
    };
    auto tma_issue_v = [&](uint32_t tile_pos) {
        if (warp_id == 0 && cute::elect_one_sync()) {
            set_barrier_transaction_bytes(
                smem_kv.tma_v_barrier[producer_stage_v],
                kv_load_bytes
            );
            copy(
                tma_desc_v.with(smem_kv.tma_v_barrier[producer_stage_v]),
                pSgV(_, _0{}, _0{}, tile_pos), pDsV(_, _0{}, _0{}, producer_stage_v)
            );
        }
    };

    // wait gmem -> smem, then LDSM to rQ
    ThrMMA thr_mma_qk = tiled_mma_qk{}.get_slice(tidx);
    ThrMMA thr_mma_pv = tiled_mma_pv{}.get_slice(tidx);
    auto rQ = thr_mma_qk.partition_fragment_A(sQ);
    auto rK = thr_mma_qk.partition_fragment_B(sK(_, _, _0{}));
    auto rV = thr_mma_pv.partition_fragment_B(sV(_, _, _0{}));

    Tensor rP_logical = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    auto rP = thr_mma_qk.partition_fragment_C(rP_logical);
    auto cP = thr_mma_qk.partition_C(rP_logical);

    auto rP_operand = make_tensor(
        static_cast<DType*>(nullptr),
        make_layout(make_shape(Int<BM>{}, Int<BN>{}), LayoutRight{})
    );
    auto rPlA = thr_mma_pv.partition_fragment_A(rP_operand);

    Tensor gO_tile = local_tile(
        mO,
        make_tile(Int<BM>{}, Int<D>{}),
        make_coord(bidx, _0{}, query_head, bidz)
    );
    auto rO = thr_mma_pv.partition_fragment_C(gO_tile); // acc fragment
    clear(rO);

    auto rM = make_fragment_like<float>(rP(make_coord(_0{}, _), _, _0{}));
    auto rL = make_fragment_like<float>(rP(make_coord(_0{}, _), _, _0{}));
    fill(rM, -MAXFLOAT);
    clear(rL);

    auto s2rQ_thr_copy = s2rQ_tiled_copy{}.get_slice(tidx);
    auto pSsQ = s2rQ_thr_copy.partition_S(sQ);
    auto pDrQ = s2rQ_thr_copy.retile_D(rQ);

    auto s2rK_thr_copy = s2rK_tiled_copy{}.get_slice(tidx);
    auto pSsK = s2rK_thr_copy.partition_S(sK);
    auto pDrK = s2rK_thr_copy.retile_D(rK);

    auto s2rV_thr_copy = s2rV_tiled_copy{}.get_slice(tidx);
    auto pSsV = s2rV_thr_copy.partition_S(sV);
    auto pDrV = s2rV_thr_copy.retile_D(rV);

    cp_async_wait<0>();
    __syncthreads();
    copy(s2rQ_tiled_copy{}, pSsQ, pDrQ);
    __syncthreads();

    // Q * scale
    // f16 version
    auto scale2_rn = Float22Half2<DType, true>()(scale_ln2, scale_ln2);
    if constexpr (is_same_v<DType, __half>) {
        auto rQ_pack = recast<__half2>(rQ);
        for (uint32_t i=0; i < size(rQ_pack); ++i) {
            rQ_pack(i) = __hmul2_rn(rQ_pack(i), scale2_rn);
        }
    }
    else {
        auto rQ_pack = recast<__nv_bfloat162>(rQ);
        for (uint32_t i=0; i < size(rQ_pack); ++i) {
            rQ_pack(i) = __hmul2_rn(rQ_pack(i), scale2_rn);
        }
    }

    // float version
    // if constexpr (is_same_v<DType, __half>) {
    //     auto rQ_cast = recast<__half>(rQ);
    //     auto scale_cast = __float2half_rn(scale_ln2);
    //     CUTE_UNROLL
    //     for (uint32_t i=0; i < size(rQ_cast); ++i) rQ_cast(i) = rQ_cast(i) * scale_cast;
    // }
    // else {
    //      auto rQ_cast = recast<__nv_bfloat16>(rQ);
    //     auto scale_cast = __float2bfloat16_rn(scale_ln2);
    //     CUTE_UNROLL
    //     for (uint32_t i=0; i < size(rQ_cast); ++i) rQ_cast(i) = rQ_cast(i) * scale_cast;
    // }

    // if (DEBUG_COND) {
        // print(rQ); print("\n");
        // print(rL); print("\n");
        // print(rP); print("\n");
    // }

    // fill the pipeline for K & V
    CUTE_UNROLL
    for (int32_t p=0; p < cute::max(PipelineK, PipelineV) - 1; ++p) {
        if (producer_k < key_tile_num && p < PipelineK - 1) {
            tma_issue_k(producer_k + key_start_tile);
            producer_k++;
            producer_stage_k = producer_k % PipelineK;
        }
        if (producer_v < key_tile_num && p < PipelineV - 1) {
            tma_issue_v(producer_v + key_start_tile);
            producer_v++;
            producer_stage_v = producer_v % PipelineV;
        }
    }

    // mainloop
    CUTE_NO_UNROLL
    for (int32_t k_tile=0; k_tile < key_tile_num; ++k_tile) {
        if (producer_k < key_tile_num) {
            tma_issue_k(producer_k + key_start_tile);
            producer_k++;
            producer_stage_k = producer_k % PipelineK;
        }
        if (producer_v < key_tile_num) {
            tma_issue_v(producer_v + key_start_tile);
            producer_v++;
            producer_stage_v = producer_v % PipelineV;
        }

        const uint32_t tile_start = (key_start_tile + k_tile) * BN;
        const uint32_t tile_end = tile_start + BN;
        bool tile_all_valid = (tile_start >= key_start) && (tile_end <= key_end);
        if constexpr (IsCausal) {
            tile_all_valid &= (tile_end - 1 <= cta_min_query);
        }

        // wait until consumer's K block arrive
        wait_barrier(smem_kv.tma_k_barrier[consumer_stage_k], k_phase[consumer_stage_k]);
        k_phase[consumer_stage_k] ^= 1;
        __syncthreads();

        // S2R for Key, then Q@K^T
        clear(rP);
        copy(s2rK_tiled_copy{}, pSsK(_, _, _, make_coord(0, consumer_stage_k)), pDrK);
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rQ); ++i) {
            if constexpr (kWarpSkipMMA) {
                if (skip_helper.execute_mma_warp[i]) {
                    CUTE_UNROLL
                    for (uint32_t j=0; j < size<1>(rK); ++j) {
                        CUTE_UNROLL
                        for (uint32_t k=0; k < size<2>(rK); ++k) {
                            gemm(tiled_mma_qk{}, rQ(_, i, k), rK(_, j, k), rP(_, i, j));
                        }
                    }
                }
            }
            else {
                CUTE_UNROLL
                for (uint32_t j=0; j < size<1>(rK); ++j) {
                    CUTE_UNROLL
                    for (uint32_t k=0; k < size<2>(rK); ++k) {
                        gemm(tiled_mma_qk{}, rQ(_, i, k), rK(_, j, k), rP(_, i, j));
                    }
                }
            }
        }
        consumer_stage_k = (consumer_stage_k + 1) % PipelineK;
        
        // row-wise apply mask & partial softmax update
        CUTE_UNROLL
        for (uint32_t mi=0; mi < size<1>(rP); ++mi) {
            CUTE_UNROLL
            for (uint32_t mma_i=0; mma_i < size<0>(rP) / 2; ++mma_i) {
                float rm = -MAXFLOAT;
                const uint32_t local_row = mi * 2 + mma_i;
                CUTE_UNROLL
                for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                    if (!tile_all_valid) {
                        auto mn = cP(mma_i * 2, mi, ni);
                        uint32_t m = get<0>(mn);
                        uint32_t n = get<1>(mn);
                        uint32_t abs_q_pos = skip_helper.rIndex[mi * 2 + mma_i];
                        uint32_t abs_k_pos = tile_start + n;

                        bool valid1 = (abs_k_pos >= key_start) && (abs_k_pos < key_end);
                        bool valid2 = (abs_k_pos + 1 >= key_start) && (abs_k_pos + 1 < key_end);
                        // if causal
                        if constexpr (IsCausal) {
                            valid1 &= (abs_k_pos <= abs_q_pos);
                            valid2 &= (abs_k_pos + 1 <= abs_q_pos);
                        }
                        rP(mma_i * 2, mi, ni) = valid1 ? rP(mma_i * 2, mi, ni) : -MAXFLOAT;
                        rP(mma_i * 2 + 1, mi, ni) = valid2 ? rP(mma_i * 2 + 1, mi, ni) : -MAXFLOAT;
                    }

                    // update max
                    rm = cute::max(rm, cute::max(rP(mma_i * 2, mi, ni), rP(mma_i * 2 + 1, mi, ni)));
                }

                rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 1, 4));
                rm = cute::max(rm, __shfl_xor_sync(0xffffffff, rm, 2, 4));
                float old_max = rM(local_row);
                float new_max = cute::max(rm, old_max);
                float row_max = new_max;
                float acc_scale = 1.0f;

                // FA4 lazy rescale
                bool do_rescale = false;
                if (old_max != -MAXFLOAT) {
                    if (new_max - old_max <= lazy_rescale_threshold) {
                        row_max = old_max;
                    }
                    else {
                        acc_scale = exp2f(old_max - new_max);
                        rM(local_row) = new_max;
                        do_rescale = true;
                    }
                }
                else {
                    rM(local_row) = new_max;
                }

                if (do_rescale) {
                    rL(local_row) *= acc_scale;
                    CUTE_UNROLL
                    for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                        rO(local_row, mi, ni) *= acc_scale;
                    }
                }

                float rl = 0.0f;
                float safe_max = (row_max == -MAXFLOAT) ? 0.0f : row_max;
                CUTE_UNROLL
                for (uint32_t ni=0; ni < size<2>(rP); ++ni) {
                    float rp0 = exp2f(rP(mma_i * 2, mi, ni) - safe_max);
                    float rp1 = exp2f(rP(mma_i * 2 + 1, mi, ni) - safe_max);

                    rP(mma_i * 2, mi, ni) = rp0;
                    rP(mma_i * 2 + 1, mi, ni) = rp1;
                    rl += rp0 + rp1;
                }
                rl += __shfl_xor_sync(0xffffffff, rl, 1, 4);
                rl += __shfl_xor_sync(0xffffffff, rl, 2, 4);
                rL(local_row) += rl;
            }
        }
        // P@V mma
        // convert QK-C layout to PV-A layout
        convert_QKrC_to_PVrA<decltype(rP), decltype(rPlA), DType>(rP, rPlA);

        wait_barrier(smem_kv.tma_v_barrier[consumer_stage_v], v_phase[consumer_stage_v]);
        v_phase[consumer_stage_v] ^= 1;
        __syncthreads();

        // if (DEBUG_COND && k_tile == 0) {
        //     print(rP); print("\n");
        //     print(rPlA); print("\n");
        // }

        copy(s2rV_tiled_copy{}, pSsV(_, _, _, make_coord(0, consumer_stage_v)), pDrV);
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rPlA); ++i) {
            if constexpr (kWarpSkipMMA) {
                if (skip_helper.execute_mma_warp[i]) {
                    CUTE_UNROLL
                    for (uint32_t j=0; j < size<1>(rV); ++j) {
                        CUTE_UNROLL
                        for (uint32_t k=0; k < size<2>(rV); ++k) {
                            gemm(tiled_mma_pv{}, rPlA(_, i, k), rV(_, j, k), rO(_, i, j));
                        }
                    }
                }
            }
            else {
                CUTE_UNROLL
                for (uint32_t j=0; j < size<1>(rV); ++j) {
                    CUTE_UNROLL
                    for (uint32_t k=0; k < size<2>(rV); ++k) {
                        gemm(tiled_mma_pv{}, rPlA(_, i, k), rV(_, j, k), rO(_, i, j));
                    }
                }
            }
        }
        consumer_stage_v = (consumer_stage_v + 1) % PipelineV;
    }
    // epilogue
    CUTE_UNROLL
    for (uint32_t mi=0; mi < size<1>(rO); ++mi) {
        CUTE_UNROLL
        for (uint32_t mma_i=0; mma_i < size<0>(rO) / 2; ++mma_i) {
            CUTE_UNROLL
            for (uint32_t ni=0; ni < size<2>(rO); ++ni) {
                rO(mma_i * 2, mi, ni) *= __frcp_rn(rL(mi * 2 + mma_i));
                rO(mma_i * 2 + 1, mi, ni) *= __frcp_rn(rL(mi * 2 + mma_i));
            }
        }
    }

    // write back
    auto r2sO_thr_copy = r2sO_tiled_copy{}.get_slice(tidx);
    auto pSrO = r2sO_thr_copy.retile_S(rO);
    auto pDsO = r2sO_thr_copy.partition_D(sO);
    auto pSrO_cast = make_tensor_like<DType>(pSrO);
    copy(pSrO, pSrO_cast);
    copy(r2sO_tiled_copy{}, pSrO_cast, pDsO);
    __syncthreads();

    auto gO_epi = local_tile(
        mO,
        make_tile(_1{}, Int<D>{}),
        make_coord(_, _0{}, query_head, bidz)
    );
    auto s2gO_thr_copy_g = s2gO_tiled_copy_g{}.get_slice(tidx % G2SQColPerCTA);
    auto s2gO_thr_copy_s = s2gO_tiled_copy_s{}.get_slice(tidx % G2SQColPerCTA);
    auto pSsO = s2gO_thr_copy_s.partition_S(sO);
    auto pDgO = s2gO_thr_copy_g.partition_D(gO_epi);
    CUTE_UNROLL
    for (uint32_t i=0, off1=thread_off_m_1, off2=thread_off_m_2; i < MMAIter; ++i, off1+=MMARowPerCTA, off2+=MMARowPerCTA) {
        if (skip_helper.rMask[i*2]) copy(s2gO_tiled_copy_g{}, pSsO(_, off1, _), pDgO(_, _0{}, _, skip_helper.rIndex[i*2]));
        if (skip_helper.rMask[i*2+1]) copy(s2gO_tiled_copy_g{}, pSsO(_, off2, _), pDgO(_, _0{}, _, skip_helper.rIndex[i*2+1]));
    }
}

template <uint32_t H, uint32_t D, typename KBox, typename VBox, typename DType>
static auto make_tma_copy_in_host(
    const void* K, const void* V,
    const uint32_t B, const uint32_t T,
    const KBox k_copy_box, const VBox v_copy_box
) {
    using namespace cute;
    using TmaDType = TmaDescriptorDTypeT<DType>;
    auto k_gmem_shape = make_shape(T, Int<D>{}, Int<H>{}, B);
    auto k_gmem_stride = make_stride(Int<H * D>{}, _1{}, Int<D>{}, T * H * D);
    auto v_gmem_shape = make_shape(Int<D>{}, T, Int<H>{}, B);
    auto v_gmem_stride = make_stride(_1{}, Int<H * D>{}, Int<D>{}, T * H * D);
    auto k_layout = make_layout(k_gmem_shape, k_gmem_stride);
    auto v_layout = make_layout(v_gmem_shape, v_gmem_stride);
    auto k_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<const TmaDType*>(K)), k_layout);
    auto v_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<const TmaDType*>(V)), v_layout);

    auto tma_load_k = make_tma_copy(SM90_TMA_LOAD{}, k_tensor, k_copy_box);
    auto tma_load_v = make_tma_copy(SM90_TMA_LOAD{}, v_tensor, v_copy_box);

    return cute::make_tuple(tma_load_k, tma_load_v);
}

template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG,
    uint32_t kBM, uint32_t kBN, uint32_t kPipelineK, uint32_t kPipelineV,
    bool kIsCausal, bool kIsLeftpad,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA,
    bool kUsePDL, typename DType
>
struct Attention_Head_V1_Host {
    static void run(
        const tvm::ffi::TensorView Q,
        const tvm::ffi::TensorView K,
        const tvm::ffi::TensorView V,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView Index,
        const tvm::ffi::TensorView Leftpad,
        const tvm::ffi::TensorView O,
        const float scale,
        const float sparsity
    ) {
        using namespace host;
        RuntimeCheck(
            sparsity >= 0 && sparsity <= 1,
            "sparsity must be in [0, 1]"
        );

        auto B = SymbolicSize{"batch_size"};
        auto Tq = SymbolicSize{"query_num_tokens"};
        auto Tk = SymbolicSize{"key_num_tokens"};
        auto Hq = SymbolicSize{"query_num_heads"};
        auto Hk = SymbolicSize{"key_num_heads"};
        auto D = SymbolicSize{"head_dim"};
        auto NG = SymbolicSize{"num_groups"};
        auto device = SymbolicDevice{};

        Hq.set_value(kHq); Hk.set_value(kHk); D.set_value(kD); NG.set_value(kNG);
        device.set_options<kDLCUDA>();

        // host-side checking
        TensorMatcher({B, Tq, Hq, D}) //
            .with_dtype<DType>()
            .with_device(device)
            .verify(Q).verify(O);
        
        TensorMatcher({B, Tk, Hk, D}) //
            .with_dtype<DType>()
            .with_device(device)
            .verify(K).verify(V);
        
        TensorMatcher({B, Tq, NG}) //
            .with_device(device)
            .verify(Mask);
        
        TensorMatcher({B, Tq, NG}) //
            .with_dtype<int64_t>()
            .with_device(device)
            .verify(Index);
        
        TensorMatcher({B}) //
            .with_dtype<uint32_t>()
            .with_device(device)
            .verify(Leftpad);
        
        RuntimeCheck(
            sizeof(DType) == 2,
            "DType must be fp16 or bf16"
        );

        const auto batch_size = static_cast<uint32_t>(B.unwrap());
        const auto num_query_tokens = static_cast<uint32_t>(Tq.unwrap());
        const auto num_key_tokens = static_cast<uint32_t>(Tk.unwrap());
        RuntimeCheck(
            kD % 32 == 0,
            "D must be multiple of 32"
        );
        RuntimeCheck(
            kHq % kHk == 0,
            "query head num must be multiple of key&value head num"
        );
        RuntimeCheck(
            kHq % kNG  == 0,
            "pruning group must be based on num query heads"
        );

        // host-side static tiling
        using warp_layout_traits = WarpLayoutTraits<kBM, kBN, kD, 4, sizeof(DType)>;
        using mma_traits = MMATraits<kBM, kBN, kD, kPipelineK, kPipelineV, 2, warp_layout_traits, DType>;
        const auto &[tma_desc_k, tma_desc_v] = make_tma_copy_in_host<
            kHk, kD,
            typename mma_traits::g2sK_copy_box,
            typename mma_traits::g2sV_copy_box,
            DType
        >(
            K.data_ptr(),
            V.data_ptr(),
            batch_size,
            num_key_tokens,
            typename mma_traits::g2sK_copy_box{},
            typename mma_traits::g2sV_copy_box{}
        );
        
        const auto params = AttentionParams<decltype(tma_desc_k), decltype(tma_desc_v)>{
            .Q = Q.data_ptr(),
            .K = K.data_ptr(),
            .V = V.data_ptr(),
            .Mask = Mask.data_ptr(),
            .Index = Index.data_ptr(),
            .Leftpad = Leftpad.data_ptr(),
            .O = O.data_ptr(),
            .B = batch_size,
            .Tq = num_query_tokens,
            .Tk = num_key_tokens,
            .scale = scale,
            .tma_desc_k = tma_desc_k,
            .tma_desc_v = tma_desc_v
        };

        constexpr auto kernel = attention_head_v1_kernel<
            kHq, kHk, kD, kNG, kIsCausal, kIsLeftpad,
            kBM, kBN, kPipelineK, kPipelineV, 2, decltype(tma_desc_k), decltype(tma_desc_v),
            warp_layout_traits, mma_traits,
            kThrSkipG2S, kWarpSkipG2S, kWarpSkipMMA, DType
        >;

        const dim3 grid_size = {cute::ceil_div(params.Tq, kBM), kHq, params.B};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
        LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
    }
};

} // namespace