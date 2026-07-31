#pragma once
// cute
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <type_traits>
#include "utils.cuh"


#define DEBUG_COND (tidx == 0 && bidx + bidy + bidz == 0)

// layout for smem
template <typename DType, class LayoutQ, bool IsReduce=false, uint32_t kWarps=4>
struct SharedMemoryQ;

template <typename DType, class LayoutQ, uint32_t kWarps>
struct SharedMemoryQ<DType, LayoutQ, false, kWarps> {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutQ>> Q;
};
template <typename DType, class LayoutQ, uint32_t kWarps>
struct SharedMemoryQ<DType, LayoutQ, true, kWarps> {
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
    class WarpLayoutTraits_, bool kIsCausal, typename DType, uint32_t kWarps=4, bool kUseTMA=true
>
struct MMATraits;

template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kPipelineReg,
    class WarpLayoutTraits_, bool kIsCausal, typename DType, uint32_t kWarps
>
struct MMATraits<kBM, kBN, kBK, kPipelineK, kPipelineV, kPipelineReg, WarpLayoutTraits_, kIsCausal, DType, kWarps, true> {
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

    using SharedStorageQ = SharedMemoryQ<DType, sQLayout, kIsCausal, kWarps>;
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
            skip_helper.rIndex[i*2] = group_all_active ? off : mIndex(make_coord(batch_id, off, group_id));
            skip_helper.rMask[i*2+1] = 1;
            skip_helper.rIndex[i*2+1] = group_all_active ? off + 8 : mIndex(make_coord(batch_id, off + 8, group_id));

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
        skip_helper.rIndex[i*2] = mask1 ? mIndex(make_coord(batch_id, off, group_id)) : 0;

        const uint8_t mask2 = off + 8 < max_off_m ? mMask(make_coord(batch_id, off + 8, group_id)) : 0;
        skip_helper.rMask[i*2+1] = mask2;
        skip_helper.rIndex[i*2+1] = mask2 ? mIndex(make_coord(batch_id, off + 8, group_id)) : 0;

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