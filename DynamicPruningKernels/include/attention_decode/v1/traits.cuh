#pragma once
// cute
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <type_traits>
#include "utils.cuh"

#define DEBUG_COND (tidx == 0 && bidx + bidy + bidz == 0)

// layout for smem
template <typename DType, class LayoutQ>
struct SharedMemoryQ {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutQ>> Q;
};

template <typename DType, class LayoutK, class LayoutV, uint32_t PipelineK, uint32_t PipelineV, bool IsCTAReduce=false, uint32_t kReduceWarp=4>
struct SharedMemoryKV;

template <typename DType, class LayoutK, class LayoutV, uint32_t PipelineK, uint32_t PipelineV, uint32_t kReduceWarp>
struct SharedMemoryKV<DType, LayoutK, LayoutV, PipelineK, PipelineV, false, kReduceWarp> {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutK>> K;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutV>> V;

    alignas(16) uint64_t tma_k_barrier[PipelineK];
    alignas(16) uint64_t tma_v_barrier[PipelineV];
};
template <typename DType, class LayoutK, class LayoutV, uint32_t PipelineK, uint32_t PipelineV, uint32_t kReduceWarp>
struct SharedMemoryKV<DType, LayoutK, LayoutV, PipelineK, PipelineV, true, kReduceWarp> {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutK>> K;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutV>> V;

    alignas(16) uint64_t tma_k_barrier[PipelineK];
    alignas(16) uint64_t tma_v_barrier[PipelineV];
    alignas(16) float cta_reduce_cache[8 * kReduceWarp];
};


template <typename DType, class LayoutO>
struct SharedMemoryO {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutO>> O;
};

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kWarps=4, uint32_t kBytes=2>
struct WarpLayoutTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;

    // mma warp layout
    // mma perm at least to be (MMARow * 16, MMACol * 8 * 2), enable s2rB copy use LDSMx4
    // using (1, Warp) layout for GQA packed GEMM, where BM <= 16
    static constexpr uint32_t MMACol = kWarps;
    static constexpr uint32_t MMARow = 1;
    static constexpr uint32_t MMAIter = 1;

    // infer the G2SQ layout
    static constexpr uint32_t G2SQColPerCTA = cute::min(kBK / 8, device::kWarpThreads);
    static constexpr uint32_t G2SQRowPerCTA = threadsPerCTA / G2SQColPerCTA;
    static constexpr uint32_t kG2SQIter = cute::ceil_div(kBM, G2SQRowPerCTA);
};

template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipelineK, uint32_t kPipelineV,
    class WarpLayoutTraits_, typename DType, uint32_t kWarps=4, bool kUseTMA=true
>
struct MMATraits;

template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipelineK, uint32_t kPipelineV,
    class WarpLayoutTraits_, typename DType, uint32_t kWarps
>
struct MMATraits<kBM, kBN, kBK, kPipelineK, kPipelineV, WarpLayoutTraits_, DType, kWarps, true> {
    static constexpr uint32_t BM = kBM;
    static constexpr uint32_t BN = kBN;
    static constexpr uint32_t BK = kBK;
    static constexpr uint32_t PipelineK = kPipelineK;
    static constexpr uint32_t PipelineV = kPipelineV;

    static constexpr uint32_t G2SQColPerCTA = WarpLayoutTraits_::G2SQColPerCTA;
    static constexpr uint32_t G2SQRowPerCTA = WarpLayoutTraits_::G2SQRowPerCTA;
    static constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    static constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    static constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;
    static constexpr uint32_t MMA_M = MMARow * 16;

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
    using mma_qk_permutations = cute::Tile<cute::_16, cute::Int<BN>, cute::_16>;

    using tiled_mma_qk = decltype(cute::make_tiled_mma(
        mma_atom{},
        mma_qk_layout{},
        mma_qk_permutations{}
    ));

    // Reuse QK's warp-N ownership as PV split-K ownership. The K permutation
    // interleaves the warp and MMA-atom modes so QK-C and PV-A have identical
    // per-thread logical coordinates.
    static_assert(BN % (16 * MMACol) == 0);
    using mma_pv_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<MMARow>{}, cute::_1{}, cute::Int<MMACol>{}),
        cute::make_step(cute::_2{}, cute::_1{}, cute::_0{})
    ));
    using mma_pv_k_permutation = decltype(cute::make_ordered_layout(
        cute::make_shape(
            cute::_8{}, cute::_2{}, cute::Int<MMACol>{},
            cute::Int<BN / (16 * MMACol)>{}
        ),
        cute::make_step(cute::_0{}, cute::_2{}, cute::_1{}, cute::_3{})
    ));
    using mma_pv_permutations = cute::Tile<
        cute::_16, cute::Int<BK>, mma_pv_k_permutation
    >;

    using tiled_mma_pv = decltype(cute::make_tiled_mma(
        mma_atom{},
        mma_pv_layout{},
        mma_pv_permutations{}
    ));

    // smem layout Q, for gmem -> smem -> RF, smem -> RF ldmatrix for mma.sync layout
    using SwizzleLayoutAtomQ = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<MMA_M>{}, cute::Int<BK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sQLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomQ{},
        cute::make_shape(cute::Int<MMA_M>{}, cute::Int<BK>{}),
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
            cute::make_shape(cute::Int<MMA_M>{}, cute::Int<BK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sOLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomO{},
        cute::make_shape(cute::Int<MMA_M>{}, cute::Int<BK>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));

    // Interleave warp-K partials so the final reducer reads all partials for
    // one output coordinate contiguously.
    using sOPartialLayout = decltype(cute::make_layout(
        cute::make_shape(
            cute::Int<MMA_M>{}, cute::Int<BK>{}, cute::Int<MMACol>{}
        ),
        cute::LayoutRight{}
    ));

    static constexpr bool IsCTAReduce = MMACol > 1;
    using SharedStorageQ = SharedMemoryQ<DType, sQLayout>;
    using SharedStorageKV = SharedMemoryKV<DType, sKLayout, sVLayout, PipelineK, PipelineV, IsCTAReduce, MMACol>;
    using SharedStorageO = SharedMemoryO<DType, sOLayout>;
    using SharedStoragePartialO = SharedMemoryO<float, sOPartialLayout>;
    static constexpr size_t smem_size = cute::max(
        cute::max(sizeof(SharedStorageQ), sizeof(SharedStorageKV), sizeof(SharedStorageO)),
        sizeof(SharedStoragePartialO)
    );

    // Query G2SQ copy with TV layout
    using g2sQ_copy_type = typename CopyWidthToType<16>::type;
    using g2sQ_copy_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2sQ_copy_type>, DType>;
    using g2sQ_tiled_copy = decltype(cute::make_tiled_copy(
        g2sQ_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SQRowPerCTA>{}, cute::Int<G2SQColPerCTA>{}),
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
    using s2gO_tiled_copy = decltype(cute::make_tiled_copy(
        s2gO_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SQRowPerCTA>{}, cute::Int<G2SQColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
};

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
    const void* Leftpad;
    void* O;
    void* pO;
    void* LSE;
    uint32_t B;
    uint32_t Tk;
    uint32_t Split;
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

// Flash Decoding Stage2
template <typename DType, class LayoutPO, uint32_t SplitRound>
struct SharedMemoryPO {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutPO>> PO;
    alignas(16) float global_scale[SplitRound + 1];
};

template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t SplitRound,
    typename DType, uint32_t kWarps=4
>
struct MergeTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;

    // warp flatten across D, 32 -> 1 warp + 1 elem per thread, 64 -> 1 warp + 2 elem per thread
    static constexpr uint32_t ElemPerThr = cute::max(1, kBK / threadsPerCTA);

    // infer the G2SO layout
    static constexpr uint32_t G2SOColPerCTA = cute::min(kBK / 8, device::kWarpThreads);
    static constexpr uint32_t G2SORowPerCTA = threadsPerCTA / G2SOColPerCTA;
    static constexpr uint32_t G2SOIter = cute::ceil_div(kBM, G2SORowPerCTA);

    // smem layout O
    using SwizzleLayoutAtomPO = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<SplitRound>{}, cute::Int<kBK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sPOLayout = decltype(cute::tile_to_shape(
        SwizzleLayoutAtomPO{},
        cute::make_shape(cute::Int<SplitRound>{}, cute::Int<kBK>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using SharedStoragePO = SharedMemoryPO<DType, sPOLayout, SplitRound>;
    static constexpr size_t smem_size = sizeof(SharedStoragePO);

    using g2sO_copy_type = typename CopyWidthToType<16>::type;
    using g2sO_copy_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2sO_copy_type>, DType>;
    using g2sO_tiled_copy = decltype(cute::make_tiled_copy(
        g2sO_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SORowPerCTA>{}, cute::Int<G2SOColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<16 / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));

    using s2rO_copy_type = typename CopyWidthToType<ElemPerThr * sizeof(DType)>::type;
    using s2rO_copy_atom = cute::Copy_Atom<cute::UniversalCopy<s2rO_copy_type>, DType>;
    using s2rO_tiled_copy = decltype(cute::make_tiled_copy(
        s2rO_copy_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<threadsPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<ElemPerThr>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
};

struct MergeParams {
    const void* Mask;
    void* O;
    void* pO;
    void* LSE;
    uint32_t B;
    uint32_t Tk;
    uint32_t Split;
    float scale;
};