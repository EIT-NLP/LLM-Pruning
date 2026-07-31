#pragma once
#include "utils.cuh"
// sglang jit plugin
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>
// cute & cooperative groups
#include <cute/tensor.hpp>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <type_traits>

namespace {

namespace cg = cooperative_groups;

// layout for smem
template <typename DType, class LayoutA, class LayoutB>
struct SharedMemoryAB {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutA>> A;
    cute::ArrayEngine<DType, cute::cosize_v<LayoutB>> B;
};
template <typename DType, class LayoutD>
struct SharedMemoryD {
    cute::ArrayEngine<DType, cute::cosize_v<LayoutD>> D;
};

// helper
template <uint32_t kG2SIter, uint32_t kMMAIter, bool kWarpSkipG2S, bool kWarpSkipMMA>
struct SkipHelper;

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper<kG2SIter, kMMAIter, false, false> {
    uint8_t execute_cta;
    uint8_t rMask[kG2SIter];
    uint32_t rIndex_ld[kG2SIter];
    uint32_t rIndex_st[kMMAIter * 2];
};

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper<kG2SIter, kMMAIter, true, false> {
    uint8_t execute_cta;
    uint8_t rMask[kG2SIter];
    uint32_t rIndex_ld[kG2SIter];
    uint32_t rIndex_st[kMMAIter * 2];

    uint8_t execute_g2s_warp[kG2SIter];
};

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper<kG2SIter, kMMAIter, false, true> {
    uint8_t execute_cta;
    uint8_t rMask[kG2SIter];
    uint32_t rIndex_ld[kG2SIter];
    uint32_t rIndex_st[kMMAIter * 2];

    uint8_t execute_mma_warp[kMMAIter];
};

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper<kG2SIter, kMMAIter, true, true> {
    uint8_t execute_cta;
    uint8_t rMask[kG2SIter];
    uint32_t rIndex_ld[kG2SIter];
    uint32_t rIndex_st[kMMAIter * 2];

    uint8_t execute_g2s_warp[kG2SIter];
    uint8_t execute_mma_warp[kMMAIter];
};

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kWarps=4, uint32_t kBytes=2>
struct WarpLayoutTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;

    // infer the G2SColPerCTA
    // cp_async.cg only support 16B copy
    // BK = 32 or 64 --> G2SColPerCTA = 4 or 8, G2SRowPerWarp = 8 or 4
    static constexpr uint32_t G2SColPerCTA = kBK / 8;
    static constexpr uint32_t G2SRowPerCTA = threadsPerCTA / G2SColPerCTA;
    static constexpr uint32_t G2SRowPerWarp = G2SRowPerCTA / kWarps;

    // mma warp layout
    // mma perm at least to be (MMARow * 16, MMACol * 8 * 2), enable s2rB copy use LDSMx4
    // set to (4, 1) --> (64, 16) for BN == 16, set to (2, 2) or (1, 4) for others
    static constexpr uint32_t MMACol = cute::min(kWarps, cute::ceil_div(kBN, 16));
    static constexpr uint32_t MMARow = kWarps / MMACol;

    // Outer iter num
    static constexpr uint32_t G2SAIter = cute::ceil_div(kBM, G2SRowPerCTA);
    static constexpr uint32_t G2SBIter = cute::ceil_div(kBN, G2SRowPerCTA);
    static constexpr uint32_t MMAIter = cute::ceil_div(kBM, (16 * MMARow));
};


template <
    uint32_t kK, uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kNGIter, uint32_t kPipelineSmem, uint32_t kPipelineReg,
    class WarpLayoutTraits_, typename DType
>
struct MmaTraits {
    static constexpr uint32_t BM = kBM;
    static constexpr uint32_t BN = kBN;
    static constexpr uint32_t BK = kBK;
    static constexpr uint32_t PipelineSmem = kPipelineSmem;
    static constexpr uint32_t PipelineReg = kPipelineReg;

    static constexpr uint32_t G2SColPerCTA = WarpLayoutTraits_::G2SColPerCTA;
    static constexpr uint32_t G2SRowPerCTA = WarpLayoutTraits_::G2SRowPerCTA;
    static constexpr uint32_t G2SRowPerWarp = WarpLayoutTraits_::G2SRowPerWarp;
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
    using mma_mnk_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<MMARow>{}, cute::Int<MMACol>{}, cute::_1{}),
        cute::make_step(cute::_2{}, cute::_1{}, cute::_0{})
    ));
    using mma_permutations = cute::Tile<cute::Int<BM>, cute::Int<BN>, cute::_16>;

    using tiled_mma = decltype(cute::make_tiled_mma(
        mma_atom{},
        mma_mnk_layout{},
        mma_permutations{}
    ));

    // smem layout A&B
    using G2SSwizzleLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<8>{}, cute::Int<BK>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sALayout = decltype(cute::tile_to_shape(
        G2SSwizzleLayoutAtom{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<BK>{}, cute::Int<PipelineSmem>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));
    using sBLayout = decltype(cute::tile_to_shape(
        G2SSwizzleLayoutAtom{},
        cute::make_shape(cute::Int<BN>{}, cute::Int<BK>{}, cute::Int<PipelineSmem>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));

    // smem layout D
    using S2GSwizzleLayoutAtom = decltype(cute::composition(
        cute::Swizzle<3, 3, 3>{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<BM>{}, cute::Int<BN>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using sDLayout = decltype(cute::tile_to_shape(
        S2GSwizzleLayoutAtom{},
        cute::make_shape(cute::Int<BM>{}, cute::Int<BN>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));

    using SharedStorageAB = SharedMemoryAB<DType, sALayout, sBLayout>;
    using SharedStorageD = SharedMemoryD<DType, sDLayout>;
    static constexpr size_t smem_size = cute::max(sizeof(SharedStorageAB), sizeof(SharedStorageD));

    // G2S copy and its tv layout
    using g2s_copy_type = typename CopyWidthToType<16>::type;
    using g2s_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEGLOBAL<g2s_copy_type>, DType>;

    using g2sA_tiled_copy_g = decltype(cute::make_tiled_copy(
        g2s_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<G2SColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<8>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using g2sA_tiled_copy_s = decltype(cute::make_tiled_copy(
        g2s_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<8>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));

    using g2sB_tiled_copy = decltype(cute::make_tiled_copy(
        g2s_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<8>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));

    // s2r ldmatrix copy and its tv layout
    using s2r_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using s2rA_tiled_copy = decltype(cute::make_tiled_copy_A(s2r_atom{}, tiled_mma{}));
    using s2rB_tiled_copy = decltype(cute::make_tiled_copy_B(s2r_atom{}, tiled_mma{}));

    // r2s copy and its tv layout
    using r2s_atom = cute::Copy_Atom<cute::UniversalCopy<uint32_t>, DType>;
    using r2s_tiled_copy = decltype(cute::make_tiled_copy_C(r2s_atom{}, tiled_mma{}));

    // s2g copy and its tv layout
    static constexpr uint32_t s2g_copy_width = cute::min(16, (kBN / G2SColPerCTA) * sizeof(DType));
    using s2g_copy_type = typename CopyWidthToType<s2g_copy_width>::type;
    using s2g_atom = cute::Copy_Atom<cute::UniversalCopy<s2g_copy_type>, DType>;
    using s2g_tiled_copy_s = decltype(cute::make_tiled_copy(
        s2g_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<s2g_copy_width / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using s2g_tiled_copy_g = decltype(cute::make_tiled_copy(
        s2g_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<1>{}, cute::Int<G2SColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<s2g_copy_width / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
};

struct GEMMParams {
    const void* A;
    const void* B;
    const void* Mask;
    const void* Index;
    void* D;
    uint32_t M;
    uint32_t SplitK;
};

// main kernel func
template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kL2Group, uint32_t kPipelineSmem, uint32_t kPipelineReg,
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
    class WarpLayoutTraits_, class MmaTraits_, typename DType,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA
>
__global__ void gemm_mn_v0_kernel(const __grid_constant__ GEMMParams params) {
    using namespace cute;
    const auto& [A, B, Mask, Index, D, M, SplitK] = params;
    constexpr uint32_t N = kN;
    constexpr uint32_t K = kK;
    constexpr uint32_t NG = kNG;
    constexpr uint32_t NGIter = kNGIter;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t BK = kBK;
    constexpr uint32_t PipelineSmem = kPipelineSmem;
    constexpr uint32_t PipelineReg = kPipelineReg;
    const uint32_t KPerCTA = K / SplitK;
    const uint32_t KOutIter = KPerCTA / (BK * NGIter);

    constexpr uint32_t G2SColPerCTA = WarpLayoutTraits_::G2SColPerCTA;
    constexpr uint32_t G2SRowPerCTA = WarpLayoutTraits_::G2SRowPerCTA;
    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;
    constexpr uint32_t G2SAIter = WarpLayoutTraits_::G2SAIter;
    constexpr uint32_t G2SBIter = WarpLayoutTraits_::G2SBIter;
    constexpr uint32_t MMAIter = WarpLayoutTraits_::MMAIter;

    using mma_atom = typename MmaTraits_::mma_atom;
    using tiled_mma = typename MmaTraits_::tiled_mma;
    using sALayout = typename MmaTraits_::sALayout;
    using sBLayout = typename MmaTraits_::sBLayout;
    using sDLayout = typename MmaTraits_::sDLayout;
    using g2sA_tiled_copy_g = typename MmaTraits_::g2sA_tiled_copy_g;
    using g2sA_tiled_copy_s = typename MmaTraits_::g2sA_tiled_copy_s;
    using g2sB_tiled_copy = typename MmaTraits_::g2sB_tiled_copy;
    using s2rA_tiled_copy = typename MmaTraits_::s2rA_tiled_copy;
    using s2rB_tiled_copy = typename MmaTraits_::s2rB_tiled_copy;
    using r2s_tiled_copy = typename MmaTraits_::r2s_tiled_copy;
    using s2g_tiled_copy_s = typename MmaTraits_::s2g_tiled_copy_s;
    using s2g_tiled_copy_g = typename MmaTraits_::s2g_tiled_copy_g;

    // (cdiv(M, BM), cdiv(N, BN), SplitK)
    const uint32_t tidx = threadIdx.x;
    const uint32_t NBM = gridDim.x;
    const uint32_t NBN = gridDim.y;
    const uint2 tile_idx = l2_swizzle<kL2Group>(
        blockIdx.x, blockIdx.y,
        NBM, NBN
    );
    const uint32_t bidx = tile_idx.x;
    const uint32_t bidy = tile_idx.y;
    const uint32_t bidz = blockIdx.z;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

    const uint32_t base_off_m = bidx * BM;
    const uint32_t base_off_n = bidy * BN;
    const uint32_t base_off_k = bidz * KPerCTA;

    const uint32_t thread_off_m = base_off_m + tidx / G2SColPerCTA;

    //
    // prepare tensors and smem layout
    //
    Tensor mA = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(A)),
        make_shape(M, Int<K>{}),
        make_stride(Int<K>{}, _1{})
    );
    Tensor mB = make_tensor(
        make_gmem_ptr(static_cast<const DType*>(B)),
        make_shape(N, Int<K>{}),
        make_stride(Int<K>{}, _1{})
    );
    Tensor mD = make_tensor(
        make_gmem_ptr(static_cast<DType*>(D)),
        make_shape(M, Int<N>{}),
        make_stride(Int<N>{}, _1{})
    );
    Tensor mMask = make_tensor(
        make_gmem_ptr(static_cast<const uint8_t*>(Mask)),
        make_shape(M, Int<NG>{}),
        make_stride(Int<NG>{}, _1{})
    );
    Tensor mIndex = make_tensor(
        make_gmem_ptr(static_cast<const int64_t*>(Index)),
        make_shape(M, Int<NG>{}),
        make_stride(Int<NG>{}, _1{})
    );

    extern __shared__ uint8_t shared_memory[];
    using SharedStorageAB = SharedMemoryAB<DType, sALayout, sBLayout>;
    SharedStorageAB &smem_ab = *reinterpret_cast<SharedStorageAB*>(shared_memory);

    using SharedStorageD = SharedMemoryD<DType, sDLayout>;
    SharedStorageD &smem_d = *reinterpret_cast<SharedStorageD*>(shared_memory);

    Tensor sA = make_tensor(
        make_smem_ptr(smem_ab.A.begin()),
        sALayout{}
    );
    Tensor sB = make_tensor(
        make_smem_ptr(smem_ab.B.begin()),
        sBLayout{}
    );
    Tensor sD = make_tensor(
        make_smem_ptr(smem_d.D.begin()),
        sDLayout{}
    );

    SkipHelper<G2SAIter, MMAIter, kWarpSkipG2S, kWarpSkipMMA> skip_helper;

    const uint32_t total_k_tile = KOutIter * NGIter;
    constexpr uint32_t RealPipeline = cute::min(PipelineSmem, NGIter);
    uint32_t k_group_id = bidz * KOutIter;

    Tensor gA = local_tile(
        mA,
        make_tile(_1{}, Int<BK>{}),
        make_coord(_, _)
    );
    Tensor gB = local_tile(
        mB,
        make_tile(Int<BN>{}, Int<BK>{}),
        make_coord(bidy, _)
    );

    auto g2sA_thr_copy_g = g2sA_tiled_copy_g{}.get_slice(tidx % G2SColPerCTA);
    auto g2sA_thr_copy_s = g2sA_tiled_copy_s{}.get_slice(tidx);
    auto g2sB_thr_copy = g2sB_tiled_copy{}.get_slice(tidx);

    auto pSgB = g2sB_thr_copy.partition_S(gB);
    auto pDsA = g2sA_thr_copy_s.partition_D(sA);
    auto pDsB = g2sB_thr_copy.partition_D(sB);

    Tensor gD = local_tile(
        mD,
        make_tile(Int<BM>{}, Int<BN>{}),
        make_coord(bidx, bidy)
    );
    ThrMMA thr_mma = tiled_mma{}.get_slice(tidx);
    auto rA = thr_mma.partition_fragment_A(sA(_, _, _0{}));
    auto rB = thr_mma.partition_fragment_B(sB(_, _, _0{}));
    auto rD = thr_mma.partition_fragment_C(gD);

    auto s2rA_thr_copy = s2rA_tiled_copy{}.get_slice(tidx);
    auto pSsA = s2rA_thr_copy.partition_S(sA);
    auto pDrA = s2rA_thr_copy.retile_D(rA);

    auto s2rB_thr_copy = s2rB_tiled_copy{}.get_slice(tidx);
    auto pSsB = s2rB_thr_copy.partition_S(sB);
    auto pDrB = s2rB_thr_copy.retile_D(rB);

    CUTE_NO_UNROLL
    for (uint32_t k_out_iter=0; k_out_iter < KOutIter; ++k_out_iter, ++k_group_id) {
        //
        // load mask & index
        // assum the BK = 32, G=128, K = 4096, SplitK = 4, then NGIter = 128 / 32 = 4, each CTA handle 4096 / 4 = 1024 -> 8 out iter, 4 inner iter.
        // for block skip, the coord is (..., k_group_id)
        //
        CUTE_UNROLL
        for (uint32_t i=0, off=thread_off_m; i < G2SAIter; ++i, off+=G2SRowPerCTA) {
            skip_helper.rMask[i] = off < M ? mMask(make_coord(off, k_group_id)) : 0;
            skip_helper.rIndex_ld[i] = off < M ? static_cast<uint32_t>(mIndex(make_coord(off, k_group_id))) : 0;
        }

        // check if early exit current block or warp's g2s load
        skip_helper.execute_cta = 0;
        cg::thread_block this_cta = cg::this_thread_block();
        auto this_cta_tile = cg::tiled_partition<threadsPerCTA>(this_cta);
        CUTE_UNROLL
        for (uint32_t i=0; i < G2SAIter; ++i) {
            skip_helper.execute_cta |= skip_helper.rMask[i];
            if constexpr (kWarpSkipG2S) {
                skip_helper.execute_g2s_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask[i] > 0)));
            }
        }
        skip_helper.execute_cta = cg::reduce(this_cta_tile, skip_helper.execute_cta, cg::bit_or<uint8_t>{});
        if (skip_helper.execute_cta == 0) continue;

        // check if mma need execute
        // mma rD layout --> (8, 8), 2, 1, each warp's thread handle 2 row: lane_id / 4 and lane_id / 4 + 8
        CUTE_UNROLL
        for (
            uint32_t i=0,
            off=base_off_m + (warp_id / MMACol) * 16 + lane_id / 4;
            i < MMAIter;
            ++i, off+=16 * MMARow
        ) {
            skip_helper.rIndex_st[i*2] = (off < M && mMask(make_coord(off, k_group_id)) > 0) ? static_cast<uint32_t>(mIndex(make_coord(off, k_group_id))) : M;
            skip_helper.rIndex_st[i*2+1] = (off + 8 < M && mMask(make_coord(off + 8, k_group_id)) > 0) ? static_cast<uint32_t>(mIndex(make_coord(off + 8, k_group_id))) : M;
        }
        
        uint32_t producer = 0;
        uint32_t consumer = 0;
        clear(rD);

        // prologue, fill the pipeline
        uint32_t k_base_tile = (bidz * KOutIter + k_out_iter) * NGIter;
        uint32_t producer_stage = producer % RealPipeline;
        CUTE_UNROLL
        for (uint32_t p=0; p < RealPipeline - 1; ++p) {
            CUTE_UNROLL
            for (uint32_t i=0; i < size<1>(pDsA); ++i) {
                if constexpr (kThrSkipG2S) {
                    if (skip_helper.rMask[i]) {
                        auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                        auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                        CUTE_UNROLL
                        for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                            copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                        }
                    }
                }
                else if constexpr (kWarpSkipG2S) {
                    if (skip_helper.execute_g2s_warp[i]) {
                        auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                        auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                        CUTE_UNROLL
                        for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                            copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                        }
                    }
                }
                else {
                    auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                    auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                    CUTE_UNROLL
                    for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                        copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                    }
                }
            }

            copy(g2sB_tiled_copy{}, pSgB(_, _, _, k_base_tile + producer), pDsB(_, _, _, make_coord(0, producer_stage)));
            cp_async_fence();
            ++producer;
            producer_stage = (producer_stage + 1) % RealPipeline;
        }

        uint32_t consumer_stage = consumer % RealPipeline;
        // ready for smem -> rmem pipeline
        cp_async_wait<cute::max(RealPipeline, 2) - 2>();
        __syncthreads();

        if constexpr (kWarpSkipMMA) {
            CUTE_UNROLL
            for (uint32_t i=0; i < MMAIter; ++i) {
                if (skip_helper.execute_mma_warp[i]) {
                    copy(s2rA_tiled_copy{}, pSsA(make_coord(_, i), _, 0, make_coord(0, consumer_stage)), pDrA(make_coord(_, i), _, 0));
                }
            }
        }
        else {
            copy(s2rA_tiled_copy{}, pSsA(_, _, 0, make_coord(0, consumer_stage)), pDrA(_, _, 0));
        }
        copy(s2rB_tiled_copy{}, pSsB(_, _, 0, make_coord(0, consumer_stage)), pDrB(_, _, 0));

        CUTE_NO_UNROLL
        for (uint32_t k_tile=0; k_tile < NGIter; ++k_tile) {
            constexpr uint32_t reg_tile_num = cute::size<2>(pDrA);
            CUTE_UNROLL
            for (uint32_t ldsm=0; ldsm < reg_tile_num; ++ldsm) {
                uint32_t ldsm_next = (ldsm + 1) % reg_tile_num;

                if (ldsm == reg_tile_num - 1) {
                    cp_async_wait<cute::max(RealPipeline, 2) - 2>();
                    __syncthreads();
                    consumer_stage = (consumer_stage + 1) % RealPipeline;
                }
                
                // s2r
                if constexpr (kWarpSkipMMA) {
                    CUTE_UNROLL
                    for (uint32_t i=0; i < MMAIter; ++i) {
                        if (skip_helper.execute_mma_warp[i]) {
                            copy(s2rA_tiled_copy{}, pSsA(make_coord(_, i), _, ldsm_next, make_coord(0, consumer_stage)), pDrA(make_coord(_, i), _, ldsm_next));
                        }
                    }
                }
                else {
                    copy(s2rA_tiled_copy{}, pSsA(_, _, ldsm_next, make_coord(0, consumer_stage)), pDrA(_, _, ldsm_next));
                }
                copy(s2rB_tiled_copy{}, pSsB(_, _, ldsm_next, make_coord(0, consumer_stage)), pDrB(_, _, ldsm_next));

                // g2s
                if (ldsm == 0) {
                    if (producer < NGIter) {
                        CUTE_UNROLL
                        for (uint32_t i=0; i < size<1>(pDsA); ++i) {
                            if constexpr (kThrSkipG2S) {
                                if (skip_helper.rMask[i]) {
                                    auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                                    auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                                    CUTE_UNROLL
                                    for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                                        copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                                    }
                                }
                            }
                            else if constexpr (kWarpSkipG2S) {
                                if (skip_helper.execute_g2s_warp[i]) {
                                    auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                                    auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                                    CUTE_UNROLL
                                    for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                                        copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                                    }
                                }
                            }
                            else {
                                auto gAtA = gA(_, _, skip_helper.rIndex_ld[i], k_base_tile + producer);
                                auto pSgA = g2sA_thr_copy_g.partition_S(gAtA);

                                CUTE_UNROLL
                                for (uint32_t j=0; j < size<2>(pDsA); ++j) {
                                    copy(g2sA_tiled_copy_g{}, pSgA(_, _, j), pDsA(_, i, j, make_coord(_, producer_stage)));
                                }
                            }
                        }
                        copy(g2sB_tiled_copy{}, pSgB(_, _, _, k_base_tile + producer), pDsB(_, _, _, make_coord(0, producer_stage)));
                        ++producer;
                        producer_stage = (producer_stage + 1) % RealPipeline;
                    }
                    cp_async_fence();
                }
                
                CUTE_UNROLL
                for (uint32_t i=0; i < size<1>(rA); ++i) {
                    if constexpr (kWarpSkipMMA) {
                        if (skip_helper.execute_mma_warp[i]) {
                            CUTE_UNROLL
                            for (uint32_t j=0; j < size<1>(rB); ++j) {
                                gemm(tiled_mma{}, rA(_, i, ldsm), rB(_, j, ldsm), rD(_, i, j));
                            }
                        }
                    }
                    else {
                        CUTE_UNROLL
                        for (uint32_t j=0; j < size<1>(rB); ++j) {
                            gemm(tiled_mma{}, rA(_, i, ldsm), rB(_, j, ldsm), rD(_, i, j));
                        }
                    }
                }
            }
        }

        // epilogue, split-k atomic-add write back
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(rD); ++i) {
            CUTE_UNROLL
            for (
                uint32_t j=0,
                gD_col_off=base_off_n + (warp_id % MMACol) * 8 + (lane_id % 4) * 2;
                j < size<2>(rD);
                ++j, gD_col_off+=MMACol * 8
            ) {
                // The packed f16x2/bf16x2 atomic must start at an even half address.
                uint32_t src_pack;
                if (skip_helper.rIndex_st[i * 2] < M) {
                    src_pack = AccumlatorPack2<DType>()(rD(1, i, j), rD(0, i, j));
                    AtomicAdd<DType, 2>()(src_pack, &mD(skip_helper.rIndex_st[i * 2], gD_col_off));
                }

                if (skip_helper.rIndex_st[i * 2 + 1] < M) {
                    src_pack = AccumlatorPack2<DType>()(rD(3, i, j), rD(2, i, j));
                    AtomicAdd<DType, 2>()(src_pack, &mD(skip_helper.rIndex_st[i * 2 + 1], gD_col_off));
                }
            }
        }
    }
}

template <
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA,
    bool kUsePDL, typename DType
>
struct GEMM_K_V0_Host {
    static void run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView Index,
        const tvm::ffi::TensorView D,
        const float sparsity,
        const uint32_t SplitK
    ) {
        using namespace host;
        RuntimeCheck(
            sparsity >= 0 && sparsity <= 1,
            "sparsity must be in [0, 1]"
        );

        auto M = SymbolicSize{"num_tokens"};
        auto N = SymbolicSize{"out_features"};
        auto K = SymbolicSize{"in_features"};
        auto NG = SymbolicSize{"num_groups"};
        auto NGIter = SymbolicSize{"num_groups_iter"};
        auto device = SymbolicDevice{};

        N.set_value(kN);
        K.set_value(kK);
        NG.set_value(kNG);
        NGIter.set_value(kNGIter);
        device.set_options<kDLCUDA>();

        // host-side checking
        TensorMatcher({M, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(A);
        
        TensorMatcher({N, K}) //
            .with_strides({K, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(B);
        
        TensorMatcher({M, N}) //
            .with_strides({N, 1})
            .with_dtype<DType>()
            .with_device(device)
            .verify(D);
        
        TensorMatcher({M, NG}) //
            .with_strides({NG, 1})
            .with_device(device)
            .verify(Mask);
        
        TensorMatcher({M, NG}) //
            .with_strides({NG, 1})
            .with_dtype<int64_t>()
            .with_device(device)
            .verify(Index);
        
        RuntimeCheck(
            sizeof(DType) == 2,
            "DType must be fp16 or bf16"
        );

        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        RuntimeCheck(
            kK % 32 == 0 && kN % 16 == 0,
            "N and K must be divisible by 16 and 32"
        );
        RuntimeCheck(
            SplitK > 0 && SplitK <= kK / (kBK * kNGIter) &&
                kK % SplitK == 0 && (kK / SplitK) % (kBK * kNGIter) == 0,
            "SplitK must evenly partition K into complete pruning groups"
        );

        const auto params = GEMMParams{
            .A = A.data_ptr(),
            .B = B.data_ptr(),
            .Mask = Mask.data_ptr(),
            .Index = Index.data_ptr(),
            .D = D.data_ptr(),
            .M = num_tokens,
            .SplitK = SplitK
        };

        // host-side static tiling
        using warp_layout_traits = WarpLayoutTraits<kBM, kBN, kBK, 4, sizeof(DType)>;
        using mma_traits = MmaTraits<kK, kBM, kBN, kBK, kNGIter, kPipeline, 2, warp_layout_traits, DType>;
        constexpr auto kernel = gemm_mn_v0_kernel<
            kBM, kBN, kBK, 4, kPipeline, 2, kN, kK, kNG, kNGIter,
            warp_layout_traits, mma_traits, DType,
            kThrSkipG2S, kWarpSkipG2S, kWarpSkipMMA
        >;

        const dim3 grid_size = {cute::ceil_div(params.M, kBM), cute::ceil_div(kN, kBN), SplitK};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size + 4 * 2 * sizeof(uint8_t));
        LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
    }
};

} // namespace
