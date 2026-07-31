#pragma once
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

namespace {

// shared memory
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 800
#define SMEM_SIZE 167936
#elif defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000)
#define SMEM_SIZE 233472
#else
#define SMEM_SIZE 102400
#endif

namespace cg = cooperative_groups;

// fast math
constexpr uint32_t next_pow_of_2(uint32_t x) {
    if (x <= 0) return 0;
    return 1 << (32 - __builtin_clz(x));
}

__device__ __forceinline__ float expf_ftz(float x) {
  // e^x = (2^m)^x
  // e = 2^m
  // m = lg2(e)
  // m = 1.4426950408889634

  constexpr float m = 1.4426950408889634f;
  float r;
  asm volatile("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(r) : "f"(x * m));
  return r;
}


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
template <uint32_t Bytes>
struct CopyWidthToType;
template <>
struct CopyWidthToType<4> {
    using type = cute::uint32_t;
};
template <>
struct CopyWidthToType<8> {
    using type = cute::uint64_t;
};
template <>
struct CopyWidthToType<16> {
    using type = cute::uint128_t;
};

template <typename DType>
struct AccumlatorPack2;

template <>
struct AccumlatorPack2<fp16_t> {
    __device__ __forceinline__ uint32_t operator()(fp32_t odd, fp32_t even) {
        uint32_t d;
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
            : "=r"(d)
            : "f"(odd), "f"(even));
        return d;
    }
    
};

template <>
struct AccumlatorPack2<bf16_t> {
    __device__ __forceinline__ uint32_t operator()(fp32_t odd, fp32_t even) {
        uint32_t d;
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
            : "=r"(d)
            : "f"(odd), "f"(even));
        return d;
    }
};

template <uint32_t kG2SIter, uint32_t kMMAIter>
struct SkipHelper {
    uint8_t execute_cta;
    uint8_t execute_g2s_warp[kG2SIter];
    uint8_t execute_mma_warp[kMMAIter];

    uint8_t rMask[kG2SIter];
    uint32_t rIndex_ld[kG2SIter];
    uint8_t rMask_st[2 * kMMAIter]; // (8,8),2,1 for rD accumlator layout
    // uint32_t rIndex_st[2 * kMMAIter]; // (8,8),2,1 for rD accumlator layout
};

template <uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kWarps=4, uint32_t kBytes=2>
struct WarpLayoutTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;
    static constexpr uint32_t MMARowPerWarp = 16;

    // infer the G2SColPerCTA
    // cp_async.cg only support 16B copy
    static constexpr uint32_t G2SColPerCTA = kBK * kBytes / 16;
    static constexpr uint32_t G2SRowPerCTA = threadsPerCTA / G2SColPerCTA;
    static constexpr uint32_t G2SRowPerWarp = G2SRowPerCTA / kWarps;

    // mma warp layout
    static constexpr uint32_t MMARow = kBM / 16 > kWarps ? kWarps : kBM / 16;
    static constexpr uint32_t MMACol = kWarps / MMARow;
};


template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline,
    class WarpLayoutTraits_, typename DType
>
struct MmaTraits {
    static constexpr uint32_t BM = kBM;
    static constexpr uint32_t BN = kBN;
    static constexpr uint32_t BK = kBK;
    static constexpr uint32_t Pipeline = kPipeline;

    static constexpr uint32_t G2SColPerCTA = WarpLayoutTraits_::G2SColPerCTA;
    static constexpr uint32_t G2SRowPerCTA = WarpLayoutTraits_::G2SRowPerCTA;
    static constexpr uint32_t G2SRowPerWarp = WarpLayoutTraits_::G2SRowPerWarp;
    static constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    static constexpr uint32_t MMARowPerWarp = WarpLayoutTraits_::MMARowPerWarp;
    static constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    static constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;

    template <uint32_t NK>
    static constexpr auto get_smem_swizzle() {
        if constexpr (NK < 64) return cute::GMMA::Layout_K_SW64_Atom<DType>{};
        else return cute::GMMA::Layout_K_SW128_Atom<DType>{};
    }

    // smem layout
    using sALayout = decltype(cute::tile_to_shape(
        decltype(get_smem_swizzle<BK>()){},
        cute::make_shape(cute::Int<kBM>{}, cute::Int<kBK>{}, cute::Int<kPipeline>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));
    using sBLayout = decltype(cute::tile_to_shape(
        decltype(get_smem_swizzle<BK>()){},
        cute::make_shape(cute::Int<kBN>{}, cute::Int<kBK>{}, cute::Int<kPipeline>{}),
        cute::make_step(cute::_1{}, cute::_0{}, cute::_2{})
    ));
    using sDLayout = decltype(cute::tile_to_shape(
        decltype(get_smem_swizzle<BN>()){},
        cute::make_shape(cute::Int<kBM>{}, cute::Int<kBN>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));

    using SharedStorageAB = SharedMemoryAB<DType, sALayout, sBLayout>;
    using SharedStorageD = SharedMemoryD<DType, sDLayout>;
    static constexpr size_t smem_size = cute::max(sizeof(SharedStorageAB), sizeof(SharedStorageD));

    // atom
    static constexpr uint32_t g2s_copy_width = (BK / G2SColPerCTA) * sizeof(DType);
    static_assert(g2s_copy_width == 16, "cp.async.cg requires 16B per-thread copy");

    using g2s_copy_type = typename CopyWidthToType<g2s_copy_width>::type;
    using g2sA_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<g2s_copy_type>, DType>;
    using s2rA_atom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, DType>;
    using g2sB_atom = cute::Copy_Atom<cute::SM80_CP_ASYNC_CACHEALWAYS<g2s_copy_type>, DType>;
    using s2rB_atom = cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, DType>;
    using mma_atom = cute::MMA_Atom<cute::SM80_16x8x16_F32F16F16F32_TN>;

    static constexpr uint32_t s2g_copy_width = cute::min((BN / G2SColPerCTA) * sizeof(DType), 16);

    using s2g_copy_type = typename CopyWidthToType<s2g_copy_width>::type;
    using r2sD_atom = cute::Copy_Atom<cute::UniversalCopy<uint32_t>, DType>;
    using s2gD_atom = cute::Copy_Atom<cute::UniversalCopy<s2g_copy_type>, DType>;

    // tv layout g2sA
    using g2sA_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sA_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<kBK / G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sA_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(g2sA_thr_layout{}, g2sA_val_layout{}))));
    using g2sA_tiled_copy = decltype(cute::make_tiled_copy(
        g2sA_atom{}, g2sA_thr_layout{}, g2sA_val_layout{}
    ));

    // tv layout g2sB
    using g2sB_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sB_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<kBN / G2SRowPerCTA>{}, cute::Int<kBK / G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using g2sB_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(g2sB_thr_layout{}, g2sB_val_layout{}))));
    using g2sB_tiled_copy = decltype(cute::make_tiled_copy(
        g2sB_atom{}, g2sB_thr_layout{}, g2sB_val_layout{}
    ));

    // tv layout D
    using D_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<G2SRowPerCTA>{}, cute::Int<G2SRowPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using D_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::Int<kBM / G2SRowPerCTA>{}, cute::Int<kBN / G2SRowPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using D_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(D_thr_layout{}, D_val_layout{}))));

    // mma & ldmatrix layout
    using tiled_mma = decltype(cute::make_tiled_mma(
        mma_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<MMARow>{}, cute::Int<MMACol>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using s2rA_tiled_copy = decltype(cute::make_tiled_copy_A(s2rA_atom{}, tiled_mma{}));
    using s2rB_tiled_copy = decltype(cute::make_tiled_copy_B(s2rB_atom{}, tiled_mma{}));
    using r2sD_tiled_copy = decltype(cute::make_tiled_copy_C(r2sD_atom{}, tiled_mma{}));

    // s2g layout, row-wise
    using s2gD_thr_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using s2gD_val_layout = decltype(cute::make_ordered_layout(
        cute::make_shape(cute::_1{}, cute::Int<kBN / G2SColPerCTA>{}),
        cute::make_step(cute::_1{}, cute::_0{})
    ));
    using s2gD_tv_tiler = decltype(cute::product_each(cute::shape(cute::raked_product(s2gD_thr_layout{}, s2gD_val_layout{}))));
    using s2gD_tiled_copy = decltype(cute::make_tiled_copy(
        s2gD_atom{}, s2gD_thr_layout{}, s2gD_val_layout{}
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

// mainloop func
template <
    typename gATensor,
    typename pDsATensor, typename pDsBTensor, typename pSgBTensor,
    typename ThrCopyA, typename TiledCopyA, typename TiledCopyB,
    uint32_t kPipeline, uint32_t G2SRowPerCTA, uint32_t G2SColPerCTA, uint32_t kG2SIter
>
__device__ __forceinline__ uint32_t load_AB(
    const gATensor& gA,
    pDsATensor& pDsA, pDsBTensor& pDsB, const pSgBTensor& pSgB,
    const uint32_t* rIndex, const uint8_t* execute_g2s_warp,
    const ThrCopyA& g2sA_thr_copy, const TiledCopyA& g2sA_tiled_copy, const TiledCopyB& g2sB_tiled_copy,
    uint32_t producer, uint32_t base_off_k_tile
) {
    using namespace cute;
    const uint32_t cur_stage = producer % kPipeline;
    const uint32_t k_tile_id = base_off_k_tile + producer;

    CUTE_UNROLL
    for (uint32_t i=0, j=threadIdx.x / G2SColPerCTA; i < kG2SIter; ++i, j+=G2SRowPerCTA) {
        if (execute_g2s_warp[i]) {
            auto gAtA = gA(make_coord(_, _), make_coord(rIndex[i], k_tile_id));
            auto pSgA = g2sA_thr_copy.partition_S(gAtA);
            copy(g2sA_tiled_copy, pSgA(make_coord(_, _), _, 0), pDsA(make_coord(_, _), j, _, cur_stage));
        }
    }
    copy(g2sB_tiled_copy, pSgB(make_coord(_, _), _, _, k_tile_id), pDsB(make_coord(_, _), _, _, cur_stage));

    cute::cp_async_fence();
    return producer + 1;
}

template <
    typename pSrATensor, typename pDrATensor, typename pSrBTensor, typename pDrBTensor,
    typename rATensor, typename rBTensor, typename rDTensor,
    typename TiledCopyA, typename TiledCopyB, typename TiledGEMM,
    uint32_t kPipeline
>
__device__ __forceinline__ uint32_t issue_mma(
    const pSrATensor& pSrA, pDrATensor& pDrA, const pSrBTensor& pSrB, pDrBTensor& pDrB,
    const rATensor& rA, const rBTensor& rB, rDTensor& rD,
    const uint8_t* execute_mma_warp,
    const TiledCopyA& s2rA_tiled_copy, const TiledCopyB& s2rB_tiled_copy, const TiledGEMM& tiled_gemm,
    uint32_t consumer
) {
    using namespace cute;
    const uint32_t cur_stage = consumer % kPipeline;

    CUTE_UNROLL
    for (uint32_t i=0; i < size<1>(pDrA); ++i) {
        if (execute_mma_warp[i]) {
            copy(s2rA_tiled_copy, pSrA(make_coord(_, _), i, _, cur_stage), pDrA(make_coord(_, _), i, _));
            copy(s2rB_tiled_copy, pSrB(make_coord(_, _), _, _, cur_stage), pDrB(make_coord(_, _), _, _));

            CUTE_UNROLL
            for (uint32_t j=0; j < size<1>(pDrB); ++j) {
                CUTE_UNROLL
                for (uint32_t k=0; k < size<2>(pSrA); ++k) {
                    gemm(
                        tiled_gemm,
                        rA(_, i, k),
                        rB(_, j, k),
                        rD(_, i, j)
                    );
                }
            }
        }
    }
    return consumer + 1;
}

// epilogue func
template <typename rDTensor, uint8_t kAct>
struct ElementWiseActivation;

template <typename rDTensor>
struct ElementWiseActivation<rDTensor, 0> {
    __device__ __forceinline__ void operator()(rDTensor& rD) {}
};

template <typename rDTensor>
struct ElementWiseActivation<rDTensor, 1> { // relu
    __device__ __forceinline__ void operator()(rDTensor& rD) {
        CUTE_UNROLL
        for (uint32_t i=0; i < cute::size(rD); ++i) {
            rD(i) = rD(i) > 0 ? rD(i) : 0;
        }
    }
};

template <typename rDTensor>
struct ElementWiseActivation<rDTensor, 2> { // silu
    __device__ __forceinline__ void operator()(rDTensor& rD) {
        CUTE_UNROLL
        for (uint32_t i=0; i < cute::size(rD); ++i) {
            rD(i) *= 1 / (1 + expf_ftz(-rD(i)));
        }
    }
};

template <
    uint32_t kWait,
    typename pSrATensor, typename pDrATensor, typename pSrBTensor, typename pDrBTensor,
    typename rATensor, typename rBTensor, typename rDTensor,
    typename TiledCopyA, typename TiledCopyB, typename TiledGEMM,
    uint32_t kPipeline
>
__device__ __forceinline__ uint32_t pipeline_drain(
    const pSrATensor& pSrA, pDrATensor& pDrA, const pSrBTensor& pSrB, pDrBTensor& pDrB,
    const rATensor& rA, const rBTensor& rB, rDTensor& rD,
    const uint8_t* execute_mma_warp,
    const TiledCopyA& s2rA_tiled_copy, const TiledCopyB& s2rB_tiled_copy, const TiledGEMM& tiled_gemm,
    uint32_t consumer
) {
    using namespace cute;
    cp_async_wait<kWait>();
    __syncthreads();
    consumer = issue_mma<
        pSrATensor, pDrATensor, pSrBTensor, pDrBTensor,
        rATensor, rBTensor, rDTensor,
        TiledCopyA, TiledCopyB, TiledGEMM,
        kPipeline
    >(
        pSrA, pDrA, pSrB, pDrB, rA, rB, rD,
        execute_mma_warp,
        s2rA_tiled_copy, s2rB_tiled_copy, tiled_gemm,
        consumer
    );
    if constexpr (kWait > 0) {
        return pipeline_drain<
            kWait - 1,
            pSrATensor, pDrATensor, pSrBTensor, pDrBTensor,
            rATensor, rBTensor, rDTensor,
            TiledCopyA, TiledCopyB, TiledGEMM,
            kPipeline
        >(
            pSrA, pDrA, pSrB, pDrB, rA, rB, rD,
            execute_mma_warp,
            s2rA_tiled_copy, s2rB_tiled_copy, tiled_gemm,
            consumer
        );
    }
    else {
        return consumer;
    }
}

// main kernel func
template <
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kL2Group, uint32_t kPipeline,
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
    class WarpLayoutTraits_, class MmaTraits_, uint8_t kAct, typename DType
>
__global__ void gemm_mn_v2_kernel(const __grid_constant__ GEMMParams params) {
    using namespace cute;
    const auto& [A, B, Mask, Index, D, M, SplitK] = params;
    constexpr uint32_t N = kN;
    constexpr uint32_t K = kK;
    constexpr uint32_t NG = kNG;
    constexpr uint32_t NGIter = kNGIter;
    constexpr uint32_t BM = kBM;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t BK = kBK;
    constexpr uint32_t Pipeline = kPipeline;
    const uint32_t SplitKSize = K / SplitK;
    constexpr uint8_t Activation = kAct;

    constexpr uint32_t G2SColPerCTA = WarpLayoutTraits_::G2SColPerCTA;
    constexpr uint32_t G2SRowPerCTA = WarpLayoutTraits_::G2SRowPerCTA;
    constexpr uint32_t MMARowPerWarp = WarpLayoutTraits_::MMARowPerWarp;
    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t MMARow = WarpLayoutTraits_::MMARow;
    constexpr uint32_t MMACol = WarpLayoutTraits_::MMACol;

    using sALayout = typename MmaTraits_::sALayout;
    using sBLayout = typename MmaTraits_::sBLayout;
    using sDLayout = typename MmaTraits_::sDLayout;
    using g2sA_tv_tiler = typename MmaTraits_::g2sA_tv_tiler;
    using g2sB_tv_tiler = typename MmaTraits_::g2sB_tv_tiler;
    using D_tv_tiler = typename MmaTraits_::D_tv_tiler;
    using g2sA_tiled_copy = typename MmaTraits_::g2sA_tiled_copy;
    using g2sB_tiled_copy = typename MmaTraits_::g2sB_tiled_copy;
    using mma_atom = typename MmaTraits_::mma_atom;
    using tiled_mma = typename MmaTraits_::tiled_mma;
    using s2rA_tiled_copy = typename MmaTraits_::s2rA_tiled_copy;
    using s2rB_tiled_copy = typename MmaTraits_::s2rB_tiled_copy;
    using r2sD_tiled_copy = typename MmaTraits_::r2sD_tiled_copy;
    using s2gD_tiled_copy = typename MmaTraits_::s2gD_tiled_copy;
    using s2gD_tv_tiler = typename MmaTraits_::s2gD_tv_tiler;

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
    const uint32_t base_off_k_tile = bidz * (SplitKSize / BK);
    const uint32_t MMARowPerCTA = MMARowPerWarp * MMARow;

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
        make_shape(Int<NG>{}, M),
        make_stride(M, _1{})
    );
    Tensor mIndex = make_tensor(
        make_gmem_ptr(static_cast<const uint32_t*>(Index)),
        make_shape(Int<NG>{}, M),
        make_stride(M, _1{})
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

    //
    // load mask & index
    // assum the BN = 128, N = 1024, NG = 2,
    // for token skip, the bdimy = 8, NG = 1, NGIter = 8, with coord (bidy // NGIter, ...)
    // for block skip, the bdimy = 8, NG = 2, NGIter = 4, with coord (bidy // NGIter, ...)
    //
    constexpr uint32_t G2SIter = BM / G2SRowPerCTA;
    constexpr uint32_t MMAIter = BM / (MMARowPerWarp * MMARow);
    SkipHelper<G2SIter, MMAIter> skip_helper;
    CUTE_UNROLL
    for (uint32_t i=0, off=thread_off_m; i < G2SIter; ++i, off+=G2SRowPerCTA) {
        skip_helper.rMask[i] = off < M ? mMask(make_coord(bidy / NGIter, off)) : 0;
        skip_helper.rIndex_ld[i] = off < M ? mIndex(make_coord(bidy / NGIter, off)) : 0;
    }

    // check if early exit current block or warp's g2s load
    skip_helper.execute_cta = 0;
    cg::thread_block this_cta = cg::this_thread_block();
    auto this_cta_tile = cg::tiled_partition<threadsPerCTA>(this_cta);
    CUTE_UNROLL
    for (uint32_t i=0; i < G2SIter; ++i) {
        skip_helper.execute_cta |= skip_helper.rMask[i];
        skip_helper.execute_g2s_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>(skip_helper.rMask[i] > 0)));
    }
    skip_helper.execute_cta = cg::reduce(this_cta_tile, skip_helper.execute_cta, cg::bit_or<uint8_t>{});
    if (skip_helper.execute_cta == 0) return;

    // check if mma need execute
    // mma rD layout --> (8, 8), 2, 1, each warp's thread handle 2 row: lane_id / 4 and lane_id / 4 + 8
    CUTE_UNROLL
    for (
        uint32_t i=0,
        off=base_off_m + (warp_id / MMACol) * MMARowPerWarp + lane_id / 4;
        i < MMAIter;
        ++i, off+=MMARowPerCTA
    ) {
        skip_helper.rMask_st[i*2] = off < M ? mMask(make_coord(bidy / NGIter, off)) : 0;
        skip_helper.rMask_st[i*2+1] = off + 8 < M ? mMask(make_coord(bidy / NGIter, off + 8)) : 0;
        // skip_helper.rIndex_st[i*2] = off < M ? mIndex(make_coord(bidy / NGIter, off)) : 0;
        // skip_helper.rIndex_st[i*2+1] = off + 8 < M ? mIndex(make_coord(bidy / NGIter, off + 8)) : 0;
        skip_helper.execute_mma_warp[i] = static_cast<uint8_t>(__any_sync(0xffffffff, static_cast<int>((skip_helper.rMask_st[i*2] > 0) || (skip_helper.rMask_st[i*2+1] > 0))));
    }

    //
    // prologue, load A & B
    //
    uint32_t producer = 0;
    uint32_t consumer = 0;

    Tensor gA = zipped_divide(mA, g2sA_tv_tiler{});
    Tensor gB = local_tile(mB, g2sB_tv_tiler{}, make_coord(bidy, _));
    Tensor gD = local_tile(mD, D_tv_tiler{}, make_coord(bidx, bidy));

    auto g2sA_thr_copy = g2sA_tiled_copy{}.get_slice(tidx % G2SColPerCTA);
    auto g2sB_thr_copy = g2sB_tiled_copy{}.get_slice(tidx);

    auto pDsA = g2sA_thr_copy.partition_D(sA);
    auto pDsB = g2sB_thr_copy.partition_D(sB);
    auto pSgB = g2sB_thr_copy.partition_S(gB);

    // if (tidx == 0 && (bidx + bidy + bidz == 0)) {
    //     cute::print(gB); cute::print("\n");
    //     cute::print(pDsB); cute::print("\n");
    //     cute::print(pSgB); cute::print("\n");
    // }

    // fill the pipeline
    CUTE_UNROLL
    for (uint32_t i=0; i < Pipeline - 1; i++) {
        producer = load_AB<
            decltype(gA),
            decltype(pDsA),
            decltype(pDsB),
            decltype(pSgB),
            decltype(g2sA_thr_copy),
            g2sA_tiled_copy,
            g2sB_tiled_copy,
            Pipeline,
            G2SRowPerCTA,
            G2SColPerCTA,
            G2SIter
        >(
            gA, pDsA, pDsB, pSgB, skip_helper.rIndex_ld, skip_helper.execute_g2s_warp,
            g2sA_thr_copy, g2sA_tiled_copy{}, g2sB_tiled_copy{},
            producer, base_off_k_tile
        );
    }

    //
    // mainloop, issue s2r & mma
    //
    ThrMMA thr_mma = tiled_mma{}.get_slice(tidx);
    auto rA = thr_mma.partition_fragment_A(sA(_, _, _0{}));
    auto rB = thr_mma.partition_fragment_B(sB(_, _, _0{}));
    auto rD = thr_mma.partition_fragment_C(gD);
    clear(rD);

    auto s2rA_thr_copy = s2rA_tiled_copy{}.get_slice(tidx);
    auto pSrA = s2rA_thr_copy.partition_S(sA);
    auto pDrA = s2rA_thr_copy.retile_D(rA);

    auto s2rB_thr_copy = s2rB_tiled_copy{}.get_slice(tidx);
    auto pSrB = s2rB_thr_copy.partition_S(sB);
    auto pDrB = s2rB_thr_copy.retile_D(rB);

    const uint32_t k_tile_num = SplitKSize / BK;
    CUTE_NO_UNROLL
    for (uint32_t kid=0; kid < k_tile_num - (Pipeline - 1); ++kid) {
        producer = load_AB<
            decltype(gA),
            decltype(pDsA),
            decltype(pDsB),
            decltype(pSgB),
            decltype(g2sA_thr_copy),
            g2sA_tiled_copy,
            g2sB_tiled_copy,
            Pipeline,
            G2SRowPerCTA,
            G2SColPerCTA,
            G2SIter
        >(
            gA, pDsA, pDsB, pSgB, skip_helper.rIndex_ld, skip_helper.execute_g2s_warp,
            g2sA_thr_copy, g2sA_tiled_copy{}, g2sB_tiled_copy{},
            producer, base_off_k_tile
        );
        cp_async_wait<Pipeline - 1>();
        __syncthreads();

        consumer = issue_mma<
            decltype(pSrA),
            decltype(pDrA),
            decltype(pSrB),
            decltype(pDrB),
            decltype(rA),
            decltype(rB),
            decltype(rD),
            s2rA_tiled_copy,
            s2rB_tiled_copy,
            tiled_mma,
            Pipeline
        >(
            pSrA, pDrA, pSrB, pDrB, rA, rB, rD,
            skip_helper.execute_mma_warp,
            s2rA_tiled_copy{}, s2rB_tiled_copy{}, tiled_mma{},
            consumer
        );

        __syncthreads();
    }

    // drain the pipeline
    consumer = pipeline_drain<
        Pipeline - 2,
        decltype(pSrA),
        decltype(pDrA),
        decltype(pSrB),
        decltype(pDrB),
        decltype(rA),
        decltype(rB),
        decltype(rD),
        s2rA_tiled_copy,
        s2rB_tiled_copy,
        tiled_mma,
        Pipeline
    >(
        pSrA, pDrA, pSrB, pDrB, rA, rB, rD,
        skip_helper.execute_mma_warp,
        s2rA_tiled_copy{}, s2rB_tiled_copy{}, tiled_mma{},
        consumer
    );

    __syncthreads();

    // epilogue
    // r2s -> s2g, reinterpret the smem to BMxBK
    if (SplitK == 1) {
        ElementWiseActivation<decltype(rD), kAct>{}(rD);
        auto r2sD_thr_copy = r2sD_tiled_copy{}.get_slice(tidx);
        auto pSrD = r2sD_thr_copy.retile_S(rD);
        auto pDsD = r2sD_thr_copy.partition_D(sD);

        auto gD_epi = zipped_divide(mD, s2gD_tv_tiler{});
        auto s2gD_thr_copy = s2gD_tiled_copy{}.get_slice(tidx % G2SColPerCTA);
        
        // r2s
        auto pSrD_tmp = make_tensor_like<DType>(pSrD(make_coord(_, _), 0, 0));
        CUTE_UNROLL
        for (uint32_t i=0; i < size<1>(pSrD); ++i) {
            CUTE_UNROLL
            for (uint32_t j=0; j < size<2>(pSrD); ++j) {
                copy(pSrD(make_coord(_, _), i, j), pSrD_tmp);
                copy(pSrD_tmp, pDsD(make_coord(_, _), i, j));
            }
        }
        __syncthreads();

        // s2g
        CUTE_UNROLL
        for (uint32_t i=0, j=threadIdx.x / G2SColPerCTA; i < G2SIter; ++i, j+=G2SRowPerCTA) {
            if (skip_helper.rMask[i]) {
                auto gDtD = gD_epi(make_coord(_, _), make_coord(skip_helper.rIndex_ld[i], bidy));
                auto pSrD_s2g = s2gD_thr_copy.partition_S(sD);
                auto pDgD = s2gD_thr_copy.partition_D(gDtD);
                copy(s2gD_tiled_copy{}, pSrD_s2g(make_coord(_, _), j, _), pDgD(make_coord(_, _), 0, _));
                // if (tidx == 0 && (bidx + bidy + bidz == 0) && i == 0) {
                //     cute::print(pSrD_s2g); cute::print("\n");
                //     cute::print(gDtD); cute::print("\n");
                //     cute::print(pSrD_s2g(make_coord(_, _), j, _)); cute::print("\n");
                //     cute::print(pDgD(make_coord(_, _), _, _)); cute::print("\n");
                // }

            }
        }
    }
}

template <
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kNGIter,
    uint32_t kBM, uint32_t kBN, uint32_t kBK, uint32_t kPipeline,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA,
    bool kUsePDL, typename DType, uint8_t kAct
>
struct GEMM_MN_V2_Host {
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
        
        TensorMatcher({NG, M}) //
            .with_strides({M, 1})
            .with_dtype<uint8_t>()
            .with_device(device)
            .verify(Mask);
        
        TensorMatcher({NG, M}) //
            .with_strides({M, 1})
            .with_dtype<uint32_t>()
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
            SplitK > 0 && kK % SplitK == 0 && (kK / SplitK) % kBK == 0,
            "SplitK must evenly partition K into complete BK tiles"
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
        using mma_traits = MmaTraits<kBM, kBN, kBK, kPipeline, warp_layout_traits, DType>;
        constexpr auto kernel = gemm_mn_v2_kernel<
            kBM, kBN, kBK, 4, kPipeline, kN, kK, kNG, kNGIter,
            warp_layout_traits, mma_traits, kAct, DType
        >;

        const dim3 grid_size = {cute::ceil_div(params.M, kBM), cute::ceil_div(kN, kBN), SplitK};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};
        LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
    }
};


} // namespace
