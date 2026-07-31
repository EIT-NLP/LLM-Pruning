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

namespace {

template <uint32_t kG, uint32_t kBN, uint32_t kWarps=4, uint32_t kBytes=2>
struct WarpLayoutTraits {
    static constexpr uint32_t threadsPerCTA = kWarps * device::kWarpThreads;

    // infer the G2RColPerCTA
    static constexpr uint32_t G2RColPerCTA = device::kWarpThreads * kWarps / kBN;
    static constexpr uint32_t G2RRowPerCTA = threadsPerCTA / G2RColPerCTA;
    static constexpr uint32_t G2RCopyWidth = cute::min(16, (kG / G2RColPerCTA) * kBytes);
};

template <uint32_t kG, class WarpLayoutTraits_, typename DType>
struct MmaTraits {
    static constexpr uint32_t G = kG;

    static constexpr uint32_t G2RCopyWidth = WarpLayoutTraits_::G2RCopyWidth;
    static constexpr uint32_t G2RColPerCTA = WarpLayoutTraits_::G2RColPerCTA;
    static constexpr uint32_t G2RRowPerCTA = WarpLayoutTraits_::G2RRowPerCTA;

    // G2R copy
    using g2r_copy_type = typename CopyWidthToType<G2RCopyWidth>::type;
    using g2r_atom = cute::Copy_Atom<cute::UniversalCopy<g2r_copy_type>, DType>;
    using g2rA_tiled_copy = decltype(cute::make_tiled_copy(
        g2r_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<1>{}, cute::Int<G2RColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<G2RCopyWidth / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
    using g2rB_tiled_copy = decltype(cute::make_tiled_copy(
        g2r_atom{},
        cute::make_ordered_layout(
            cute::make_shape(cute::Int<G2RRowPerCTA>{}, cute::Int<G2RColPerCTA>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        ),
        cute::make_ordered_layout(
            cute::make_shape(cute::_1{}, cute::Int<G2RCopyWidth / sizeof(DType)>{}),
            cute::make_step(cute::_1{}, cute::_0{})
        )
    ));
};

struct GEMMParams {
    const void* A;
    const void* B;
    const void* Mask;
    void* D;
    uint32_t M;
};

// main kernel func
template <
    uint32_t kBN, uint32_t kPipelineReg,
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kG,
    class WarpLayoutTraits_, class MmaTraits_, typename DType
>
__global__ void gemv_k_v0_kernel(const __grid_constant__ GEMMParams params) {
    using namespace cute;
    const auto& [A, B, Mask, D, M] = params;
    constexpr uint32_t N = kN;
    constexpr uint32_t K = kK;
    constexpr uint32_t NG = kNG;
    constexpr uint32_t G = kG;
    constexpr uint32_t BN = kBN;
    constexpr uint32_t PipelineReg = kPipelineReg;

    constexpr uint32_t threadsPerCTA = WarpLayoutTraits_::threadsPerCTA;
    constexpr uint32_t G2RColPerCTA = WarpLayoutTraits_::G2RColPerCTA;

    using g2rA_tiled_copy = typename MmaTraits_::g2rA_tiled_copy;
    using g2rB_tiled_copy = typename MmaTraits_::g2rB_tiled_copy;

    // (M, cdiv(N, BN), SplitK)
    const uint32_t tidx = threadIdx.x;
    const uint32_t bidx = blockIdx.x;
    const uint32_t bidy = blockIdx.y;
    const uint32_t warp_id = tidx / device::kWarpThreads;
    const uint32_t lane_id = tidx % device::kWarpThreads;

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
        make_stride(Int<NG>{}, 1)
    );

    //
    // mainloop, issue FMA, on-the-fly skip per G-size k_tile
    //
    Tensor gA = local_tile(
        mA,
        make_tile(_1{}, Int<G>{}),
        make_coord(bidx, _)
    );
    Tensor gB = local_tile(
        mB,
        make_tile(Int<BN>{}, Int<G>{}),
        make_coord(bidy, _)
    );
    Tensor gD = local_tile(
        mD,
        make_tile(_1{}, Int<BN>{}),
        make_coord(bidx, bidy)
    );

    auto g2rA_thr_copy = g2rA_tiled_copy{}.get_slice(tidx % G2RColPerCTA);
    auto g2rB_thr_copy = g2rB_tiled_copy{}.get_slice(tidx);

    auto pSgA = g2rA_thr_copy.partition_S(gA);
    auto pSgB = g2rB_thr_copy.partition_S(gB);

    using rALayout = decltype(pSgA(_, _, _0{}, _0{}).layout());
    using rBLayout = decltype(pSgB(_, _, _0{}, _0{}).layout());
    auto rAPipeLayout = make_layout(
        append(shape(rALayout{}), Int<PipelineReg>{}),
        append(stride(rALayout{}), cute::cosize(rALayout{}))
    );
    auto rBPipeLayout = make_layout(
        append(shape(rBLayout{}), Int<PipelineReg>{}),
        append(stride(rBLayout{}), cute::cosize(rBLayout{}))
    );
    auto rA = make_fragment_like<DType>(rAPipeLayout);
    auto rB = make_fragment_like<DType>(rBPipeLayout);

    float acc = 0;

    CUTE_NO_UNROLL
    for (uint32_t k_tile=0; k_tile < NG; ++k_tile) {
        constexpr uint32_t reg_tile_num = cute::size<2>(pSgB);

        // skip
        uint8_t execute_tile = mMask(make_coord(bidx, k_tile));
        if (execute_tile == 0) continue;

        // Prologue: fill the first PipelineReg - 1 stages.
        CUTE_UNROLL
        for (uint32_t i=0; i < cute::min(PipelineReg - 1, reg_tile_num); ++i) {
            copy(g2rA_tiled_copy{}, pSgA(_, _, i, k_tile), rA(_, _, i));
            copy(g2rB_tiled_copy{}, pSgB(_, _, i, k_tile), rB(_, _, i));
        }
        uint32_t consumer_stage = 0;
        uint32_t producer_stage = PipelineReg - 1;
        uint32_t reg_pipe_next = PipelineReg - 1;

        CUTE_UNROLL
        for (uint32_t reg_pipe=0; reg_pipe < reg_tile_num; ++reg_pipe, ++reg_pipe_next) {
            // Prefetch the next chunk in the same G tile.
            if (reg_pipe_next < reg_tile_num) {
                copy(g2rA_tiled_copy{}, pSgA(_, _, reg_pipe_next, k_tile), rA(_, _, producer_stage));
                copy(g2rB_tiled_copy{}, pSgB(_, _, reg_pipe_next, k_tile), rB(_, _, producer_stage));
                producer_stage = (producer_stage + 1) % PipelineReg;
            }

            CUTE_UNROLL
            for (uint32_t i=0; i < cute::size<0>(rA); ++i) {
                // acc += Half2Float<DType>()(rA(i, 0, consumer_stage)) * Half2Float<DType>()(rB(i, 0, consumer_stage));
                acc = FMAF32F16<DType>()(rA(i, 0, consumer_stage), rB(i, 0, consumer_stage), acc);
            }
            consumer_stage = (consumer_stage + 1) % PipelineReg;
        }
    }

    // final warp shuffle
    CUTE_UNROLL
    for (int32_t i=G2RColPerCTA / 2; i >= 1; i >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, i);
    }

    if (lane_id % G2RColPerCTA == 0) {
        gD(0, tidx / G2RColPerCTA) = Float2Half<DType>()(acc);
    }
}

template <
    uint32_t kN, uint32_t kK, uint32_t kNG, uint32_t kG,
    uint32_t kBN, uint32_t kPipeline,
    bool kUsePDL, typename DType
>
struct GEMV_K_V0_Host {
    static void run(
        const tvm::ffi::TensorView A,
        const tvm::ffi::TensorView B,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView D,
        const float sparsity
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
        auto device = SymbolicDevice{};

        N.set_value(kN);
        K.set_value(kK);
        NG.set_value(kNG);
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
        
        RuntimeCheck(
            sizeof(DType) == 2,
            "DType must be fp16 or bf16"
        );

        const auto num_tokens = static_cast<uint32_t>(M.unwrap());
        RuntimeCheck(
            kK % 32 == 0 && kN % 16 == 0,
            "N and K must be divisible by 16 and 32"
        );

        const auto params = GEMMParams{
            .A = A.data_ptr(),
            .B = B.data_ptr(),
            .Mask = Mask.data_ptr(),
            .D = D.data_ptr(),
            .M = num_tokens
        };

        // host-side static tiling
        using warp_layout_traits = WarpLayoutTraits<kG, kBN, 4, sizeof(DType)>;
        using mma_traits = MmaTraits<kG, warp_layout_traits, DType>;
        constexpr auto kernel = gemv_k_v0_kernel<
            kBN, kPipeline, kN, kK, kNG, kG,
            warp_layout_traits, mma_traits, DType
        >;

        const dim3 grid_size = {num_tokens, cute::ceil_div(kN, kBN), 1};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};
        LaunchKernel(grid_size, block_size, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
    }
};

}