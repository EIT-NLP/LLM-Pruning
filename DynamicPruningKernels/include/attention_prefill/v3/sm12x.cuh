#pragma once
// sglang jit plugin
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include <type_traits>
#include "attention_prefill/v3/traits.cuh"
#include "attention_prefill/v3/sm12x_dense.cuh"
#include "attention_prefill/v3/sm12x_boundary.cuh"
#include "attention_prefill/v3/sm12x_causal.cuh"

namespace {

template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG,
    uint32_t kBM, uint32_t kBN, uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kWarps,
    bool kIsCausal, bool kIsLeftpad,
    bool kThrSkipG2S, bool kWarpSkipG2S, bool kWarpSkipMMA,
    bool kUsePDL, typename DType
>
struct Attention_Head_V3_Host {
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
        using warp_layout_traits = WarpLayoutTraits<kBM, kBN, kD, kWarps, sizeof(DType)>;
        using mma_traits = MMATraits<kBM, kBN, kD, kPipelineK, kPipelineV, 2, warp_layout_traits, kIsCausal, DType, kWarps, true>;
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

        const dim3 grid_size = {cute::ceil_div(params.Tq, kBM), kHq, params.B};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};

        if constexpr (!kIsCausal) {
            if (!kIsLeftpad && num_key_tokens % kBN == 0) {
                constexpr auto kernel = attention_head_v3_dense<
                    kHq, kHk, kD, kNG, false, false,
                    kBM, kBN, kPipelineK, kPipelineV, 2, decltype(tma_desc_k), decltype(tma_desc_v),
                    warp_layout_traits, mma_traits,
                    kThrSkipG2S, kWarpSkipG2S, kWarpSkipMMA, DType, kWarps
                >;
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
                LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
            }
            else {
                constexpr auto kernel = attention_head_v3_boundary<
                    kHq, kHk, kD, kNG, false, kIsLeftpad,
                    kBM, kBN, kPipelineK, kPipelineV, 2, decltype(tma_desc_k), decltype(tma_desc_v),
                    warp_layout_traits, mma_traits,
                    kThrSkipG2S, kWarpSkipG2S, kWarpSkipMMA, DType, kWarps
                >;
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
                LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
            }
        }
        else {
            constexpr auto kernel = attention_head_v3_causal<
                kHq, kHk, kD, kNG, true, kIsLeftpad,
                kBM, kBN, kPipelineK, kPipelineV, 2, decltype(tma_desc_k), decltype(tma_desc_v),
                warp_layout_traits, mma_traits,
                kThrSkipG2S, kWarpSkipG2S, kWarpSkipMMA, DType, kWarps
            >;
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
            LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel, params);
        }
    }
};

} // namespace