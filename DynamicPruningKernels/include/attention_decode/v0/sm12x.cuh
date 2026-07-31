#pragma once
// sglang jit plugin
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <tvm/ffi/container/tensor.h>

#include "attention_decode/v0/traits.cuh"
#include "attention_decode/v0/sm12x_dense.cuh"
#include "attention_decode/v0/sm12x_boundary.cuh"
#include "attention_decode/v0/sm12x_merge.cuh"

namespace {

template <
    uint32_t kHq, uint32_t kHk, uint32_t kD, uint32_t kNG,
    uint32_t kBM, uint32_t kBN,
    uint32_t kPipelineK, uint32_t kPipelineV, uint32_t kSplitRound, uint32_t kWarps,
    bool kIsLeftpad, bool kUsePDL, typename DType
>
struct Attention_Head_V0_Host {
    static void run(
        const tvm::ffi::TensorView Q,
        const tvm::ffi::TensorView K,
        const tvm::ffi::TensorView V,
        const tvm::ffi::TensorView Mask,
        const tvm::ffi::TensorView Leftpad,
        const tvm::ffi::TensorView O,
        const tvm::ffi::TensorView pO,
        const tvm::ffi::TensorView LSE,
        const float scale,
        const float sparsity,
        const uint32_t Split
    ) {
        using namespace host;
        RuntimeCheck(
            sparsity >= 0 && sparsity <= 1,
            "sparsity must be in [0, 1]"
        );

        auto B = SymbolicSize{"batch_size"};
        auto Tk = SymbolicSize{"key_num_tokens"};
        auto Hq = SymbolicSize{"query_num_heads"};
        auto Hk = SymbolicSize{"key_num_heads"};
        auto D = SymbolicSize{"head_dim"};
        auto NG = SymbolicSize{"num_groups"};
        auto SplitKV = SymbolicSize{"split_kv_num"};
        auto device = SymbolicDevice{};

        Hq.set_value(kHq); Hk.set_value(kHk); D.set_value(kD); NG.set_value(kNG);
        SplitKV.set_value(kSplitRound);
        device.set_options<kDLCUDA>();

        // host-side checking
        TensorMatcher({B, 1, Hq, D}) //
            .with_dtype<DType>()
            .with_device(device)
            .verify(Q).verify(O);
        
        TensorMatcher({B, Tk, Hk, D}) //
            .with_dtype<DType>()
            .with_device(device)
            .verify(K).verify(V);
        
        TensorMatcher({B, Hq, SplitKV, D}) //
            .with_dtype<DType>()
            .with_device(device)
            .verify(pO);
        
        TensorMatcher({B, Hq, SplitKV}) //
            .with_dtype<float>()
            .with_device(device)
            .verify(LSE);
        
        TensorMatcher({B, 1, NG}) //
            .with_device(device)
            .verify(Mask);
        
        TensorMatcher({B}) //
            .with_dtype<uint32_t>()
            .with_device(device)
            .verify(Leftpad);
        
        RuntimeCheck(
            sizeof(DType) == 2,
            "DType must be fp16 or bf16"
        );

        const auto batch_size = static_cast<uint32_t>(B.unwrap());
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
        RuntimeCheck(
            Split > 0 && Split <= kSplitRound,
            "Split must be in [1, SplitRound]"
        );

        // host-side static tiling
        using warp_layout_traits = WarpLayoutTraits<kBM, kBN, kD, kWarps, sizeof(DType)>;
        using mma_traits = MMATraits<kBM, kBN, kD, kPipelineK, kPipelineV, warp_layout_traits, DType, kWarps, true>;
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
        
        const auto params_k1 = AttentionParams<decltype(tma_desc_k), decltype(tma_desc_v)>{
            .Q = Q.data_ptr(),
            .K = K.data_ptr(),
            .V = V.data_ptr(),
            .Mask = Mask.data_ptr(),
            .Leftpad = Leftpad.data_ptr(),
            .O = O.data_ptr(),
            .pO = pO.data_ptr(),
            .LSE = LSE.data_ptr(),
            .B = batch_size,
            .Tk = num_key_tokens,
            .Split = Split,
            .scale = scale,
            .tma_desc_k = tma_desc_k,
            .tma_desc_v = tma_desc_v
        };

        const dim3 grid_size = {Split, cute::max(kNG, kHk), params_k1.B};
        const dim3 block_size = {warp_layout_traits::threadsPerCTA, 1, 1};

        constexpr uint32_t MergeThreads = cute::min(128, cute::max(32, kD / 2));
        constexpr uint32_t MergeWarps = MergeThreads / 32;

        const uint32_t num_key_tiles = cute::ceil_div(num_key_tokens, kBN);
        const bool use_dense = !kIsLeftpad && num_key_tokens % kBN == 0 && num_key_tiles % Split == 0;
        if (use_dense) {
            constexpr auto kernel1 = attention_head_v0_dense<
                kHq, kHk, kD, kNG, false,
                kBM, kBN, kPipelineK, kPipelineV, kSplitRound,
                decltype(tma_desc_k), decltype(tma_desc_v),
                warp_layout_traits, mma_traits,
                DType, kWarps
            >;
            cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
            LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel1, params_k1);

        }
        else {
            constexpr auto kernel1 = attention_head_v0_boundary<
                kHq, kHk, kD, kNG, kIsLeftpad,
                kBM, kBN, kPipelineK, kPipelineV, kSplitRound,
                decltype(tma_desc_k), decltype(tma_desc_v),
                warp_layout_traits, mma_traits,
                DType, kWarps
            >;
            cudaFuncSetAttribute(kernel1, cudaFuncAttributeMaxDynamicSharedMemorySize, mma_traits::smem_size);
            LaunchKernel(grid_size, block_size, device.unwrap(), mma_traits::smem_size).enable_pdl(kUsePDL)(kernel1, params_k1);
        }

        if (Split > 1) {
            using merge_traits = MergeTraits<kBM, kBN, kD, kSplitRound, DType, MergeWarps>;
            const auto params_k2 = MergeParams{
                .Mask = Mask.data_ptr(),
                .O = O.data_ptr(),
                .pO = pO.data_ptr(),
                .LSE = LSE.data_ptr(),
                .B = batch_size,
                .Tk = num_key_tokens,
                .Split = Split,
                .scale = scale,
            };
            constexpr auto kernel2 = attention_head_v0_merge<
                kHq, kHk, kD, kNG,
                kBM, kBN,
                kSplitRound, merge_traits,
                DType, MergeWarps
            >;
            cudaFuncSetAttribute(kernel2, cudaFuncAttributeMaxDynamicSharedMemorySize, merge_traits::smem_size);
            LaunchKernel({1, kHq, params_k2.B}, {MergeThreads, 1, 1}, device.unwrap(), merge_traits::smem_size).enable_pdl(kUsePDL)(kernel2, params_k2);
        }
    }
};

} // namespace
