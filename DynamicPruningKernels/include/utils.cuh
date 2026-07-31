#include <cute/tensor.hpp>

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
#define __JIT_USE_VEC_ATOMIC__ 1
#define __JIT_USE_BULK_COPY__ 1
#else
#define __JIT_USE_VEC_ATOMIC__ 0
#define __JIT_USE_BULK_COPY__ 0
#endif

// Cutlass DType
template <typename DType>
struct TmaDescriptorDType {
    using type = DType;
};

template <>
struct TmaDescriptorDType<__half> {
    using type = cute::half_t;
};

template <>
struct TmaDescriptorDType<__nv_bfloat16> {
    using type = cute::bfloat16_t;
};

template <typename DType>
using TmaDescriptorDTypeT = typename TmaDescriptorDType<std::remove_cv_t<DType>>::type;

// Fast math
constexpr uint32_t next_pow_of_2(uint32_t x) {
    if (x <= 0) return 0;
    return 1 << (32 - __builtin_clz(x));
}

constexpr uint32_t prev_pow_of_2(uint32_t x) {
    return x == 0 ? 0 : 1u << (31 - __builtin_clz(x));
}

constexpr bool is_pow_of_2(uint32_t x) {
    return x > 0 && (x & (x - 1)) == 0;
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

// TMA stage & phase
template <uint32_t Stages>
__device__ static __forceinline__ uint32_t pipe_stage(uint32_t iter) {
    if constexpr (Stages == 1) return 0;
    else if constexpr (Stages == 2) return iter & 1u;
    else return iter % Stages;
}

template <uint32_t Stages>
__device__ static __forceinline__ uint32_t pipe_phase(uint32_t iter) {
    if constexpr (Stages == 1) return iter & 1u;
    else if constexpr (Stages == 2) return (iter >> 1) & 1u;
    else return (iter / Stages) & 1u;
}

template <typename DType>
struct FMAF32F16;

template <>
struct FMAF32F16<__half> {
    __device__ __forceinline__ float operator()(__half lhs, __half rhs, float acc) {
        #if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        const uint16_t ulhs = static_cast<__half_raw>(lhs).x;
        const uint16_t urhs = static_cast<__half_raw>(rhs).x;
        asm volatile(
            "fma.rn.f32.f16 %0, %1, %2, %3;"
            : "=f"(acc)
            : "h"(ulhs), "h"(urhs), "f"(acc)
        );
        #else
        acc += __half2float(lhs) * __half2float(rhs);
        #endif
        return acc;
    }
};

template <>
struct FMAF32F16<__nv_bfloat16> {
    __device__ __forceinline__ float operator()(__nv_bfloat16 lhs, __nv_bfloat16 rhs, float acc) {
        #if !defined(__CUDA_ARCH__) || defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
        const uint16_t ulhs = static_cast<__nv_bfloat16_raw>(lhs).x;
        const uint16_t urhs = static_cast<__nv_bfloat16_raw>(rhs).x;
        asm volatile(
            "fma.rn.f32.bf16 %0, %1, %2, %3;"
            : "=f"(acc)
            : "h"(ulhs), "h"(urhs), "f"(acc)
        );
        #else
        acc += __bfloat162float(lhs) * __bfloat162float(rhs);
        #endif
        return acc;
    }
};

// Helper func
// L2 swizzle
template <uint32_t kL2Group>
__device__ __forceinline__ uint2 l2_swizzle(
    const uint32_t bidx, const uint32_t bidy,
    const uint32_t bdimx, const uint32_t bdimy
) {
    if constexpr (kL2Group <= 1) {
        return make_uint2(bidx, bidy);
    } else {
        const uint32_t pid = bidy * bdimx + bidx;
        const uint32_t num_pid_in_group = kL2Group * bdimy;
        const uint32_t group_id = pid / num_pid_in_group;
        const uint32_t first_pid_m = group_id * kL2Group;
        const uint32_t group_size_m = std::min(bdimx - first_pid_m, kL2Group);
        const uint32_t pid_in_group = pid % num_pid_in_group;
        const uint32_t pid_m = first_pid_m + (pid_in_group % group_size_m);
        const uint32_t pid_n = pid_in_group / group_size_m;
        return make_uint2(pid_m, pid_n);
    }
}

// width to dtype
template <uint32_t Bytes>
struct CopyWidthToType;
template <>
struct CopyWidthToType<1> {
    using type = uint8_t;
};
template <>
struct CopyWidthToType<2> {
    using type = uint16_t;
};
template <>
struct CopyWidthToType<4> {
    using type = uint32_t;
};
template <>
struct CopyWidthToType<8> {
    using type = uint64_t;
};
template <>
struct CopyWidthToType<16> {
    using type = cute::uint128_t;
};

// half2float
template <typename DType>
struct Half2Float;
template <>
struct Half2Float<__half> {
    __device__ __forceinline__ float operator()(__half h) {
        return __half2float(h);
    }
};
template <>
struct Half2Float<__nv_bfloat16> {
    __device__ __forceinline__ float operator()(__nv_bfloat16 h) {
        return __bfloat162float(h);
    }
};

// float2half
template <typename DType>
struct Float2Half;
template <>
struct Float2Half<__half> {
    __device__ __forceinline__ __half operator()(float h) {
        return __float2half(h);
    }
};
template <>
struct Float2Half<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat16 operator()(float h) {
        return __float2bfloat16(h);
    }
};

// half packing
template <typename DType>
struct HalfPack;
template <>
struct HalfPack<__half> {
    __device__ __forceinline__ __half2 operator()(__half lhs, __half rhs) {
        return __halves2half2(lhs, rhs);
    }
};
template <>
struct HalfPack<__nv_bfloat16> {
    __device__ __forceinline__ __nv_bfloat162 operator()(__nv_bfloat16 lhs, __nv_bfloat16 rhs) {
        return __halves2bfloat162(lhs, rhs);
    }
};

template <typename DType, bool ReturnPack=false>
struct Float22Half2;
template <>
struct Float22Half2<__half> {
    __device__ __forceinline__ uint32_t operator()(float h1, float h2) {
        auto v = __floats2half2_rn(h1, h2);
        return *reinterpret_cast<uint32_t*>(&v);
    }
};
template <>
struct Float22Half2<__nv_bfloat16> {
    __device__ __forceinline__ uint32_t operator()(float h1, float h2) {
        auto v = __floats2bfloat162_rn(h1, h2);
        return *reinterpret_cast<uint32_t*>(&v);
    }
};
template <>
struct Float22Half2<__half, true> {
    __device__ __forceinline__ __half2 operator()(float h1, float h2) {
        auto v = __floats2half2_rn(h1, h2);
        return v;
    }
};
template <>
struct Float22Half2<__nv_bfloat16, true> {
    __device__ __forceinline__ __nv_bfloat162 operator()(float h1, float h2) {
        auto v = __floats2bfloat162_rn(h1, h2);
        return v;
    }
};

// Epilogue element-wise
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

// PTX
// Packing and Type Conversion
template <typename DType>
struct AccumlatorPack2;
template <>
struct AccumlatorPack2<__half> {
    __device__ __forceinline__ uint32_t operator()(float odd, float even) {
        uint32_t d;
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;"
            : "=r"(d)
            : "f"(odd), "f"(even));
        return d;
    }
    
};
template <>
struct AccumlatorPack2<__nv_bfloat16> {
    __device__ __forceinline__ uint32_t operator()(float odd, float even) {
        uint32_t d;
        asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;"
            : "=r"(d)
            : "f"(odd), "f"(even));
        return d;
    }
};

// Atomic Operation

template <typename DType, uint32_t NPack=2>
struct AtomicAdd;

// 32-bit
#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
template <>
struct AtomicAdd<__half, 2> {
    __device__ __forceinline__ void operator()(uint32_t packed_src, __half* dst) {
        uint32_t* packed_dst = reinterpret_cast<uint32_t*>(dst);
        asm volatile (
            "red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;\n"
            :
            : "l"(packed_dst), "r"(packed_src)
            : "memory"
        );
    }
};
#endif

#if !defined(__CUDA_ARCH__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
template <>
struct AtomicAdd<__nv_bfloat16, 2> {
    __device__ __forceinline__ void operator()(uint32_t packed_src, __nv_bfloat16* dst) {
        uint32_t* packed_dst = reinterpret_cast<uint32_t*>(dst);
        asm volatile (
            "red.relaxed.gpu.global.add.noftz.bf16x2 [%0], %1;\n"
            :
            : "l"(packed_dst), "r"(packed_src)
            : "memory"
        );
    }
};

// 64-bit
template <>
struct AtomicAdd<__half, 4> {
    __device__ __forceinline__ void operator()(const void* src, __half* dst) {
        uint2 packed_src = *reinterpret_cast<const uint2*>(src);
        uint2* packed_dst = reinterpret_cast<uint2*>(dst);
        asm volatile(
            "red.relaxed.gpu.global.add.noftz.v2.f16x2 [%0], {%1, %2};\n"
            :
            : "l"(packed_dst), "r"(packed_src.x), "r"(packed_src.y)
            : "memory"
        );
    }
};

template <>
struct AtomicAdd<__nv_bfloat16, 4> {
    __device__ __forceinline__ void operator()(const void* src, __nv_bfloat16* dst) {
        uint2 packed_src = *reinterpret_cast<const uint2*>(src);
        uint2* packed_dst = reinterpret_cast<uint2*>(dst);
        asm volatile(
            "red.relaxed.gpu.global.add.noftz.v2.bf16x2 [%0], {%1, %2};\n"
            :
            : "l"(packed_dst), "r"(packed_src.x), "r"(packed_src.y)
            : "memory"
        );
    }
};

// 128-bit
template <>
struct AtomicAdd<__half, 8> {
    __device__ __forceinline__ void operator()(const void* src, __half* dst) {
        uint4 packed_src = *reinterpret_cast<const uint4*>(src);
        uint4* packed_dst = reinterpret_cast<uint4*>(dst);
        asm volatile(
            "red.relaxed.gpu.global.add.noftz.v4.f16x2 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(packed_dst), "r"(packed_src.x), "r"(packed_src.y), "r"(packed_src.z), "r"(packed_src.w)
            : "memory"
        );
    }
};

template <>
struct AtomicAdd<__nv_bfloat16, 8> {
    __device__ __forceinline__ void operator()(const void* src, __nv_bfloat16* dst) {
        uint4 packed_src = *reinterpret_cast<const uint4*>(src);
        uint4* packed_dst = reinterpret_cast<uint4*>(dst);
        asm volatile(
            "red.relaxed.gpu.global.add.noftz.v4.bf16x2 [%0], {%1, %2, %3, %4};\n"
            :
            : "l"(packed_dst), "r"(packed_src.x), "r"(packed_src.y), "r"(packed_src.z), "r"(packed_src.w)
            : "memory"
        );
    }
};

// async bulk atomic
template <typename DType, uint32_t NPack>
struct BulkAsyncAtomicAdd;

template <uint32_t NPack>
struct BulkAsyncAtomicAdd<__half, NPack> {
    __device__ __forceinline__ void operator()(const void* src, __half* dst) {
        uint32_t src_smem = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile(
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 "
            "[%0], [%1], %2;\n"
            :
            : "l"(dst), "r"(src_smem), "r"(NPack * 2)
            : "memory"
        );
    }
};

template <uint32_t NPack>
struct BulkAsyncAtomicAdd<__nv_bfloat16, NPack> {
    __device__ __forceinline__ void operator()(const void* src, __nv_bfloat16* dst) {
        uint32_t src_smem = static_cast<uint32_t>(__cvta_generic_to_shared(src));
        asm volatile(
            "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 "
            "[%0], [%1], %2;\n"
            :
            : "l"(dst), "r"(src_smem), "r"(NPack * 2)
            : "memory"
        );
    }
};

__device__ __forceinline__ void cp_reduce_async_bulk_fence() {
    asm volatile("cp.async.bulk.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_reduce_async_bulk_fence_proxy() {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

template <uint32_t N>
__device__ __forceinline__ void cp_reduce_async_bulk_wait() {
    asm volatile("cp.async.bulk.wait_group %0;\n" :: "n"(N) : "memory");
}

template <uint32_t N>
__device__ __forceinline__ void cp_reduce_async_bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ void cp_async_bulk_g2s(
    void* dst_smem, const void* src_gmem, uint32_t bytes, uint64_t* smem_bar
) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    uint32_t bar = static_cast<uint32_t>(__cvta_generic_to_shared(smem_bar));
    asm volatile(
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
        "[%0], [%1], %2, [%3];\n"
        :
        : "r"(dst), "l"(src_gmem), "r"(bytes), "r"(bar)
        : "memory"
    );
}
#endif

// Additional CuTe Wrapper
namespace cute {

struct SM120_U32x4_STSM_N
{
  using SRegisters = uint32_t[4];
  using DRegisters = uint128_t[1];

  CUTE_HOST_DEVICE static void
  copy(uint32_t const& src0, uint32_t const& src1, uint32_t const& src2, uint32_t const& src3,
       uint128_t& smem_dst)
  {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile ("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n"
        :: "r"(smem_int_ptr),
          "r"(src0), "r"(src1), "r"(src2), "r"(src3));
#else
    CUTE_INVALID_CONTROL_PATH("Trying to use stmatrix without device STSM support.");
#endif
  }
};

template <>
struct Copy_Traits<SM120_U32x4_STSM_N>
{
  using ThrID = Layout<_32>;
  using SrcLayout = typename Copy_Traits<SM75_U32x4_LDSM_N>::DstLayout;
  using DstLayout = typename Copy_Traits<SM75_U32x4_LDSM_N>::SrcLayout;
  using RefLayout = SrcLayout;
};

} // end namespace cute
