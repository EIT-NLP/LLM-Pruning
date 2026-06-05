#pragma once

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/tile.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <cusparseLt.h>

#include <cstdint>
#include <cstdio>

namespace {

#define CHECK_CUDA(func)                                                                          \
  {                                                                                               \
    cudaError_t status = (func);                                                                  \
    if (status != cudaSuccess) {                                                                  \
      std::printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__,                 \
                  cudaGetErrorString(status), status);                                            \
      return EXIT_FAILURE;                                                                        \
    }                                                                                             \
  }

#define CHECK_CUSPARSE(func)                                                                      \
  {                                                                                               \
    cusparseStatus_t status = (func);                                                             \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                                      \
      std::printf("cuSPARSELt API failed at line %d with status: %d\n", __LINE__, status);     \
      return EXIT_FAILURE;                                                                        \
    }                                                                                             \
  }

template <typename value_t>
struct cuda_type {};

template <>
struct cuda_type<__half> {
  static constexpr cudaDataType value = CUDA_R_16F;
};

template <>
struct cuda_type<__nv_bfloat16> {
  static constexpr cudaDataType value = CUDA_R_16BF;
};

constexpr auto kOrder = CUSPARSE_ORDER_ROW;
constexpr auto kSparseOp = CUSPARSE_OPERATION_NON_TRANSPOSE;
constexpr auto kDenseOp = CUSPARSE_OPERATION_TRANSPOSE;
constexpr float kAlpha = 1.0f;
constexpr float kBeta = 0.0f;
constexpr uint32_t kAlignment = 16;

struct SparseLinearBaseState {
  cusparseLtHandle_t handle;
  cusparseLtMatDescriptor_t b_desc;
  int64_t n = 0;
  int64_t k = 0;
};

struct SparseLinearPlanState {
  cusparseLtMatDescriptor_t a_desc;
  cusparseLtMatDescriptor_t d_desc;
  cusparseLtMatmulDescriptor_t mm_desc;
  cusparseLtMatmulAlgSelection_t alg_desc;
  cusparseLtMatmulPlan_t plan;
  void* workspace = nullptr;
  size_t workspace_size = 0;
  int64_t m = 0;
};

inline auto round_up(int64_t value, int64_t factor) -> int64_t {
  return ((value + factor - 1) / factor) * factor;
}

inline auto get_base_state(const tvm::ffi::TensorView state_tensor) -> SparseLinearBaseState* {
  return reinterpret_cast<SparseLinearBaseState*>(reinterpret_cast<uintptr_t*>(state_tensor.data_ptr())[0]);
}

inline auto get_plan_state(const tvm::ffi::TensorView state_tensor) -> SparseLinearPlanState* {
  return reinterpret_cast<SparseLinearPlanState*>(reinterpret_cast<uintptr_t*>(state_tensor.data_ptr())[0]);
}

inline auto set_state_ptr(const tvm::ffi::TensorView state_tensor, const void* ptr) -> void {
  reinterpret_cast<uintptr_t*>(state_tensor.data_ptr())[0] = reinterpret_cast<uintptr_t>(ptr);
}

inline auto get_current_stream(const tvm::ffi::TensorView tensor) -> cudaStream_t {
  auto device_index = static_cast<c10::DeviceIndex>(tensor.device().device_id);
  return c10::cuda::getCurrentCUDAStream(device_index).stream();
}

template <bool kUsePDL, typename DType>
struct StructuredSparseKernel {
  static int init(const tvm::ffi::TensorView B, const tvm::ffi::TensorView base_state,
                  const tvm::ffi::TensorView compress_desc) {
    using namespace host;
    auto N = SymbolicSize{"intermediate_size"};
    auto K = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    auto host_device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    host_device.set_options<kDLCPU>();

    TensorMatcher({N, K}).with_strides({K, 1}).with_dtype<DType>().with_device(device).verify(B);
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(base_state);
    TensorMatcher({2}).with_strides({1}).with_dtype<int64_t>().with_device(host_device).verify(compress_desc);

    auto* state = new SparseLinearBaseState();
    const auto kN = static_cast<int64_t>(N.unwrap());
    const auto kK = static_cast<int64_t>(K.unwrap());
    auto stream = get_current_stream(B);

    state->n = kN;
    state->k = kK;

    CHECK_CUSPARSE(cusparseLtInit(&state->handle));
    CHECK_CUSPARSE(cusparseLtStructuredDescriptorInit(&state->handle, &state->b_desc, kN, kK, kK, kAlignment,
                                                      cuda_type<DType>::value, kOrder,
                                                      CUSPARSELT_SPARSITY_50_PERCENT));

    int* is_valid_gpu = nullptr;
    int is_valid = 0;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&is_valid_gpu), sizeof(int)));
    CHECK_CUSPARSE(cusparseLtSpMMAPruneCheck2(&state->handle, &state->b_desc, 1, kSparseOp, B.data_ptr(), is_valid_gpu,
                                             stream));
    CHECK_CUDA(cudaMemcpyAsync(&is_valid, is_valid_gpu, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaFree(is_valid_gpu));
    if (is_valid != 0) {
      std::printf("Input weight is not a valid 2:4 semi-structured sparse matrix for cuSPARSELt\n");
      delete state;
      return EXIT_FAILURE;
    }

    size_t compressed_size = 0;
    size_t compressed_buffer_size = 0;
    CHECK_CUSPARSE(cusparseLtSpMMACompressedSize2(&state->handle, &state->b_desc, &compressed_size,
                                                  &compressed_buffer_size));

    reinterpret_cast<int64_t*>(compress_desc.data_ptr())[0] = static_cast<int64_t>(compressed_size);
    reinterpret_cast<int64_t*>(compress_desc.data_ptr())[1] = static_cast<int64_t>(compressed_buffer_size);
    set_state_ptr(base_state, state);
    return 0;
  }

  static int compress(const tvm::ffi::TensorView B, const tvm::ffi::TensorView dB,
                      const tvm::ffi::TensorView base_state, const tvm::ffi::TensorView compress_desc) {
    using namespace host;
    auto N = SymbolicSize{"intermediate_size"};
    auto K = SymbolicSize{"hidden_size"};
    auto device = SymbolicDevice{};
    auto host_device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    host_device.set_options<kDLCPU>();

    TensorMatcher({N, K}).with_strides({K, 1}).with_dtype<DType>().with_device(device).verify(B);
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(base_state);
    TensorMatcher({2}).with_strides({1}).with_dtype<int64_t>().with_device(host_device).verify(compress_desc);

    auto* state = get_base_state(base_state);
    host::RuntimeCheck(state != nullptr, "SparseLinearBaseState is not initialized");
    auto stream = get_current_stream(B);

    void* compressed_buffer = nullptr;
    const auto compressed_buffer_size =
        static_cast<size_t>(reinterpret_cast<int64_t*>(compress_desc.data_ptr())[1]);
    if (compressed_buffer_size > 0) {
      CHECK_CUDA(cudaMalloc(&compressed_buffer, compressed_buffer_size));
    }

    CHECK_CUSPARSE(cusparseLtSpMMACompress2(&state->handle, &state->b_desc, 1, kSparseOp, B.data_ptr(), dB.data_ptr(),
                                            compressed_buffer, stream));

    if (compressed_buffer != nullptr) {
      CHECK_CUDA(cudaFree(compressed_buffer));
    }
    return 0;
  }

  static int update(const tvm::ffi::TensorView A, const tvm::ffi::TensorView D, const tvm::ffi::TensorView dB,
                    const tvm::ffi::TensorView base_state, const tvm::ffi::TensorView plan_state) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto N = SymbolicSize{"intermediate_size"};
    auto P = SymbolicSize{"padded_num_tokens"};
    auto device = SymbolicDevice{};
    auto host_device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    host_device.set_options<kDLCPU>();

    TensorMatcher({M, K}).with_strides({K, 1}).with_dtype<DType>().with_device(device).verify(A);
    TensorMatcher({N, P}).with_strides({P, 1}).with_dtype<DType>().with_device(device).verify(D);
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(base_state)
        .verify(plan_state);

    auto* shared = get_base_state(base_state);
    host::RuntimeCheck(shared != nullptr, "SparseLinearBaseState is not initialized");
    auto stream = get_current_stream(A);

    auto* state = new SparseLinearPlanState();
    const auto kM = static_cast<int64_t>(M.unwrap());
    const auto kK = static_cast<int64_t>(K.unwrap());
    const auto kN = static_cast<int64_t>(N.unwrap());
    const auto kP = static_cast<int64_t>(P.unwrap());
    host::RuntimeCheck(kP >= kM, "Output leading dimension must be >= logical M");
    host::RuntimeCheck(kP % (kAlignment / static_cast<uint32_t>(sizeof(DType))) == 0,
                       "Output leading dimension must satisfy cuSPARSELt alignment");
    state->m = kM;

    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&shared->handle, &state->a_desc, kM, kK, kK, kAlignment,
                                                 cuda_type<DType>::value, kOrder));
    CHECK_CUSPARSE(cusparseLtDenseDescriptorInit(&shared->handle, &state->d_desc, kN, kM, kP, kAlignment,
                                                 cuda_type<DType>::value, kOrder));
    CHECK_CUSPARSE(cusparseLtMatmulDescriptorInit(&shared->handle, &state->mm_desc, kSparseOp, kDenseOp,
                                                  &shared->b_desc, &state->a_desc, &state->d_desc, &state->d_desc,
                                                  CUSPARSE_COMPUTE_32F));

    void* sparse_ptr = dB.data_ptr();
    CHECK_CUSPARSE(cusparseLtMatmulDescSetAttribute(&shared->handle, &state->mm_desc,
                                                    CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &sparse_ptr,
                                                    sizeof(sparse_ptr)));
    CHECK_CUSPARSE(cusparseLtMatmulAlgSelectionInit(&shared->handle, &state->alg_desc, &state->mm_desc,
                                                    CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSE(cusparseLtMatmulPlanInit(&shared->handle, &state->plan, &state->mm_desc, &state->alg_desc));

    CHECK_CUSPARSE(cusparseLtMatmulSearch(&shared->handle, &state->plan, &kAlpha, dB.data_ptr(), A.data_ptr(), &kBeta,
                                          D.data_ptr(), D.data_ptr(), nullptr, &stream, 1));

    CHECK_CUSPARSE(cusparseLtMatmulGetWorkspace(&shared->handle, &state->plan, &state->workspace_size));
    if (state->workspace_size > 0) {
      CHECK_CUDA(cudaMalloc(&state->workspace, state->workspace_size));
    }

    set_state_ptr(plan_state, state);
    return 0;
  }

  static int run(const tvm::ffi::TensorView A, const tvm::ffi::TensorView dB, const tvm::ffi::TensorView D,
                 const tvm::ffi::TensorView base_state, const tvm::ffi::TensorView plan_state) {
    using namespace host;
    auto M = SymbolicSize{"num_tokens"};
    auto K = SymbolicSize{"hidden_size"};
    auto N = SymbolicSize{"intermediate_size"};
    auto P = SymbolicSize{"padded_num_tokens"};
    auto device = SymbolicDevice{};
    auto host_device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    host_device.set_options<kDLCPU>();

    TensorMatcher({M, K}).with_strides({K, 1}).with_dtype<DType>().with_device(device).verify(A);
    TensorMatcher({N, P}).with_strides({P, 1}).with_dtype<DType>().with_device(device).verify(D);
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(base_state)
        .verify(plan_state);

    auto* shared = get_base_state(base_state);
    auto* state = get_plan_state(plan_state);
    host::RuntimeCheck(shared != nullptr, "SparseLinearBaseState is not initialized");
    host::RuntimeCheck(state != nullptr, "SparseLinearPlanState is not initialized");
    auto stream = get_current_stream(A);

    CHECK_CUSPARSE(cusparseLtMatmul(&shared->handle, &state->plan, &kAlpha, dB.data_ptr(), A.data_ptr(), &kBeta,
                                    D.data_ptr(), D.data_ptr(), state->workspace, &stream, 1));
    return 0;
  }

  static int destroy(const tvm::ffi::TensorView base_state) {
    using namespace host;
    auto host_device = SymbolicDevice{};
    host_device.set_options<kDLCPU>();
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(base_state);

    auto* state = get_base_state(base_state);
    if (state == nullptr) {
      return 0;
    }

    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&state->b_desc));
    CHECK_CUSPARSE(cusparseLtDestroy(&state->handle));
    delete state;
    set_state_ptr(base_state, nullptr);
    return 0;
  }

  static int destroy_plan(const tvm::ffi::TensorView plan_state, const tvm::ffi::TensorView base_state) {
    using namespace host;
    auto host_device = SymbolicDevice{};
    host_device.set_options<kDLCPU>();
    TensorMatcher({1}).with_strides({1}).with_dtype<uint64_t>().with_device(host_device).verify(plan_state)
        .verify(base_state);

    auto* shared = get_base_state(base_state);
    auto* state = get_plan_state(plan_state);
    if (state == nullptr) {
      return 0;
    }

    if (state->workspace != nullptr) {
      CHECK_CUDA(cudaFree(state->workspace));
    }
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&state->a_desc));
    CHECK_CUSPARSE(cusparseLtMatDescriptorDestroy(&state->d_desc));
    CHECK_CUSPARSE(cusparseLtMatmulPlanDestroy(&state->plan));
    (void)shared;
    delete state;
    set_state_ptr(plan_state, nullptr);
    return 0;
  }
};

}  // namespace
