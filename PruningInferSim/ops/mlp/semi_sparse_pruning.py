from __future__ import annotations

import torch
import random

from collections import OrderedDict
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Sequence, Tuple
from einops import rearrange
from functools import lru_cache
from collections import namedtuple
from torch.sparse.semi_structured import SparseSemiStructuredTensorCUSPARSELT
from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

from .utils import (
    ROOT_PATH,
    THIRD_PARTY_HEADER_DIRS,
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
    DEFAULT_LDFLAGS,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

def random_sample(
    shape,
    sparsity: Optional[float] = 0.5,
    block_size: Optional[int] = 4,
    device: Optional[torch.device | str] = None,
    **kwargs,
) -> torch.Tensor:
    assert sparsity is not None
    assert block_size is not None
    left = int(sparsity * block_size)
    right = block_size
    assert left > 0

    mask = torch.ones(shape, device=device).flatten()
    mask = rearrange(mask, "(a b) -> a b", b=right)
    indices = torch.multinomial(mask, left, replacement=False)

    rows = torch.arange(mask.shape[0], device=device).view(-1, 1).expand(-1, left)
    mask[rows, indices] = False
    mask = mask.reshape(shape)
    return mask.to(torch.bool)

@cache_once
def _jit_cusparselt_spmm_module(
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), dtype)
    return load_jit(
        "cusparselt_spmm",
        *args,
        cuda_files=[str(ROOT_PATH / "jit_kernel" / "cusparselt_spmm.cuh")],
        cuda_wrappers=[
            ("init", f"StructuredSparseKernel<{args}>::init"),
            ("compress", f"StructuredSparseKernel<{args}>::compress"),
            ("update", f"StructuredSparseKernel<{args}>::update"),
            ("run", f"StructuredSparseKernel<{args}>::run"),
            ("destroy", f"StructuredSparseKernel<{args}>::destroy"),
            ("destroy_plan", f"StructuredSparseKernel<{args}>::destroy_plan"),
        ],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
        extra_ldflags=DEFAULT_LDFLAGS,
    )

#########################
# CuSPARSELt JIT Linear
#########################
class CuSPARSELtLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        mask: torch.Tensor,
        cache_num: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight * mask, requires_grad=False)
        self.N, self.K = self.weight.shape
        self._cache_num = cache_num
        self._plan_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._input_scratch_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._output_scratch_cache: OrderedDict[int, torch.Tensor] = OrderedDict()

        dtype = self.weight.dtype
        device = self.weight.device
        self.cusparselt_wrapper = _jit_cusparselt_spmm_module(dtype)

        self._base_state = torch.zeros((1,), dtype=torch.uint64, device="cpu")
        compress_desc = torch.zeros((2,), dtype=torch.int64, device="cpu")
        self.cusparselt_wrapper.init(self.weight, self._base_state, compress_desc)

        compressed_size = int(compress_desc[0].item())
        weight_tmp = torch.empty((compressed_size,), dtype=torch.int8, device=device)
        self.cusparselt_wrapper.compress(
            self.weight, weight_tmp, self._base_state, compress_desc
        )
        self.weight = torch.nn.Parameter(weight_tmp, requires_grad=False)

    @staticmethod
    def _padded_tokens(num_tokens: int) -> int:
        return ((num_tokens + 7) // 8) * 8

    def _pad_input(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        original_m = x.shape[0]
        padded_m = self._padded_tokens(original_m)
        if padded_m == original_m:
            return x, original_m, padded_m
        return x, original_m, padded_m

    def _evict_oldest_plan(self) -> None:
        padded_m, plan_state = self._plan_cache.popitem(last=False)
        self.cusparselt_wrapper.destroy_plan(plan_state, self._base_state)
        # self._input_scratch_cache.pop(padded_m, None)
        # self._output_scratch_cache.pop(padded_m, None)

    def _get_input_scratch(self, padded_m: int, x: torch.Tensor) -> torch.Tensor:
        scratch = self._input_scratch_cache.get(padded_m)
        if scratch is None:
            scratch = torch.empty(
                (padded_m, x.shape[1]), dtype=x.dtype, device=x.device
            )
            self._input_scratch_cache[padded_m] = scratch
        else:
            self._input_scratch_cache.move_to_end(padded_m)
        return scratch

    def _get_output_scratch(self, padded_m: int, x: torch.Tensor) -> torch.Tensor:
        scratch = self._output_scratch_cache.get(padded_m)
        if scratch is None:
            scratch = torch.empty((self.N, padded_m), dtype=x.dtype, device=x.device)
            self._output_scratch_cache[padded_m] = scratch
        else:
            self._output_scratch_cache.move_to_end(padded_m)
        return scratch

    def _get_plan_state(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        plan_state = self._plan_cache.get(M)
        if plan_state is not None:
            self._plan_cache.move_to_end(M)
            return plan_state

        padded_m = self._padded_tokens(M)
        out_nt = torch.empty((self.N, padded_m), dtype=x.dtype, device=x.device)
        plan_state = torch.zeros((1,), dtype=torch.uint64, device="cpu")
        self.cusparselt_wrapper.update(
            x, out_nt, self.weight, self._base_state, plan_state
        )
        out_nt.zero_()

        if len(self._plan_cache) >= self._cache_num and len(self._plan_cache) > 0:
            self._evict_oldest_plan()

        self._plan_cache[M] = plan_state
        return plan_state

    def forward(
        self, x: torch.Tensor, output: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not x.is_contiguous(): x = x.contiguous()
        M_size = x.shape[:2] if x.dim() > 2 else -1
        x = x.flatten(0, 1) if M_size != -1 else x
        x_padded, original_m, padded_m = self._pad_input(x)

        if padded_m != original_m:
            # x_padded = self._get_input_scratch(padded_m, x)
            # x_padded = torch.empty((padded_m, x.shape[1]), dtype=x.dtype, device=x.device)
            # x_padded[:original_m].copy_(x)
            x_padded = torch.nn.functional.pad(x_padded, (0, 0, 0, padded_m - original_m), value=0)
            # x_padded[original_m:].zero_()

        if output is not None:
            assert output.shape == (original_m, self.N)
            assert output.dtype == x.dtype
            assert output.device == x.device
            assert output.is_contiguous()
            out_nt = output.t()
        else:
            out_nt = torch.empty((self.N, padded_m), dtype=x.dtype, device=x.device)

        plan_state = self._get_plan_state(x_padded)
        self.cusparselt_wrapper.run(
            x_padded, self.weight, out_nt, self._base_state, plan_state
        )

        out = out_nt[:, :original_m].t()
        if output is not None:
            output.copy_(out)
            return output
        return out if M_size == -1 else out.reshape(*M_size, -1)

    def __del__(self):
        wrapper = getattr(self, "cusparselt_wrapper", None)
        base_state = getattr(self, "_base_state", None)
        plan_cache = getattr(self, "_plan_cache", None)
        if wrapper is None or base_state is None or plan_cache is None:
            return

        try:
            for plan_state in plan_cache.values():
                wrapper.destroy_plan(plan_state, base_state)
            plan_cache.clear()
            wrapper.destroy(base_state)
        except Exception:
            pass

#########################
# PyTorch cuSPARSELt Linear
#########################
from torch.sparse.semi_structured import SparseSemiStructuredTensor

@lru_cache
def search_for_alg_id(A: SparseSemiStructuredTensor, B_shape: Tuple[int]) -> Tuple[int, int, int]:
    B = torch.rand(B_shape, device=A.device, dtype=A.dtype)
    alg_id, split_k, split_k_mode, _ = torch._C._cusparselt.mm_search(A.packed, B.t(), None, None, None, False)
    print(f"[INFO] spmm ALG_ID = {alg_id}, split_k = {split_k}, split_k_mode = {split_k_mode} for shape A{B.shape} @ B{A.shape}")
    return (alg_id, split_k, split_k_mode)

class TorchCuSPARSELtLinear(torch.nn.Module):
    def __init__(
        self,
        weight: torch.Tensor,
        mask: torch.Tensor,
        cache_num: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight * mask, requires_grad=False)
        self.N, self.K = self.weight.shape

        self.weight = torch.nn.Parameter(SparseSemiStructuredTensorCUSPARSELT.from_dense(self.weight), requires_grad=False)
        self._cache_num = cache_num
        self._plan_cache: OrderedDict[int, Tuple[int, int, int]] = OrderedDict()
    
    def _evict_oldest_plan(self) -> None:
        _ = self._plan_cache.popitem(last=False)
    
    def _get_plan_state(self, x: torch.Tensor) -> torch.Tensor:
        M = x.shape[0]
        plan_state = self._plan_cache.get(M)
        if plan_state is not None:
            self._plan_cache.move_to_end(M)
            return plan_state

        input_shape = list(x.shape)
        plan_state = search_for_alg_id(self.weight, tuple(input_shape))
        if len(self._plan_cache) >= self._cache_num:
            self._evict_oldest_plan()

        self._plan_cache[M] = plan_state
        return plan_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        M_size = x.shape[:2] if x.dim() > 2 else -1
        A = x.flatten(0, 1) if x.dim() > 2 else x

        B = self.weight.data
        A_padded = B._pad_dense_input(A)
        plan_state = self._get_plan_state(A_padded)
        
        # sm80 will failed when split-k > 1
        assert isinstance(B, torch.sparse.SparseSemiStructuredTensor)
        row, col = A.shape
        res = torch._cslt_sparse_mm(
            B.packed,
            A_padded.t(),
            alg_id=plan_state[0],
            split_k=plan_state[1],
            split_k_mode=plan_state[2],
        ).t()
        return res[:row, :] if M_size == -1 else res[:row, :].reshape(*M_size, -1)

def _dense_equivalent_tflops(m: int, n: int, k: int, ms: float) -> float:
    return (2.0 * m * n * k) / (ms * 1e9)

def _benchmark_pad_copy_only(
    linear: CuSPARSELtLinear,
    x: torch.Tensor,
    warmup: int,
    rep: int,
) -> float:
    import triton

    original_m = x.shape[0]
    padded_m = linear._padded_tokens(original_m)
    if padded_m == original_m:
        return 0.0

    x_padded = linear._get_input_scratch(padded_m, x)

    def fn() -> None:
        x_padded[:original_m].copy_(x)
        x_padded[original_m:].zero_()

    with torch.no_grad():
        result = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    assert result is not None
    if isinstance(result, (tuple, list)):
        result = result[0]
    return float(result)


def benchmark_m_staircase(
    dtype: torch.dtype = torch.float16,
    device: str = "cuda:0",
    m_values: Optional[list[int]] = None,
    n: int = 4096,
    k: int = 4096,
    warmup: int = 40,
    rep: int = 80,
) -> None:
    import triton

    quantiles = [0.5, 0.2, 0.8]
    if m_values is None:
        m_values = [2**i for i in range(13)]

    def bench_provider(fn) -> tuple[float, float, float]:
        with torch.no_grad():
            result = triton.testing.do_bench(
                fn, warmup=warmup, rep=rep, quantiles=quantiles
            )
        assert result is not None
        ms, min_ms, max_ms = result
        return float(ms), float(min_ms), float(max_ms)

    print(f"Benchmarking staircase M with fixed N={n}, K={k}")
    weight = torch.rand((n, k), dtype=dtype, device=device)
    mask = random_sample(weight.shape, device=device)
    masked_weight = weight * mask

    dense_linear = torch.nn.Linear(k, n, bias=False, dtype=dtype, device=device)
    dense_linear.weight = torch.nn.Parameter(
        deepcopy(masked_weight), requires_grad=False
    )

    torch_sparse_linear = TorchCuSPARSELtLinear(deepcopy(weight), mask)

    custom_linear = CuSPARSELtLinear(deepcopy(weight), mask)

    for m in m_values:
        x = torch.rand((m, k), dtype=dtype, device=device)
        custom_out = torch.empty((m, n), dtype=dtype, device=device)

        dense_ref = dense_linear(x)
        torch_sparse_ref = torch_sparse_linear(x)
        custom_ref = custom_linear(x, output=custom_out)

        assert torch.allclose(dense_ref, torch_sparse_ref, atol=1e-2, rtol=1e-2)
        assert torch.allclose(dense_ref, custom_ref, atol=1e-2, rtol=1e-2)

        dense_ms, _, _ = bench_provider(lambda: dense_linear(x))
        torch_sparse_ms, _, _ = bench_provider(lambda: torch_sparse_linear(x))
        custom_ms, _, _ = bench_provider(lambda: custom_linear(x, output=custom_out))
        pad_copy_ms = _benchmark_pad_copy_only(custom_linear, x, warmup, rep)

        dense_tflops = _dense_equivalent_tflops(m, n, k, dense_ms)
        torch_sparse_tflops = _dense_equivalent_tflops(m, n, k, torch_sparse_ms)
        custom_tflops = _dense_equivalent_tflops(m, n, k, custom_ms)
        pad_ratio = 0.0 if custom_ms == 0 else (pad_copy_ms / custom_ms) * 100.0

        print(
            f"M={m:4d} | "
            f"dense={dense_ms:7.3f} ms / {dense_tflops:8.3f} TFLOPS | "
            f"torch_cusparselt={torch_sparse_ms:7.3f} ms / {torch_sparse_tflops:8.3f} TFLOPS | "
            f"custom={custom_ms:7.3f} ms / {custom_tflops:8.3f} TFLOPS | "
            f"pad+copy={pad_copy_ms:7.3f} ms ({pad_ratio:5.1f}%)"
        )


if __name__ == "__main__":
    dtype = torch.float16
    device = "cuda:0"

    benchmark_m_staircase(dtype=dtype, device=device, n=4096, k=4096)
    benchmark_m_staircase(dtype=dtype, device=device, n=14336, k=4096)
    benchmark_m_staircase(dtype=dtype, device=device, n=4096, k=14336)
