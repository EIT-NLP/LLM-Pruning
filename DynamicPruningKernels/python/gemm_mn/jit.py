from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Any, List, Mapping
from einops import rearrange
from copy import deepcopy
from functools import lru_cache

import torch

from sglang.jit_kernel.utils import (
    cache_once,
    is_arch_support_pdl,
    make_cpp_args,
)

from ..utils import (
    ROOT_PATH,
    THIRD_PARTY_HEADER_DIRS,
    DEFAULT_CFLAGS,
    DEFAULT_CUDA_CFLAGS,
    STYLES,
    load_jit_for_arch,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

from ..utils import (
    ACTIVATION,
    _get_smem_size,
    next_pow_of_2,
    cdiv,
)
from ..autotune import JitConfig, autotune, wrap_jit_module
from ..catalog import cute_architectures, cute_versions
from .triton import gemm_mn_host as triton_ref_program

__SUPPORT_VERSION__ = list(cute_versions("gemm_mn"))
__SUPPORT_ARCH__ = list(cute_architectures("gemm_mn"))
_GEMV_MN_MAX_M = 4

############################### GEMM JIT KERNEL WRAPPER ###############################

### Default Search Space
@lru_cache
def gemm_mn_space():
    BM = [16, 32, 64, 128, 256]
    BN = [32, 64, 128, 256]
    BK = [32, 64]
    SplitK = [1, 2, 4, 8, 16, 32]
    Pipeline = [2, 3]
    ThrSkipG2S = [False, True]
    WarpSkipG2S = [False]
    WarpSkipMMA = [False]

    return [
        dict(
            BM=bm,
            BN=bn,
            BK=bk,
            SplitK=splitk,
            Pipeline=pipeline,
            ThrSkipG2S=thrskipg2s,
            WarpSkipG2S=warpskipg2s,
            WarpSkipMMA=warpskipmma,
        )
        for bm in BM
        for bn in BN
        for bk in BK
        for splitk in SplitK
        for pipeline in Pipeline
        for thrskipg2s in ThrSkipG2S
        for warpskipg2s in WarpSkipG2S
        for warpskipmma in WarpSkipMMA
    ]

### Search Space Pruning
def prune_gemm_mn_space(configs: List[JitConfig], named_args: Mapping[str, Any]) -> list[JitConfig]:
    cc = int(named_args["cc"])
    smem_size = int(named_args["smem_size"])
    num_sm = int(named_args["num_sm"])
    data_width = int(named_args["data_width"])

    rM = int(named_args["rM"])
    N = int(named_args["N"])
    K = int(named_args["K"])
    G = int(named_args["G"])
    filtered: list[JitConfig] = []

    def get_mn_candidate(x: int, max_mn: int) -> set[int]:
        candidate = min(max_mn, max(16, next_pow_of_2(x)))
        return {candidate, max(16, candidate // 2)}

    max_mn = 256 if cc // 10 in [9, 10] else 128
    bm_candidate, bn_candidate = get_mn_candidate(rM, max_mn), get_mn_candidate(G, max_mn)

    for config in configs:
        kwargs = config.kwargs
        BM = int(kwargs["BM"])
        BN = int(kwargs["BN"])
        BK = int(kwargs["BK"])
        SplitK = int(kwargs["SplitK"])
        Pipeline = int(kwargs["Pipeline"])
        if BM not in bm_candidate or BN not in bn_candidate or BK < 32: continue
        if G % BN != 0: continue
        if min(BM, BN) <= 16 and (BK < 64 and kwargs['ThrSkipG2S'] == False): continue
        if data_width * Pipeline * (BM * BK + BN * BK) > smem_size: continue
        if cc // 10 in [8, 12] and data_width * Pipeline * (BM * BK + BN * BK) > smem_size / 2: continue
        if K % SplitK != 0 or (K // SplitK) % BK != 0: continue
        filtered.append(config)

    return filtered

### Heuristic Fallback
def heuristic_gemm_mn_config(named_args: Mapping[str, Any]) -> dict[str, Any]:
    cc = int(named_args["cc"])
    smem_size = int(named_args["smem_size"])
    num_sm = int(named_args["num_sm"])
    data_width = int(named_args["data_width"])

    rM = int(named_args["rM"])
    N = int(named_args["N"])
    K = int(named_args["K"])
    G = int(named_args["G"])

    cc_major = cc // 10
    if cc_major not in (8, 12):
        if cc_major == 9:
            raise NotImplementedError("sm90-like wgmma is not supported")
        if cc_major == 10:
            raise NotImplementedError("sm100-like umma is not supported")
        raise ValueError(f"Unsupported compute capability {cc}")

    BM = min(256, max(16, rM))
    BK = 64
    BN = min(G, 256)
    Pipeline = 3

    assert BN >= 16, "group size must be at least 16 for sm80-like warp mma"

    while data_width * Pipeline * (BM * BK + BN * BK) > smem_size / 2:
        if BN > 128:
            BN >>= 1
        elif BK > 32 and BN > 16 and BM > 16:
            BK >>= 1
        elif (BM > 16 and BK == 32) or (BM > 32 and BK == 64):
            BM >>= 1
        elif (BN > 16 and BK == 32) or (BN > 32 and BK == 64):
            BN >>= 1
        else:
            break

    SplitK = 1
    base_tile_num = cdiv(rM, BM) * cdiv(N, BN)
    if base_tile_num < num_sm and K >= 8 * BK:
        min_waste = 1.0
        best_split_k = 1
        for split_k in [1, 2, 4, 8]:
            if K % split_k != 0 or (K // split_k) % BK != 0:
                continue
            waste = float(num_sm - ((base_tile_num * split_k) % num_sm)) / float(num_sm)
            if min_waste > 0 and waste < min_waste:
                min_waste = waste
                best_split_k = split_k
        SplitK = best_split_k

    return {
        "BM": BM,
        "BN": BN,
        "BK": BK,
        "SplitK": SplitK,
        "Pipeline": Pipeline,
        "ThrSkipG2S": False if rM > 16 else True,
        "WarpSkipG2S": False,
        "WarpSkipMMA": False,
    }

### Compile Module
@cache_once
def _compile_gemm_mn_module(
    N: int, K: int, NG: int, G: int, BM: int, BN: int, BK: int, Pipeline: int,
    ThrSkipG2S: bool, WarpSkipG2S: bool, WarpSkipMMA: bool,
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
    activation: Optional[str]="identity",
) -> Module:
    # check kernel version and arch
    if (kernel_version == -1 or (kernel_version < len(__SUPPORT_VERSION__) and kernel_version >= 0)):
        kernel_version = __SUPPORT_VERSION__[kernel_version]
    else:
        raise ValueError(f"Unsupported kernel version '{kernel_version}'. Available versions: {__SUPPORT_VERSION__}")

    if arch not in __SUPPORT_ARCH__:
        raise ValueError(f"Unsupported arch '{arch}'. Available archs: {__SUPPORT_ARCH__}")

    cpp_args = [N, K, NG, G // BN, BM, BN, BK, Pipeline, ThrSkipG2S, WarpSkipG2S, WarpSkipMMA] + [is_arch_support_pdl(), dtype, ACTIVATION.get(activation, 0)]
    cpp_args = make_cpp_args(*cpp_args)

    return load_jit_for_arch(
        arch,
        "gemm_mn",
        *cpp_args,
        cuda_files=[str(ROOT_PATH / "include" / f"gemm_mn/v{kernel_version}/{arch}.cuh")],
        cuda_wrappers=[("kernel", f"GEMM_MN_V{kernel_version}_Host<{cpp_args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )

### Autotune Wrapper
@autotune(
    kernel_id="gemm_mn",
    config_params=['BM', 'BN', 'BK', 'SplitK', 'Pipeline', 'ThrSkipG2S', 'WarpSkipG2S', 'WarpSkipMMA'],
    runtime_config_params=['SplitK'],
    configs=gemm_mn_space,
    key=['rM', 'N', 'K', 'NG', 'G', 'dtype', 'kernel_version', 'arch', 'activation', 'estimate_sparsity', 'cc'],
    prune_configs_by=prune_gemm_mn_space,
    runtime_params=["A_f", "B", "Mask_s", "Index", "D", "estimate_sparsity"],
    restore_params=["D"],
    heuristic=heuristic_gemm_mn_config,
    cudagraph=True,
)
def _jit_gemm_mn_module(
    N: int, K: int, NG: int, G: int, BM: int, BN: int, BK: int, Pipeline: int,
    ThrSkipG2S: bool, WarpSkipG2S: bool, WarpSkipMMA: bool,
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
    activation: Optional[str]="identity",
) -> Module:
    return _compile_gemm_mn_module(
        N, K, NG, G, BM, BN, BK, Pipeline,
        ThrSkipG2S, WarpSkipG2S, WarpSkipMMA,
        kernel_version, arch,
        dtype=dtype,
        activation=activation,
    )

############################### GEMV JIT KERNEL WRAPPER ###############################

### Default Search Space
@lru_cache
def gemv_mn_space():
    BN = [4, 8, 16, 32]
    BK = [128, 256, 512, 1024, 2048, 4096]
    Pipeline = [1, 2, 3]

    return [
        dict(
            BN=bn,
            BK=bk,
            Pipeline=pipeline,
        )
        for bn in BN
        for bk in BK
        for pipeline in Pipeline
    ]

### Search Space Pruning
def prune_gemv_mn_space(configs: List[JitConfig], named_args: Mapping[str, Any]) -> list[JitConfig]:
    K = int(named_args["K"])
    max_k = min(4096, K & -K)
    filtered: list[JitConfig] = []
    
    for config in configs:
        kwargs = config.kwargs
        BK = int(kwargs["BK"])
        if BK != max_k: continue
        filtered.append(config)

    return filtered

### Heuristic Fallback
def heuristic_gemv_mn_config(named_args: Mapping[str, Any]) -> dict[str, Any]:
    cc = int(named_args["cc"])

    K = int(named_args["K"])

    cc_major = cc // 10
    if cc_major not in (8, 12):
        if cc_major == 9:
            raise NotImplementedError("sm90-like wgmma is not supported")
        if cc_major == 10:
            raise NotImplementedError("sm100-like umma is not supported")
        raise ValueError(f"Unsupported compute capability {cc}")

    BK = min(4096, K & -K)
    BN = 16
    Pipeline = 3

    return {
        "BN": BN,
        "BK": BK,
        "Pipeline": Pipeline,
    }

### Compile Module
@cache_once
def _compile_gemv_mn_module(
    N: int, K: int, NG: int, G: int, BN: int, BK: int, Pipeline: int,
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
    activation: Optional[str]="identity",
) -> Module:
    # check kernel version and arch
    if (kernel_version == -1 or (kernel_version < len(__SUPPORT_VERSION__) and kernel_version >= 0)):
        kernel_version = __SUPPORT_VERSION__[kernel_version]
    else:
        raise ValueError(f"Unsupported kernel version '{kernel_version}'. Available versions: {__SUPPORT_VERSION__}")

    if arch not in __SUPPORT_ARCH__:
        raise ValueError(f"Unsupported arch '{arch}'. Available archs: {__SUPPORT_ARCH__}")

    cpp_args = [N, K, NG, G // BN, BN, BK, Pipeline] + [is_arch_support_pdl(), dtype, ACTIVATION.get(activation, 0)]
    cpp_args = make_cpp_args(*cpp_args)

    return load_jit_for_arch(
        arch,
        "gemv_mn",
        *cpp_args,
        cuda_files=[str(ROOT_PATH / "include" / f"gemv_mn/v{kernel_version}/{arch}.cuh")],
        cuda_wrappers=[("kernel", f"GEMV_MN_V{kernel_version}_Host<{cpp_args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )

### Autotune Wrapper
@autotune(
    kernel_id="gemv_mn",
    config_params=['BN', 'BK', 'Pipeline'],
    configs=gemv_mn_space,
    key=['M', 'N', 'K', 'NG', 'G', 'dtype', 'kernel_version', 'arch', 'activation', 'estimate_sparsity', 'cc'],
    prune_configs_by=prune_gemv_mn_space,
    runtime_params=["A_f", "B", "Mask_f", "D", "estimate_sparsity"],
    restore_params=["D"],
    heuristic=heuristic_gemv_mn_config,
    cudagraph=True,
)
def _jit_gemv_mn_module(
    N: int, K: int, NG: int, G: int, BN: int, BK: int, Pipeline: int,
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
    activation: Optional[str]="identity",
) -> Module:
    return _compile_gemv_mn_module(
        N, K, NG, G, BN, BK, Pipeline,
        kernel_version, arch,
        dtype=dtype,
        activation=activation,
    )

def gemm_mn(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor]=None,
    sorted_mask: Optional[torch.Tensor]=None,
    sorted_indices: Optional[torch.Tensor]=None,
    activation: Optional[str]="identity",
    estimate_sparsity: Optional[float]=0.5,
    kernel_version: Optional[int]=1,
    arch: Optional[str]="sm8x",
    autotune: Optional[bool]=True,
    cudagraph: bool=False,
) -> torch.Tensor:
    assert A.dim() == 3 and B.dim() == 2
    assert Mask.dim() in (2, 3)
    assert estimate_sparsity <= 1
    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity

    Bsz, T, K = A.shape
    A_f = A.flatten(0, 1)

    M, _ = A_f.shape
    N, _ = B.shape
    if D is None: D = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    NG = Mask.shape[-1]
    G = N // NG

    if Mask.dim() == 2: Mask = Mask.unsqueeze(-1) # [B, T, NG]
    metadata = {}
    if M <= _GEMV_MN_MAX_M:
        Mask_f = Mask.flatten(0, 1)
        module = _jit_gemv_mn_module(
            M=M,
            N=N,
            K=K,
            NG=NG,
            G=G,
            kernel_version=0,
            arch='sm8x',
            dtype=A.dtype,
            activation=activation,
            A_f=A_f,
            B=B,
            Mask_f=Mask_f,
            D=D,
            estimate_sparsity=estimate_sparsity,
            autotune=autotune,
            cudagraph=cudagraph,
        )
        module.run(A_f, B, Mask_f, D, estimate_sparsity)
    else:
        if sorted_indices is None:
            Mask_s, Index = torch.sort(Mask.flatten(0, 1), dim=0, descending=True, stable=False)
        else:
            Mask_s = sorted_mask.view(-1, Mask.shape[-1])
            Index = sorted_indices.view(-1, Mask.shape[-1])
        metadata = dict(sorted_mask=Mask_s, sorted_indices=Index)

        kernel_version = __SUPPORT_VERSION__[kernel_version] if kernel_version == -1 or kernel_version < len(__SUPPORT_VERSION__) else kernel_version

        module = _jit_gemm_mn_module(
            rM=next_pow_of_2(M),
            N=N,
            K=K,
            NG=NG,
            G=G,
            kernel_version=kernel_version,
            arch=arch,
            dtype=A.dtype,
            activation=activation,
            A_f=A_f,
            B=B,
            Mask_s=Mask_s,
            Index=Index,
            D=D,
            estimate_sparsity=estimate_sparsity,
            autotune=autotune,
            cudagraph=cudagraph,
        )
        module.run(A_f, B, Mask_s, Index, D, estimate_sparsity)

    return D.reshape(Bsz, T, N), metadata

def torch_ref_program(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: Optional[torch.Tensor]=None,
    D: Optional[torch.Tensor]=None,
    activation: Optional[str]="identity",
    estimate_sparsity: Optional[float]=0.5,
    **kwargs,
) -> torch.Tensor:
    D = A.flatten(0, 1) @ B.T
    if Mask is not None:
        D = rearrange(D, '(b t) (ng d) -> b t ng d', b=A.shape[0], ng=Mask.shape[-1])
        if Mask.dim() == 2: Mask = Mask.unsqueeze(-1) # [B, T, NG]

        D.masked_fill_(Mask.logical_not()[:, :, :, None], 0)
        D = rearrange(D, 'b t ng d -> b t (ng d)')
    else:
        D = rearrange(D, '(b t) d -> b t d', b=A.shape[0])

    if activation == 'silu': D = D * torch.sigmoid(D)
    elif activation == 'relu': D = torch.relu(D)

    return D

def run_once(**kwargs):
    import math
    from torch.nn import init as torch_init

    device = kwargs.get("device", "cuda:0")
    if torch.device(device).type == "cuda": torch.cuda.set_device(torch.device(device))
    dtype = kwargs.get("dtype", torch.float16)

    M = kwargs.get("M", [8192])[-1]
    N = kwargs.get("N", 8192)
    K = kwargs.get("K", 8192)
    G = kwargs.get("G", 128)

    sparsity = kwargs.get("sparsity", [0.5])[-1]

    check_precision = kwargs.get("check_precision", True)

    A = torch.randn((1, M, K), dtype=dtype, device=device)
    B = torch.empty((N, K), dtype=dtype, device=device)
    Mask = torch.rand((1, M, N // G), device=device) >= sparsity

    torch_init.kaiming_uniform_(B, a=math.sqrt(5))

    compile_kwargs = dict(
        arch=kwargs.get("arch"),
        autotune=kwargs.get("autotune", True),
        cudagraph=kwargs.get("cudagraph", False),
    )

    D_triton = triton_ref_program(
        deepcopy(A), deepcopy(B), deepcopy(Mask), None, activation='identity', estimate_sparsity=sparsity
    )
    D_torch = torch_ref_program(
        deepcopy(A), deepcopy(B), deepcopy(Mask), None, activation='identity', estimate_sparsity=sparsity
    )

    for version in kwargs.get('kernel_version'):
        kernel_version = __SUPPORT_VERSION__[version]
        compile_kwargs["kernel_version"] = kernel_version

        D_cute, _ = gemm_mn(
            deepcopy(A), deepcopy(B), deepcopy(Mask), None, activation='identity', estimate_sparsity=sparsity, **compile_kwargs
        )

        if check_precision:
            diff = (D_triton - D_torch).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"[INFO] triton vs torch: max_diff: {max_diff.item()}, mean_diff: {mean_diff.item()}")

            diff = (D_cute - D_torch).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"[INFO] cute_v{kernel_version} vs torch: max_diff: {max_diff.item()}, mean_diff: {mean_diff.item()}")

            diff = (D_triton - D_cute).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"[INFO] triton vs cute_v{kernel_version}: max_diff: {max_diff.item()}, mean_diff: {mean_diff.item()}")

def _benchmark_interface(**kwargs):
    import math
    from functools import partial
    from torch.nn import init as torch_init
    from triton.testing import Benchmark, perf_report, do_bench, do_bench_cudagraph

    device = kwargs.get("device", "cuda:0")
    if torch.device(device).type == "cuda": torch.cuda.set_device(torch.device(device))
    dtype = kwargs.get("dtype", torch.float16)

    M = kwargs.get("M", [8192])
    N = kwargs.get("N", 8192)
    K = kwargs.get("K", 8192)
    G = kwargs.get("G", 128)
    sparsity = kwargs.get("sparsity", [0.5])

    assert not (len(M) > 1 and len(sparsity) > 1), "Each profiling step only support one variable"

    variable = dict(M=M)
    if len(sparsity) > 1: variable = dict(sparsity=sparsity)

    A = torch.randn((1, M[-1], K), dtype=dtype, device=device)
    B = torch.empty((N, K), dtype=dtype, device=device)
    torch_init.kaiming_uniform_(B, a=math.sqrt(5))
    Mask = torch.rand((1, M[-1], N // G), device='cpu')

    benchmark_kwargs = dict(
        M=M[-1], N=N, K=K, G=G, A=A, B=B, Mask=Mask, sparsity=sparsity[-1],
        check_precision=kwargs.get("check_precision", True),
        res_cache=dict(),
        compile_kwargs=dict(
            arch=kwargs.get("arch"),
            autotune=kwargs.get("autotune", True),
            cudagraph=kwargs.get("cudagraph", False),
        ),
        cudagraph=kwargs.get("cudagraph", False),
    )
    variable_name = list(variable.keys())[-1]
    variable_vals = variable[variable_name]
    benchmark_kwargs.pop(variable_name)

    possible_providers = ['torch', 'triton'] + [f"cute_v{version}" for version in kwargs.get('kernel_version')]

    @perf_report((
        Benchmark(
            x_names=[variable_name],
            x_vals=variable_vals,
            x_log=True if (variable_vals[-1] > 1 and len(variable_vals) > 1) else False,
            line_arg='provider',
            line_vals=possible_providers,
            line_names=possible_providers,
            styles=STYLES,
            ylabel='TFLOPS',
            plot_name='triton_bench',
            args=benchmark_kwargs,
        )
    ))
    def benchmark(**bench_kwargs):
        mA = bench_kwargs.get("A")[:, :bench_kwargs.get("M"), :].contiguous()
        mB = deepcopy(bench_kwargs.get("B"))
        mSparsity = bench_kwargs.get("sparsity")
        mMask = (bench_kwargs.get("Mask")[:, :bench_kwargs.get("M"), :].to(device) > mSparsity).contiguous()

        check_precision = bench_kwargs.get("check_precision")
        res_cache = bench_kwargs.get("res_cache")
        compile_kwargs = bench_kwargs.get("compile_kwargs")
        cudagraph = bench_kwargs.get("cudagraph")

        run_args = dict(
            M=bench_kwargs.get("M"),
            N=bench_kwargs.get("N"),
            K=bench_kwargs.get("K"),
            G=bench_kwargs.get("G"),
            sparsity=mSparsity,
        )

        provider = bench_kwargs.get("provider")
        kernel = None
        if provider == 'torch': kernel = torch_ref_program
        elif provider == 'triton': kernel = triton_ref_program
        elif provider.startswith('cute_v'):
            kernel = gemm_mn
            compile_kwargs["kernel_version"] = int(provider.split('cute_v')[-1])

        func = partial(
            kernel,
            A=mA,
            B=mB,
            Mask=None if provider == 'torch' else mMask,
            activation='identity',
            estimate_sparsity=mSparsity,
            **compile_kwargs,
        )
        quantiles = [0.5, 0.2, 0.8]
        with torch.no_grad():
            # warmup and record res
            res = func()
            if not isinstance(res, torch.Tensor):
                res = res[0]
            if check_precision: res_cache[provider] = res.detach().clone() if isinstance(res, torch.Tensor) else res

            try:
                if cudagraph:
                    ms, min_ms, max_ms = do_bench_cudagraph(func, rep=200, quantiles=quantiles)
                else:
                    ms, min_ms, max_ms = do_bench(func, rep=200, quantiles=quantiles)
                print(f"[INFO] ✅ finish {provider} in args: {run_args}.")
            except Exception as e:
                print(e)
                return 0, 0, 0

        # checking precision
        if check_precision and len(res_cache) == len(possible_providers):
            torch_res = res_cache.pop('torch')
            triton_res = res_cache.pop('triton')
            diff = (torch_res - triton_res).abs()
            print(f"[INFO] torch vs triton: max_diff: {diff.max().item()}, mean_diff: {diff.mean().item()}")

            for prov in possible_providers:
                if prov in res_cache:
                    cute_res = res_cache.pop(prov)
                    torch_diff = (torch_res - cute_res).abs()
                    triton_diff = (triton_res - cute_res).abs()
                    print(f"[INFO] {prov} vs torch: max_diff: {torch_diff.max().item()}, mean_diff: {torch_diff.mean().item()}")
                    print(f"[INFO] {prov} vs triton: max_diff: {triton_diff.max().item()}, mean_diff: {triton_diff.mean().item()}")

        tflops = lambda ms: (2 * (run_args["M"] * run_args["N"] * run_args["K"]) * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)

    return benchmark

def run_throughput(**kwargs):
    bench = _benchmark_interface(**kwargs)
    bench.run(show_plots=False, print_data=True)
