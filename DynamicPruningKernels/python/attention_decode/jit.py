from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Any, List, Mapping
from einops import rearrange
from copy import deepcopy
from functools import lru_cache

import torch
from torch.nn.functional import scaled_dot_product_attention
from torch.nn.attention import SDPBackend, sdpa_kernel

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
from .tilelang import decoding_head_host as tilelang_ref_program

__SUPPORT_VERSION__ = list(cute_versions("attention_decode"))
__SUPPORT_ARCH__ = list(cute_architectures("attention_decode"))
MAXSPLIT = 16

############################### Attention JIT KERNEL WRAPPER ###############################

### Default Search Space
@lru_cache
def attention_head_space():
    BN = [32, 64, 128]
    PipelineK = [1, 2]
    PipelineV = [1, 2]
    Warps = [1, 2, 4]
    Split = [_ for _ in range(1, MAXSPLIT + 1)]

    return [
        dict(
            BN=bn,
            PipelineK=pipelinek,
            PipelineV=pipelinev,
            Warps=warp,
            Split=split,
        )
        for bn in BN
        for pipelinek in PipelineK
        for pipelinev in PipelineV
        for warp in Warps
        for split in Split
    ]

### Search Space Pruning
def prune_attention_head_space(configs: List[JitConfig], named_args: Mapping[str, Any]) -> list[JitConfig]:
    cc = int(named_args["cc"])
    smem_size = int(named_args["smem_size"])
    num_sm = int(named_args["num_sm"])
    data_width = int(named_args["data_width"])

    rTk = int(named_args["rTk"])
    D = int(named_args["D"])
    filtered: list[JitConfig] = []

    def get_mn_candidate(x: int, max_mn: int) -> set[int]:
        candidate = min(max_mn, max(32, next_pow_of_2(x)))
        return {candidate, max(32, candidate // 2), max(32, candidate // 4)}

    max_mn = 128
    bn_candidate = get_mn_candidate(rTk, max_mn)

    for config in configs:
        kwargs = config.kwargs
        BN = int(kwargs["BN"])
        PipelineK = int(kwargs["PipelineK"])
        PipelineV = int(kwargs["PipelineV"])
        Split = int(kwargs["Split"])
        Warps = int(kwargs["Warps"])

        if BN not in bn_candidate or BN < Warps * 16: continue
        if data_width * (PipelineK * (BN * D) + PipelineV * (BN * D) + 16 * BN) + 8 > smem_size: continue
        if Split > 1 and BN >= (rTk // Split): continue
        if Warps != min(4, max(1, D // 32)): continue
        filtered.append(config)

    return filtered

### Heuristic Fallback
def heuristic_attention_head_config(named_args: Mapping[str, Any]) -> dict[str, Any]:
    cc = int(named_args["cc"])
    smem_size = int(named_args["smem_size"])
    num_sm = int(named_args["num_sm"])
    data_width = int(named_args["data_width"])

    rTk = named_args["rTk"]
    D = int(named_args["D"])

    cc_major = cc // 10
    if cc_major not in (8, 12):
        if cc_major == 9:
            raise NotImplementedError("sm90-like wgmma is not supported")
        if cc_major == 10:
            raise NotImplementedError("sm100-like umma is not supported")
        raise ValueError(f"Unsupported compute capability {cc}")

    BN = min(64, max(32, rTk))
    PipelineK = 2
    PipelineV = 1
    Warps = 4

    assert BN >= 16, "group size must be at least 16 for sm80-like warp mma"

    Split = max(1, min(MAXSPLIT, rTk // 512))

    return {
        "BN": BN,
        "PipelineK": PipelineK,
        "PipelineV": PipelineV,
        "Warps": Warps,
        "Split": Split,
    }

### Compile Module
@cache_once
def _compile_attention_head_module(
    Hq: int, Hk: int, D: int, NG: int, 
    BM: int, BN: int, PipelineK: int, PipelineV: int, SplitRound: int, Warps: int,
    IsLeftpad: bool, 
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
) -> Module:
    # check kernel version and arch
    if (kernel_version == -1 or (kernel_version < len(__SUPPORT_VERSION__) and kernel_version >= 0)):
        kernel_version = __SUPPORT_VERSION__[kernel_version]
    else:
        raise ValueError(f"Unsupported kernel version '{kernel_version}'. Available versions: {__SUPPORT_VERSION__}")

    if arch not in __SUPPORT_ARCH__:
        raise ValueError(f"Unsupported arch '{arch}'. Available archs: {__SUPPORT_ARCH__}")

    cpp_args = [Hq, Hk, D, NG, BM, BN, PipelineK, PipelineV, SplitRound, Warps, IsLeftpad] + [is_arch_support_pdl(), dtype]
    cpp_args = make_cpp_args(*cpp_args)

    return load_jit_for_arch(
        arch,
        "attention_head",
        *cpp_args,
        cuda_files=[str(ROOT_PATH / "include" / f"attention_decode/v{kernel_version}/{arch}.cuh")],
        cuda_wrappers=[("kernel", f"Attention_Head_V{kernel_version}_Host<{cpp_args}>::run")],
        extra_cflags=DEFAULT_CFLAGS,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS,
        extra_include_paths=THIRD_PARTY_HEADER_DIRS,
    )

### Autotune Wrapper
@autotune(
    kernel_id="attention_head",
    config_params=['BN', 'PipelineK', 'PipelineV', 'Split', 'Warps'],
    runtime_config_params=['Split'],
    configs=attention_head_space,
    key=['B', 'BM', 'rTk', 'Hq', 'Hk', 'D', 'NG', 'IsLeftpad', 'SplitRound', 'dtype', 'kernel_version', 'arch', 'estimate_sparsity', 'cc'],
    prune_configs_by=prune_attention_head_space,
    runtime_params=["Q", "K", "V", "Mask", "Leftpad", "O", "pO", "LSE", "scale", "estimate_sparsity"],
    restore_params=["O"],
    heuristic=heuristic_attention_head_config,
    cudagraph=True,
)
def _jit_attention_head_module(
    Hq: int, Hk: int, D: int, NG: int, 
    BM: int, BN: int, PipelineK: int, PipelineV: int, SplitRound: int, Warps: int,
    IsLeftpad: bool, 
    kernel_version: int,
    arch: str,
    dtype: Optional[torch.dtype]=torch.float16,
) -> Module:
    return _compile_attention_head_module(
        Hq, Hk, D, NG, BM, BN, PipelineK, PipelineV, SplitRound, Warps,
        IsLeftpad,
        kernel_version, arch,
        dtype=dtype,
    )

def attention_head(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    Leftpad: Optional[torch.Tensor]=None,
    O: Optional[torch.Tensor]=None,
    sorted_mask: Optional[torch.Tensor]=None,
    sorted_indices: Optional[torch.Tensor]=None,
    scale: Optional[float]=None,
    is_causal: Optional[bool]=True,
    estimate_sparsity: Optional[float]=0.5,
    kernel_version: Optional[int]=0,
    arch: Optional[str]="sm12x",
    autotune: Optional[bool]=True,
    cudagraph: bool=False,
):
    assert Q.dim() == 4 and K.dim() ==4 and V.dim() == 4
    assert Q.shape[1] == 1
    assert K.shape == V.shape
    assert Mask.dim() == 3
    assert estimate_sparsity <= 1
    estimate_sparsity = 1 if estimate_sparsity == 0 else estimate_sparsity

    B, Tq, Hq, D = Q.shape
    _, Tk, Hk, _ = K.shape
    _, _, NG = Mask.shape

    if O is None: O = torch.zeros((B, Tq, Hq, D), dtype=Q.dtype, device=Q.device)
    pO = torch.empty((B, Hq, MAXSPLIT, D), dtype=Q.dtype, device=Q.device)
    LSE = torch.empty((B, Hq, MAXSPLIT), dtype=torch.float32, device=Q.device)

    is_leftpad = True
    if Leftpad is None:
        Leftpad = torch.empty((B,), dtype=torch.uint32, device=Q.device)
        is_leftpad = False

    if scale is None: scale = D ** -0.5
    
    module = _jit_attention_head_module(
        B=B, rTk=next_pow_of_2(Tk), Hq=Hq, Hk=Hk, D=D, NG=NG, BM=Hq // max(NG, Hk),
        SplitRound=MAXSPLIT,
        IsLeftpad=is_leftpad,
        kernel_version=kernel_version,
        arch=arch,
        dtype=Q.dtype,
        Q=Q,
        K=K,
        V=V,
        Mask=Mask,
        Leftpad=Leftpad,
        O=O,
        pO=pO,
        LSE=LSE,
        scale=scale,
        estimate_sparsity=estimate_sparsity,
        autotune=autotune,
        cudagraph=cudagraph,
    )
    module.run(Q, K, V, Mask, Leftpad, O, pO, LSE, scale, estimate_sparsity)
    return O, dict()

def torch_ref_program(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    O: Optional[torch.Tensor]=None,
    scale: Optional[float]=None,
    estimate_sparsity: Optional[float]=0.5,
    **kwargs,
):
    Q = rearrange(Q, "B T H D -> B H T D")
    K = rearrange(K, "B T H D -> B H T D")
    V = rearrange(V, "B T H D -> B H T D")
    with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.FLASH_ATTENTION]):
        O = scaled_dot_product_attention(
            Q, K, V,
            scale=scale,
            enable_gqa=True
        )
    O = rearrange(O, "B H T D -> B T H D")
    
    if Mask is not None:
        NG = Mask.shape[-1]
        O = rearrange(O, "B T (NG G) D -> B T NG G D", NG=NG)
        O.masked_fill_(Mask.logical_not()[..., None, None], 0)
        O = rearrange(O, "B T NG G D -> B T (NG G) D")
    
    return O

def run_once(**kwargs):
    import math
    from torch.nn import init as torch_init

    device = kwargs.get("device", "cuda:0")
    if torch.device(device).type == "cuda": torch.cuda.set_device(torch.device(device))
    dtype = kwargs.get("dtype", torch.float16)

    B = kwargs.get("B", [1])[-1]
    Tq = 1
    Tk = kwargs.get("Tk", [8192])[-1]
    Hq = kwargs.get("Hq", 32)
    Hk = kwargs.get("Hk", 8)
    D = kwargs.get("D", 128)
    NG = kwargs.get("NG", 8)

    sparsity = kwargs.get("sparsity", [0.5])[-1]
    check_precision = kwargs.get("check_precision", True)

    Q = torch.randn((B, Tq, Hq, D), dtype=dtype, device=device)
    K = torch.randn((B, Tk, Hk, D), dtype=dtype, device=device)
    V = torch.randn((B, Tk, Hk, D), dtype=dtype, device=device)
    Mask = torch.rand((B, Tq, NG), device=device) >= sparsity

    compile_kwargs = dict(
        arch=kwargs.get("arch"),
        autotune=kwargs.get("autotune", True),
        cudagraph=kwargs.get("cudagraph", False),
    )
    # Leftpad = torch.full([B], 3, dtype=torch.uint32, device=device)
    Leftpad = None

    D_tilelang = tilelang_ref_program(
        deepcopy(Q), deepcopy(K), deepcopy(V), deepcopy(Mask), Leftpad, None, None,
        estimate_sparsity=sparsity,
    )
    D_torch = torch_ref_program(
        deepcopy(Q), deepcopy(K), deepcopy(V), deepcopy(Mask), None, None,
        estimate_sparsity=sparsity,
    )

    for version in kwargs.get('kernel_version'):
        kernel_version = __SUPPORT_VERSION__[version]
        compile_kwargs["kernel_version"] = kernel_version

        D_cute, _ = attention_head(
            deepcopy(Q), deepcopy(K), deepcopy(V), deepcopy(Mask), Leftpad,
            estimate_sparsity=sparsity,
            **compile_kwargs,
        )

        if check_precision:
            diff = (D_tilelang - D_torch).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"[INFO] tilelang vs torch: max_diff: {max_diff.item()}, mean_diff: {mean_diff.item()}")

            diff = (D_cute - D_torch).abs()
            max_diff = diff.max()
            mean_diff = diff.mean()
            print(f"[INFO] cute_v{kernel_version} vs torch: max_diff: {max_diff.item()}, mean_diff: {mean_diff.item()}")

            diff = (D_tilelang - D_cute).abs()
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

    B = kwargs.get("B", [1])
    Tq = 1
    Tk = kwargs.get("Tk", [8192])
    Hq = kwargs.get("Hq", 32)
    Hk = kwargs.get("Hk", 8)
    D = kwargs.get("D", 128)
    NG = kwargs.get("NG", 8)

    sparsity = kwargs.get("sparsity", [0.5])

    assert not (len(B) > 1 and len(Tk) > 1 and len(sparsity) > 1), "Each profiling step only support one variable"

    variable = dict(B=B)
    if len(Tk) > 1: variable = dict(Tk=Tk)
    elif len(sparsity) > 1: variable = dict(sparsity=sparsity)

    Q = torch.randn((B[-1], Tq, Hq, D), dtype=dtype, device=device)
    K = torch.randn((B[-1], Tk[-1], Hk, D), dtype=dtype, device=device)
    V = torch.randn((B[-1], Tk[-1], Hk, D), dtype=dtype, device=device)
    Mask = torch.rand((B[-1], Tq, NG), device='cpu')

    benchmark_kwargs = dict(
        B=B[-1], Tq=Tq, Tk=Tk[-1], Hq=Hq, Hk=Hk, D=D, NG=NG,
        Q=Q, K=K, V=V, Mask=Mask, sparsity=sparsity[-1],
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

    possible_providers = ['torch', 'tilelang'] + [f"cute_v{version}" for version in kwargs.get('kernel_version')]

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
        current_tq = bench_kwargs.get("Tq")
        current_tk = bench_kwargs.get("Tk")
        mQ = bench_kwargs.get("Q")[:bench_kwargs.get("B"), :current_tq, :, :].contiguous()
        mK = bench_kwargs.get("K")[:bench_kwargs.get("B"), :current_tk, :, :].contiguous()
        mV = bench_kwargs.get("V")[:bench_kwargs.get("B"), :current_tk, :, :].contiguous()
        mSparsity = bench_kwargs.get("sparsity")
        mMask = (bench_kwargs.get("Mask")[:bench_kwargs.get("B"), :current_tq, :].to(device) > mSparsity).contiguous()

        check_precision = bench_kwargs.get("check_precision")
        res_cache = bench_kwargs.get("res_cache")
        compile_kwargs = bench_kwargs.get("compile_kwargs")
        cudagraph = bench_kwargs.get("cudagraph")

        run_args = dict(
            B=bench_kwargs.get("B"),
            Tq=current_tq,
            Tk=current_tk,
            Hq=bench_kwargs.get("Hq"),
            Hk=bench_kwargs.get("Hk"),
            D=bench_kwargs.get("D"),
            NG=bench_kwargs.get("NG"),
            sparsity=mSparsity,
            is_causal=bench_kwargs.get("is_causal"),
        )

        provider = bench_kwargs.get("provider")
        kernel = None
        if provider == 'torch': kernel = torch_ref_program
        elif provider == 'tilelang': kernel = tilelang_ref_program
        elif provider.startswith('cute_v'):
            kernel = attention_head
            compile_kwargs["kernel_version"] = int(provider.split('cute_v')[-1])

        func = partial(
            kernel,
            Q=mQ,
            K=mK,
            V=mV,
            Mask=None if provider == 'torch' else mMask,
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
            tilelang_res = res_cache.pop('tilelang')
            diff = (torch_res - tilelang_res).abs()
            print(f"[INFO] torch vs tilelang: max_diff: {diff.max().item()}, mean_diff: {diff.mean().item()}")

            for prov in possible_providers:
                if prov in res_cache:
                    cute_res = res_cache.pop(prov)
                    torch_diff = (torch_res - cute_res).abs()
                    tilelang_diff = (tilelang_res - cute_res).abs()
                    print(f"[INFO] {prov} vs torch: max_diff: {torch_diff.max().item()}, mean_diff: {torch_diff.mean().item()}")
                    print(f"[INFO] {prov} vs tilelang: max_diff: {tilelang_diff.max().item()}, mean_diff: {tilelang_diff.mean().item()}")

        tflops = lambda ms: (2 * 2 * (run_args["B"] * run_args["Tk"] * run_args["Hq"] * run_args["D"]) * 1e-12) / (ms * 1e-3)
        return tflops(ms), tflops(min_ms), tflops(max_ms)

    return benchmark

def run_throughput(**kwargs):
    bench = _benchmark_interface(**kwargs)
    bench.run(show_plots=False, print_data=True)
