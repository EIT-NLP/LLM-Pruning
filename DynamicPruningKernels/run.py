from __future__ import annotations

import argparse
import importlib
import os
from dataclasses import asdict, dataclass, fields
from typing import Any

import torch

from python.catalog import KERNEL_CATALOG


DEFAULT_ARCH = (
    f"sm{torch.cuda.get_device_capability()[0]}x"
    if torch.cuda.is_available()
    else None
)


DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


@dataclass
class CommonArgs:
    kernel_family: str = "gemm"
    kernel: str = "gemm_mn"
    device: str = "cuda:0"
    dtype: str = "float16"
    sparsity: float = 0
    check_precision: bool = True
    kernel_version: list[int] | None = None
    arch: str | None = DEFAULT_ARCH
    seed: int = 17
    run_type: str = "run_once"
    autotune: bool = True
    cudagraph: bool = False
    tracing: bool = False


@dataclass
class GemmArgs:
    M: int = 8192
    N: int = 8192
    K: int = 8192
    G: int = 128


@dataclass
class AttentionArgs:
    B: int = 1
    Tq: int = 8192
    Tk: int = 8192
    Hq: int = 32
    Hk: int = 8
    D: int = 128
    NG: int = 8
    is_causal: bool = True


ARGUMENT_TYPES = {
    "gemm": GemmArgs,
    "attention": AttentionArgs,
}


# Keep the original ``python.*`` import path so run.py and the shell scripts do
# not require an editable/package install.  Installed APIs consume the same
# catalog through ``dynamic_width_jit.catalog``.
KERNEL_REGISTRY = {
    name: {
        "family": definition.family,
        "module": f"python.{definition.runner_module}",
        "arg_type": ARGUMENT_TYPES[definition.argument_kind],
    }
    for name, definition in KERNEL_CATALOG.items()
}


DEFAULT_KERNEL_VERSIONS = {
    name: definition.legacy_default_version
    for name, definition in KERNEL_CATALOG.items()
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value

    normalized = value.lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def resolve_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.lower()
    if key not in DTYPE_MAP:
        supported = ", ".join(sorted(DTYPE_MAP))
        raise ValueError(f"Unsupported dtype '{dtype_name}'. Supported: {supported}")
    return DTYPE_MAP[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run DynamicPruningKernels correctness and benchmark entrypoints.",
    )

    common_group = parser.add_argument_group("common")
    common_group.add_argument(
        "--kernel-family",
        choices=sorted({definition.family for definition in KERNEL_CATALOG.values()}),
        default=CommonArgs.kernel_family,
        help="High-level kernel category.",
    )
    common_group.add_argument(
        "--kernel",
        default=CommonArgs.kernel,
        choices=sorted(KERNEL_REGISTRY),
        help="Concrete kernel implementation to run.",
    )
    common_group.add_argument(
        "--device",
        default=CommonArgs.device,
        help="Torch device string passed into run_once().",
    )
    common_group.add_argument(
        "--dtype",
        default=CommonArgs.dtype,
        choices=sorted(DTYPE_MAP),
        help="Data type passed into run_once().",
    )
    common_group.add_argument(
        "--sparsity",
        type=float,
        nargs="+",
        default=[CommonArgs.sparsity],
        help="Mask sparsity used by pruning.",
    )
    common_group.add_argument(
        "--check-precision",
        type=str2bool,
        default=CommonArgs.check_precision,
        help="Whether to compare kernel output with the reference implementation.",
    )
    common_group.add_argument(
        "--kernel-version",
        type=int,
        nargs="+",
        default=None,
        help="Optional kernel implementation version override, e.g. v0 -> 0.",
    )
    common_group.add_argument(
        "--arch",
        default=CommonArgs.arch,
        help="Optional kernel arch override such as sm8x or sm12x.",
    )
    common_group.add_argument(
        "--seed",
        type=int,
        default=CommonArgs.seed,
        help="Random seed used by profiling",
    )
    common_group.add_argument(
        "--run-type",
        default="run_once",
        help="Function name to run.",
    )
    common_group.add_argument(
        "--autotune",
        type=str2bool,
        default=CommonArgs.autotune,
        help="Tune and cache the best JIT configuration.",
    )
    common_group.add_argument(
        "--cudagraph",
        type=str2bool,
        default=CommonArgs.cudagraph,
        help="Whether to use cudagraph for benchmarking.",
    )
    common_group.add_argument(
        "--tracing",
        type=str2bool,
        default=CommonArgs.tracing,
        help="Whether to use torch profiler to trace the kernel.",
    )

    gemm_group = parser.add_argument_group("gemm")
    gemm_group.add_argument("--M", "--m", dest="M", type=int, nargs="+", default=[GemmArgs.M], help="GEMM M dimension.")
    gemm_group.add_argument("--N", "--n", dest="N", type=int, default=GemmArgs.N, help="GEMM N dimension.")
    gemm_group.add_argument("--K", "--k", dest="K", type=int, default=GemmArgs.K, help="GEMM K dimension.")
    gemm_group.add_argument("--G", "--g", dest="G", type=int, default=GemmArgs.G, help="GEMM group width.")

    attention_group = parser.add_argument_group("attention")
    attention_group.add_argument("--B", type=int, nargs="+", default=[AttentionArgs.B], help="Attention Batch Size.")
    attention_group.add_argument("--Tq", type=int, nargs="+", default=[AttentionArgs.Tq], help="Attention Query Seqlen.")
    attention_group.add_argument("--Tk", type=int, nargs="+", default=[], help="Attention Key/Value Seqlen.")
    attention_group.add_argument("--Hq", type=int, default=AttentionArgs.Hq, help="Attention Query Head Num.")
    attention_group.add_argument("--Hk", type=int, default=AttentionArgs.Hk, help="Attention Key/Value Head Num.")
    attention_group.add_argument("--D", type=int, default=AttentionArgs.D, help="Attention Head Dim.")
    attention_group.add_argument("--NG", type=int, default=AttentionArgs.NG, help="Attention Pruning Num Groups.")
    attention_group.add_argument("--is_causal", type=str2bool, default=AttentionArgs.is_causal, help="Causal Attention or not.")

    return parser


def build_common_args(args: argparse.Namespace) -> CommonArgs:
    kernel_versions = args.kernel_version
    if kernel_versions is None:
        kernel_versions = [DEFAULT_KERNEL_VERSIONS[args.kernel]]

    return CommonArgs(
        kernel_family=args.kernel_family,
        kernel=args.kernel,
        device=args.device,
        dtype=args.dtype,
        sparsity=args.sparsity,
        check_precision=args.check_precision,
        kernel_version=kernel_versions,
        arch=args.arch,
        seed=args.seed,
        run_type=args.run_type,
        autotune=args.autotune,
        cudagraph=args.cudagraph,
        tracing=args.tracing,
    )


def build_kernel_args(args: argparse.Namespace, kernel_name: str) -> Any:
    arg_type = KERNEL_REGISTRY[kernel_name]["arg_type"]
    return arg_type(**{field.name: getattr(args, field.name) for field in fields(arg_type)})


def validate_selection(common_args: CommonArgs) -> None:
    if common_args.kernel not in KERNEL_REGISTRY:
        available = ", ".join(sorted(KERNEL_REGISTRY))
        raise ValueError(f"Unknown kernel '{common_args.kernel}'. Available kernels: {available}")

    expected_family = KERNEL_REGISTRY[common_args.kernel]["family"]
    if common_args.kernel_family != expected_family:
        raise ValueError(
            f"Kernel '{common_args.kernel}' belongs to family '{expected_family}', "
            f"but received '--kernel-family {common_args.kernel_family}'."
        )


def build_run_config(common_args: CommonArgs, kernel_args: Any) -> dict[str, Any]:
    config = {**asdict(common_args), **asdict(kernel_args)}
    config["dtype"] = resolve_dtype(config["dtype"])
    config.pop("kernel_family", None)
    config.pop("kernel", None)
    return {key: value for key, value in config.items() if value is not None}


def load_kernel_runner(kernel_name: str, func_name: str="run_once"):
    module_name = KERNEL_REGISTRY[kernel_name]["module"]
    module = importlib.import_module(module_name)

    if not hasattr(module, func_name):
        raise AttributeError(f"Module '{module_name}' does not define '{func_name}'()")
    return getattr(module, func_name)

def set_seed(seed: int):
    import os
    import random
    import numpy as np

    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    common_args = build_common_args(args)
    validate_selection(common_args)
    kernel_args = build_kernel_args(args, common_args.kernel)
    run_config = build_run_config(common_args, kernel_args)

    print(f"[INFO] kernel_family={common_args.kernel_family} kernel={common_args.kernel} version={common_args.kernel_version} arch={common_args.arch}")
    print(f"[INFO] run_config={run_config}")

    runner = load_kernel_runner(common_args.kernel, common_args.run_type)

    if os.environ.get("NCU_PROFILE_AFTER_WARMUP", "").lower() in {"1", "true", "yes", "on"}:
        print("[INFO] NCU warmup/autotune pass before profiler start")
        runner(**run_config)
        torch.cuda.synchronize()

        cudart = torch.cuda.cudart()
        print("[INFO] cudaProfilerStart")
        cudart.cudaProfilerStart()
        runner(**run_config)
        torch.cuda.synchronize()
        print("[INFO] cudaProfilerStop")
        cudart.cudaProfilerStop()
    else:
        runner(**run_config)

    if args.tracing:
        from torch.profiler import profile
        profiler_name = f"{common_args.kernel}_{common_args.arch}"
        profiler = profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./torch_profile', worker_name=profiler_name, use_gzip=True),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            acc_events=True,
        )

        profiler.start()
        for i in range(5):
            runner(**run_config)
            profiler.step()
        profiler.stop()

if __name__ == "__main__":
    main()
