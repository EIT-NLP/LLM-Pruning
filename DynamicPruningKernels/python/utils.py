import pathlib
import os
import shlex
import torch

from functools import lru_cache
from typing import Optional

########################################################
#                     Static Func
########################################################
@lru_cache()
def _resolve_kernel_path() -> pathlib.Path:
    package_dir = pathlib.Path(__file__).parent.resolve()
    source_root = package_dir.parent

    # Source/editable installs keep the original repository layout. Wheels copy
    # the JIT headers next to the installed Python package.
    if (source_root / "include").is_dir() and (source_root / "3rdparty").is_dir():
        return source_root
    if (package_dir / "include").is_dir() and (package_dir / "3rdparty").is_dir():
        return package_dir
    raise RuntimeError(
        "dynamic-width JIT headers were not found; reinstall the package with "
        "package data enabled"
    )

@lru_cache()
def _get_device_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


def _configure_tvm_ffi_cuda_arch_list() -> None:
    if "TVM_FFI_CUDA_ARCH_LIST" in os.environ or not torch.cuda.is_available():
        return
    major, minor = _get_device_capability()
    suffix = "a" if major == 12 else ""
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = f"{major}.{minor}{suffix}"


def _configure_sglang_jit_cuda_arch(arch: str) -> None:
    if arch != "sm12x":
        return

    # SGLang's temporary override restores a shared global and races when the
    # autotuner compiles several candidates in parallel.  Keep the actual
    # Blackwell target process-wide so every worker emits sm_120a.
    from sglang.jit_kernel import utils as jit_utils

    current = jit_utils.get_jit_cuda_arch()
    if (current.major, current.minor, current.suffix) != (12, 0, "a"):
        jit_utils._CUDA_ARCH = jit_utils.ArchInfo(12, 0, "a")


def load_jit_for_arch(arch: str, *args, **kwargs):
    """Call SGLang's JIT loader with the architecture-specific ISA target."""
    from sglang.jit_kernel import utils as jit_utils

    _configure_sglang_jit_cuda_arch(arch)
    return jit_utils.load_jit(*args, **kwargs)


@lru_cache
def _get_sm_count() -> int:
    return torch.cuda.get_device_properties("cuda").multi_processor_count

@lru_cache
def _get_smem_size(cc: Optional[int] = None) -> int:
    if cc is None:
        major_cc, minor_cc = _get_device_capability()
    else:
        major_cc, minor_cc = divmod(int(cc), 10)

    if major_cc == 8:
        return 167963 if minor_cc == 0 else 101376
    elif major_cc in [9, 10]:
        return 233472
    else:
        return 101376

get_smem_size = _get_smem_size

ROOT_PATH = _resolve_kernel_path()
THIRD_PARTY_HEADER_DIR = _resolve_kernel_path() / "3rdparty"
THIRD_PARTY_HEADER_DIRS = [
    str(THIRD_PARTY_HEADER_DIR / "cutlass/include"),
    str(THIRD_PARTY_HEADER_DIR / "dlpack/include"),
    str(THIRD_PARTY_HEADER_DIR / "tvm-ffi/include"),
    str(THIRD_PARTY_HEADER_DIR / "sglang/include"),
    str(ROOT_PATH / "include"),
]

DEFAULT_CFLAGS = shlex.split(os.environ.get("CUTE_JIT_CFLAGS", ""))
DEFAULT_CUDA_CFLAGS = shlex.split(os.environ.get("CUTE_JIT_CUDA_CFLAGS", ""))

_configure_tvm_ffi_cuda_arch_list()

ACTIVATION = {
    "identity": 0,
    "relu": 1,
    "silu": 2,
}

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

########################################################
#                     Helper Func
########################################################
def next_pow_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()

def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y
