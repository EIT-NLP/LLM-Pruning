import pathlib
from functools import lru_cache

import torch
from torch.utils.cpp_extension import include_paths, library_paths

@lru_cache()
def _resolve_kernel_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).parent.resolve()
    return cur_dir

@lru_cache()
def _resolve_3rdparty_path() -> pathlib.Path:
    cur_dir = pathlib.Path(__file__).resolve().parent.parent.parent.resolve()
    return cur_dir

# install from https://developer.nvidia.com/cusparselt-downloads
CUSPARSELT_INCLUDE_DIR = "/usr/include/libcusparseLt/13"
CUSPARSELT_LIB_DIR = "/usr/lib/x86_64-linux-gnu/libcusparseLt/13"

ROOT_PATH = _resolve_kernel_path()
THIRD_PARTY_HEADER_DIR = _resolve_3rdparty_path() / "3rdparty"
THIRD_PARTY_HEADER_DIRS = [
    str(THIRD_PARTY_HEADER_DIR / "cutlass/include"),
    str(THIRD_PARTY_HEADER_DIR / "dlpack/include"),
    str(THIRD_PARTY_HEADER_DIR / "tvm-ffi/include"),
    str(THIRD_PARTY_HEADER_DIR / "sglang/include"),
    CUSPARSELT_INCLUDE_DIR,
    *include_paths(),
]

DEFAULT_CFLAGS = []
DEFAULT_CUDA_CFLAGS = []
DEFAULT_LDFLAGS = [
    f"-L{CUSPARSELT_LIB_DIR}",
    *[f"-L{path}" for path in library_paths()],
    "-lcusparseLt",
    "-lcusparse",
    "-lc10_cuda",
    "-ldl",
]
