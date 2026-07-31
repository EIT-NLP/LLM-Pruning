import os
import shutil
from pathlib import Path

from .autotuner import JitConfig, JitModule, autotune, wrap_jit_module

os.environ.setdefault("JIT_AUTOTUNE_CACHE_DIR", str(Path.home() / ".cache" / "jit_kernel" / "autotune"))
os.environ.setdefault("JIT_AUTOTUNE_COMPILE_WORKERS", "8")
os.environ.setdefault("JIT_AUTOTUNE_VERBOSE", "1")
os.environ.setdefault("TVM_FFI_CACHE_DIR", str(Path.home() / ".cache" / "tvm-ffi"))

if os.environ.get("JIT_AUTOTUNE_FORCE_TUNE", "").lower() in {"1", "true", "yes", "on"}:
    cache_dir = Path(os.environ["JIT_AUTOTUNE_CACHE_DIR"]).expanduser()
    print(f"[autotune] force tune: remove cache {cache_dir}", flush=True)
    shutil.rmtree(cache_dir, ignore_errors=True)

if os.environ.get("JIT_AUTOTUNE_FORCE_RECOMPILE", "").lower() in {"1", "true", "yes", "on"}:
    cache_dir = Path(os.environ["TVM_FFI_CACHE_DIR"]).expanduser()
    print(f"[autotune] force recompile: remove cache {cache_dir}", flush=True)
    shutil.rmtree(cache_dir, ignore_errors=True)

__all__ = [
    "JitConfig",
    "JitModule",
    "autotune",
    "wrap_jit_module",
]
