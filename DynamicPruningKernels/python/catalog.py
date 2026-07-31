"""Single source of truth for shipped kernels and source-tree runners.

This module deliberately has no Torch/JIT imports.  It is imported as
``python.catalog`` by the original repository workflow and as
``dynamic_width_jit.catalog`` after packaging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


_ARCH_HEADER = re.compile(r"sm[0-9]+[a-z0-9]*[.]cuh")


@lru_cache(maxsize=1)
def header_root() -> Path:
    """Locate headers in either the checkout or an installed wheel."""

    package_dir = Path(__file__).resolve().parent
    for candidate in (package_dir.parent / "include", package_dir / "include"):
        if candidate.is_dir():
            return candidate
    raise RuntimeError(
        "dynamic-width JIT headers were not found beside the source tree or "
        "installed package"
    )


@lru_cache(maxsize=None)
def discover_cute_variants(family: str) -> dict[int, tuple[str, ...]]:
    """Discover ``vN/sm*.cuh`` implementations for one CuTe family."""

    family_dir = header_root() / family
    variants: dict[int, set[str]] = {}
    if not family_dir.is_dir():
        return {}

    for version_dir in family_dir.iterdir():
        if not version_dir.is_dir() or not version_dir.name.startswith("v"):
            continue
        try:
            version = int(version_dir.name[1:])
        except ValueError:
            continue
        for header in version_dir.iterdir():
            if header.is_file() and _ARCH_HEADER.fullmatch(header.name):
                variants.setdefault(version, set()).add(header.stem)

    return {
        version: tuple(sorted(architectures))
        for version, architectures in sorted(variants.items())
    }


def cute_versions(family: str) -> tuple[int, ...]:
    return tuple(discover_cute_variants(family))


def cute_architectures(family: str) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                architecture
                for architectures in discover_cute_variants(family).values()
                for architecture in architectures
            }
        )
    )


@dataclass(frozen=True)
class BackendDefinition:
    name: str
    target: str
    requires: tuple[str, ...]
    priority: int
    description: str
    default_version: int | None = None
    discover_variants: bool = False

    def variants(self, family: str) -> dict[int, tuple[str, ...]]:
        if not self.discover_variants:
            return {}
        return discover_cute_variants(family)


@dataclass(frozen=True)
class KernelDefinition:
    family: str
    runner_module: str
    argument_kind: str
    legacy_default_version: int
    backends: tuple[BackendDefinition, ...]


_CUTE_REQUIRES = ("torch", "triton", "sglang", "tvm_ffi")


def _cute(
    target: str,
    default_version: int,
    description: str,
) -> BackendDefinition:
    return BackendDefinition(
        name="cute",
        target=target,
        requires=_CUTE_REQUIRES,
        priority=100,
        description=description,
        default_version=default_version,
        discover_variants=True,
    )


def _triton(target: str) -> BackendDefinition:
    return BackendDefinition(
        name="triton",
        target=target,
        requires=("torch", "triton"),
        priority=80,
        description="Triton reference implementation",
    )


def _torch(target: str) -> BackendDefinition:
    return BackendDefinition(
        name="torch",
        target=target,
        requires=("torch",),
        priority=10,
        description="PyTorch correctness reference",
    )


KERNEL_CATALOG: dict[str, KernelDefinition] = {
    "gemm_mn": KernelDefinition(
        family="gemm",
        runner_module="gemm_mn.jit",
        argument_kind="gemm",
        legacy_default_version=1,
        backends=(
            _cute(
                "gemm_mn.jit:gemm_mn",
                1,
                "CuTe JIT output-width-pruned GEMM/GEMV",
            ),
            _triton("gemm_mn.triton:gemm_mn_host"),
            _torch("references:gemm_mn"),
        ),
    ),
    "gemm_k": KernelDefinition(
        family="gemm",
        runner_module="gemm_k.jit",
        argument_kind="gemm",
        legacy_default_version=1,
        backends=(
            _cute(
                "gemm_k.jit:gemm_k",
                1,
                "CuTe JIT input-width-pruned GEMM/GEMV",
            ),
            _triton("gemm_k.triton:gemm_k_host"),
            _torch("references:gemm_k"),
        ),
    ),
    "attention_prefill": KernelDefinition(
        family="attention",
        runner_module="attention_prefill.jit",
        argument_kind="attention",
        legacy_default_version=3,
        backends=(
            _cute(
                "attention_prefill.jit:attention_head",
                3,
                "CuTe JIT query/head-width-pruned prefill attention",
            ),
            _triton("attention_prefill.triton:attention_head_host"),
            _torch("references:attention_prefill"),
        ),
    ),
    "attention_decode": KernelDefinition(
        family="attention",
        runner_module="attention_decode.jit",
        argument_kind="attention",
        legacy_default_version=0,
        backends=(
            _cute(
                "attention_decode.jit:attention_head",
                0,
                "CuTe JIT query/head-width-pruned decode attention",
            ),
            BackendDefinition(
                name="tilelang",
                target="attention_decode.tilelang:decoding_head_host",
                requires=("torch", "tilelang"),
                priority=70,
                description="TileLang decode reference implementation",
            ),
            _torch("references:attention_decode"),
        ),
    ),
}
