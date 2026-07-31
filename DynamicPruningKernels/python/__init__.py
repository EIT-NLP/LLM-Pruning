"""Public API for DynamicPruningKernels."""

from __future__ import annotations

from typing import Any

from .registry import (
    KERNELS,
    KernelHandle,
    KernelNotFoundError,
    KernelRegistry,
    KernelRegistryError,
    KernelSpec,
    KernelUnavailableError,
    KernelVariant,
    KernelVariantError,
    VersionRequest,
    detect_arch,
    register_kernel,
)

__version__ = "0.1.0"


def get_kernel(
    family: str,
    backend: str = "auto",
    *,
    version: VersionRequest = "default",
    arch: str | None = "auto",
) -> KernelHandle:
    """Resolve a configured kernel without importing its implementation eagerly."""

    return KERNELS.get(family, backend, version=version, arch=arch)


def run_kernel(
    family: str,
    *args: Any,
    backend: str = "auto",
    version: VersionRequest = "default",
    arch: str | None = "auto",
    **kwargs: Any,
) -> Any:
    """Resolve and execute one kernel family."""

    kernel = get_kernel(family, backend, version=version, arch=arch)
    return kernel(*args, **kwargs)


def list_kernels(
    family: str | None = None,
    *,
    available_only: bool = False,
) -> tuple[dict[str, Any], ...]:
    """Return a serializable catalog of registered implementations."""

    return tuple(
        spec.as_dict()
        for spec in KERNELS.list(family, available_only=available_only)
    )


def available_backends(family: str) -> tuple[str, ...]:
    """Return importable backends in auto-selection order."""

    return tuple(
        spec.backend
        for spec in KERNELS.list(family, available_only=True)
    )


__all__ = [
    "KERNELS",
    "KernelHandle",
    "KernelNotFoundError",
    "KernelRegistry",
    "KernelRegistryError",
    "KernelSpec",
    "KernelUnavailableError",
    "KernelVariant",
    "KernelVariantError",
    "available_backends",
    "detect_arch",
    "get_kernel",
    "list_kernels",
    "register_kernel",
    "run_kernel",
]
