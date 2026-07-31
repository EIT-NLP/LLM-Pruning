"""Lazy, extensible kernel registry for dynamic-width operators."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable, Iterable, Mapping, Sequence

from .catalog import KERNEL_CATALOG


KernelCallable = Callable[..., Any]
KernelTarget = str | KernelCallable
VersionRequest = int | str | None
_PACKAGE_ROOT = __package__ or "dynamic_width_jit"


class KernelRegistryError(RuntimeError):
    """Base error raised by the public kernel registry."""


class KernelNotFoundError(KernelRegistryError):
    """Raised when a family/backend pair is not registered."""


class KernelUnavailableError(KernelRegistryError):
    """Raised when a registered backend cannot be imported."""


class KernelVariantError(KernelRegistryError, ValueError):
    """Raised when a version/architecture combination is not shipped."""


@dataclass(frozen=True)
class KernelVariant:
    """One shipped CuTe implementation and its compatible architectures."""

    version: int
    architectures: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("Kernel versions must be non-negative")
        if not self.architectures:
            raise ValueError("A versioned kernel needs at least one architecture")


def _canonical_name(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def _canonical_arch(value: str) -> str:
    arch = value.strip().lower().replace("_", "")
    aliases = {
        "80": "sm8x",
        "8.0": "sm8x",
        "86": "sm8x",
        "8.6": "sm8x",
        "89": "sm8x",
        "8.9": "sm8x",
        "sm80": "sm8x",
        "sm86": "sm8x",
        "sm89": "sm8x",
        "120": "sm12x",
        "12.0": "sm12x",
        "sm120": "sm12x",
    }
    return aliases.get(arch, arch)


def detect_arch(device: Any = None) -> str:
    """Return the repository architecture tag for the active CUDA device."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a base dependency
        raise KernelVariantError("arch='auto' requires PyTorch") from exc

    if not torch.cuda.is_available():
        raise KernelVariantError(
            "arch='auto' requires a CUDA device; pass an explicit architecture "
            "when only inspecting the registry"
        )
    major, minor = torch.cuda.get_device_capability(device)
    if major == 8:
        return "sm8x"
    if major == 12:
        return "sm12x"
    return f"sm{major}{minor}"


@dataclass(frozen=True)
class KernelSpec:
    """Metadata and lazy import target for one family/backend implementation."""

    family: str
    backend: str
    target: KernelTarget
    variants: tuple[KernelVariant, ...] = ()
    default_version: int | None = None
    requires: tuple[str, ...] = ()
    priority: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _canonical_name(self.family))
        object.__setattr__(self, "backend", _canonical_name(self.backend))
        versions = [variant.version for variant in self.variants]
        if len(versions) != len(set(versions)):
            raise ValueError(
                f"Duplicate versions in {self.family}.{self.backend}: {versions}"
            )
        if self.default_version is not None and self.default_version not in versions:
            raise ValueError(
                f"Default v{self.default_version} is not registered for "
                f"{self.family}.{self.backend}"
            )

    @property
    def key(self) -> tuple[str, str]:
        return self.family, self.backend

    @property
    def versions(self) -> tuple[int, ...]:
        return tuple(sorted(variant.version for variant in self.variants))

    @property
    def architectures(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    architecture
                    for variant in self.variants
                    for architecture in variant.architectures
                }
            )
        )

    def missing_requirements(self) -> tuple[str, ...]:
        missing: list[str] = []
        for module_name in self.requires:
            try:
                found = importlib.util.find_spec(module_name)
            except (ImportError, AttributeError, ValueError):
                found = None
            if found is None:
                missing.append(module_name)
        return tuple(missing)

    def is_available(self) -> bool:
        return not self.missing_requirements()

    def load(self) -> KernelCallable:
        missing = self.missing_requirements()
        if missing:
            joined = ", ".join(missing)
            raise KernelUnavailableError(
                f"Backend {self.family}.{self.backend} requires: {joined}"
            )
        if callable(self.target):
            return self.target

        module_name, separator, attribute = self.target.partition(":")
        if not separator or not module_name or not attribute:
            raise KernelRegistryError(
                f"Invalid target {self.target!r}; expected 'module:callable'"
            )
        try:
            module = importlib.import_module(module_name)
            target = getattr(module, attribute)
        except (ImportError, AttributeError) as exc:
            raise KernelUnavailableError(
                f"Could not load {self.family}.{self.backend} from {self.target}"
            ) from exc
        if not callable(target):
            raise KernelRegistryError(f"Kernel target {self.target} is not callable")
        return target

    def resolve_variant(
        self,
        version: VersionRequest = "default",
        arch: str | None = "auto",
    ) -> tuple[int | None, str | None]:
        """Resolve the newest compatible shipped version for an architecture."""

        if not self.variants:
            return None, None

        requested_arch = (
            detect_arch()
            if arch is None or str(arch).lower() in {"auto", "native"}
            else _canonical_arch(str(arch))
        )
        if requested_arch not in self.architectures:
            supported = ", ".join(self.architectures)
            raise KernelVariantError(
                f"{self.family}.{self.backend} does not support {requested_arch}; "
                f"available architectures: {supported}"
            )

        compatible = [
            variant.version
            for variant in self.variants
            if requested_arch in variant.architectures
        ]
        normalized = "default" if version is None else str(version).lower()
        if normalized in {"auto", "default"}:
            requested_version = self.default_version
            if requested_version is None:
                return max(compatible), requested_arch
        elif version == -1 or normalized == "latest":
            return max(compatible), requested_arch
        else:
            if normalized.startswith("v"):
                normalized = normalized[1:]
            try:
                requested_version = int(normalized)
            except ValueError as exc:
                raise KernelVariantError(f"Invalid kernel version: {version!r}") from exc

        variants = {variant.version: variant for variant in self.variants}
        if requested_version not in variants:
            supported = ", ".join(f"v{item}" for item in self.versions)
            raise KernelVariantError(
                f"{self.family}.{self.backend} has no v{requested_version}; "
                f"available versions: {supported}"
            )
        if requested_arch not in variants[requested_version].architectures:
            supported = ", ".join(variants[requested_version].architectures)
            raise KernelVariantError(
                f"{self.family}.{self.backend} v{requested_version} does not "
                f"support {requested_arch}; available architectures: {supported}"
            )
        return requested_version, requested_arch

    def as_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "backend": self.backend,
            "default_version": self.default_version,
            "versions": list(self.versions),
            "architectures": list(self.architectures),
            "variants": {
                f"v{variant.version}": list(variant.architectures)
                for variant in sorted(self.variants, key=lambda item: item.version)
            },
            "requires": list(self.requires),
            "available": self.is_available(),
            "priority": self.priority,
            "description": self.description,
        }


@dataclass(frozen=True)
class KernelHandle:
    """Callable returned by :func:`get_kernel` with a fixed configuration."""

    spec: KernelSpec
    version: int | None = None
    arch: str | None = None

    @property
    def family(self) -> str:
        return self.spec.family

    @property
    def backend(self) -> str:
        return self.spec.backend

    @cached_property
    def implementation(self) -> KernelCallable:
        """Load and validate the selected backend only once per handle."""

        return self.spec.load()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.spec.variants:
            requested_version = kwargs.pop("kernel_version", self.version)
            requested_arch = kwargs.pop("arch", self.arch)
            version, arch = self.spec.resolve_variant(
                version=requested_version,
                arch=requested_arch,
            )
            kwargs["kernel_version"] = version
            kwargs["arch"] = arch
        return self.implementation(*args, **kwargs)


class KernelRegistry:
    """Mutable registry with lazy backend loading and deterministic auto selection."""

    backend_aliases = {
        "jit": "cute",
        "cute_jit": "cute",
        "cutlass": "cute",
        "pytorch": "torch",
        "ref": "torch",
    }

    def __init__(self) -> None:
        self._specs: dict[tuple[str, str], KernelSpec] = {}

    def register(self, spec: KernelSpec, *, replace: bool = False) -> KernelSpec:
        if spec.key in self._specs and not replace:
            raise KernelRegistryError(
                f"Kernel {spec.family}.{spec.backend} is already registered"
            )
        self._specs[spec.key] = spec
        return spec

    def list(
        self,
        family: str | None = None,
        *,
        available_only: bool = False,
    ) -> tuple[KernelSpec, ...]:
        canonical_family = _canonical_name(family) if family else None
        specs = [
            spec
            for spec in self._specs.values()
            if canonical_family is None or spec.family == canonical_family
        ]
        if available_only:
            specs = [spec for spec in specs if spec.is_available()]
        return tuple(
            sorted(specs, key=lambda spec: (spec.family, -spec.priority, spec.backend))
        )

    def get(
        self,
        family: str,
        backend: str = "auto",
        *,
        version: VersionRequest = "default",
        arch: str | None = "auto",
    ) -> KernelHandle:
        canonical_family = _canonical_name(family)
        canonical_backend = _canonical_name(backend)
        canonical_backend = self.backend_aliases.get(
            canonical_backend, canonical_backend
        )

        if canonical_backend != "auto":
            key = canonical_family, canonical_backend
            if key not in self._specs:
                available = ", ".join(
                    spec.backend for spec in self.list(canonical_family)
                )
                raise KernelNotFoundError(
                    f"Unknown backend {canonical_family}.{canonical_backend}; "
                    f"available backends: {available or 'none'}"
                )
            spec = self._specs[key]
            if not spec.is_available():
                missing = ", ".join(spec.missing_requirements())
                raise KernelUnavailableError(
                    f"Backend {canonical_family}.{canonical_backend} requires: "
                    f"{missing}"
                )
            resolved_version, resolved_arch = spec.resolve_variant(version, arch)
            return KernelHandle(spec, resolved_version, resolved_arch)

        errors: list[str] = []
        for spec in sorted(
            self.list(canonical_family),
            key=lambda candidate: candidate.priority,
            reverse=True,
        ):
            if not spec.is_available():
                errors.append(
                    f"{spec.backend}: missing {', '.join(spec.missing_requirements())}"
                )
                continue
            try:
                resolved_version, resolved_arch = spec.resolve_variant(version, arch)
            except KernelVariantError as exc:
                errors.append(f"{spec.backend}: {exc}")
                continue
            return KernelHandle(spec, resolved_version, resolved_arch)

        detail = "; ".join(errors) if errors else "no backends are registered"
        raise KernelUnavailableError(
            f"No usable backend for {canonical_family}: {detail}"
        )


def _variants(
    values: Mapping[int, Sequence[str]] | Iterable[KernelVariant],
) -> tuple[KernelVariant, ...]:
    if isinstance(values, Mapping):
        return tuple(
            KernelVariant(version, tuple(architectures))
            for version, architectures in values.items()
        )
    return tuple(values)


KERNELS = KernelRegistry()


def register_kernel(
    family: str,
    backend: str,
    target: KernelTarget | None = None,
    *,
    variants: Mapping[int, Sequence[str]] | Iterable[KernelVariant] = (),
    default_version: int | None = None,
    requires: Sequence[str] = (),
    priority: int = 0,
    description: str = "",
    replace: bool = False,
) -> KernelSpec | Callable[[KernelCallable], KernelCallable]:
    """Register a callable/target directly or use this function as a decorator."""

    def install(kernel_target: KernelTarget) -> KernelSpec:
        return KERNELS.register(
            KernelSpec(
                family=family,
                backend=backend,
                target=kernel_target,
                variants=_variants(variants),
                default_version=default_version,
                requires=tuple(requires),
                priority=priority,
                description=description,
            ),
            replace=replace,
        )

    if target is not None:
        return install(target)

    def decorator(function: KernelCallable) -> KernelCallable:
        install(function)
        return function

    return decorator


def _register_builtins() -> None:
    for family, definition in KERNEL_CATALOG.items():
        for backend in definition.backends:
            register_kernel(
                family,
                backend.name,
                f"{_PACKAGE_ROOT}.{backend.target}",
                variants=backend.variants(family),
                default_version=backend.default_version,
                requires=backend.requires,
                priority=backend.priority,
                description=backend.description,
            )


_register_builtins()
