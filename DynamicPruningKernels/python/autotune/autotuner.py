from __future__ import annotations

import hashlib
import inspect
import json
import os
import pathlib
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch
from triton.testing import do_bench, do_bench_cudagraph


ConfigPruner = Callable[[list["JitConfig"], Mapping[str, Any]], list["JitConfig"]]


def _stable_value(value: Any) -> Any:
    if isinstance(value, JitConfig):
        return value.to_json()
    if isinstance(value, Mapping):
        return {str(k): _stable_value(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_value(v) for v in value]
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_stable_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _autotune_verbose() -> bool:
    return _env_flag("JIT_AUTOTUNE_VERBOSE")


def _print_compile_progress(done: int, total: int) -> None:
    width = 24
    filled = width * done // max(1, total)
    bar = "=" * filled + "." * (width - filled)
    print(f"\r[autotune] compile [{bar}] {done}/{total}", end="\n" if done == total else "", flush=True)


_reported_cache_hits: set[tuple[str, str, str]] = set()
_reported_cache_hits_lock = threading.RLock()
_hardware_args_cache: dict[tuple[int, Optional[int]], dict[str, Any]] = {}
_hardware_args_cache_lock = threading.RLock()
_dtype_width_cache: dict[torch.dtype, int] = {}


def _report_cache_hit_once(kernel_id: str, space_hash: str, key_hash: str, message: str) -> None:
    report_key = (kernel_id, space_hash, key_hash)
    with _reported_cache_hits_lock:
        if report_key in _reported_cache_hits:
            return
        _reported_cache_hits.add(report_key)
    print(message, flush=True)


def _bench_latency_ms(result: Any) -> float:
    first = result[0] if isinstance(result, tuple) else result
    if hasattr(first, "latency"):
        return float(first.latency)
    return float(first)


def _normalize_configs(configs: Iterable[Any], config_params: Sequence[str]) -> list["JitConfig"]:
    normalized: list[JitConfig] = []
    for item in configs:
        if isinstance(item, JitConfig):
            normalized.append(item)
        elif isinstance(item, Mapping):
            normalized.append(JitConfig(kwargs={name: item[name] for name in config_params if name in item}))
        else:
            values = tuple(item)
            if len(values) != len(config_params):
                raise ValueError(
                    f"Autotune config expects {len(config_params)} values {tuple(config_params)}, got {len(values)}"
                )
            normalized.append(JitConfig(kwargs=dict(zip(config_params, values))))
    return normalized


def _call_with_available_args(fn: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    signature = inspect.signature(fn)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return fn(**available)
    kwargs = {name: available[name] for name in signature.parameters if name in available}
    try:
        return fn(**kwargs)
    except TypeError:
        if len(signature.parameters) == 1:
            return fn(available)
        raise


def _config_from_any(value: Any, config_params: Sequence[str], *, name: str) -> "JitConfig":
    if isinstance(value, JitConfig):
        return value
    if isinstance(value, Mapping):
        return JitConfig(kwargs={key: value[key] for key in config_params if key in value}, name=name)
    values = tuple(value)
    if len(values) != len(config_params):
        raise ValueError(f"Heuristic config expects {len(config_params)} values {tuple(config_params)}, got {len(values)}")
    return JitConfig(kwargs=dict(zip(config_params, values)), name=name)


def _named_args(available: Mapping[str, Any], config_params: Sequence[str]) -> dict[str, Any]:
    return {
        key: value
        for key, value in available.items()
        if key not in config_params and not isinstance(value, torch.Tensor)
    }


def _hot_cache_key(named_args: Mapping[str, Any], key_names: Optional[Sequence[str]]) -> Any:
    names = key_names if key_names is not None else tuple(named_args)
    values = tuple(named_args[name] for name in names if name in named_args)
    try:
        hash(values)
        return values
    except TypeError:
        return _stable_hash({name: named_args[name] for name in names if name in named_args})


def _clone_outputs(outputs: Sequence[Any]) -> list[Any]:
    return [output.detach().clone() if isinstance(output, torch.Tensor) else deepcopy(output) for output in outputs]


def _restore_outputs(outputs: Sequence[Any], snapshots: Sequence[Any]) -> None:
    for output, snapshot in zip(outputs, snapshots):
        if isinstance(output, torch.Tensor):
            output.copy_(snapshot)


def _first_cuda_tensor(values: Mapping[str, Any]) -> Optional[torch.Tensor]:
    for value in values.values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value
    return None


def _dtype_width(dtype: Any) -> Optional[int]:
    if isinstance(dtype, torch.dtype):
        cached = _dtype_width_cache.get(dtype)
        if cached is None:
            cached = torch.empty((), dtype=dtype).element_size()
            _dtype_width_cache[dtype] = cached
        return cached
    return None


def _smem_size(cc: int) -> int:
    major, minor = divmod(int(cc), 10)
    if major == 8:
        return 167963 if minor == 0 else 101376
    if major in (9, 10):
        return 233472
    return 101376


def _hardware_args(values: Mapping[str, Any]) -> dict[str, Any]:
    tensor = _first_cuda_tensor(values)
    if tensor is None:
        if not torch.cuda.is_available():
            return {}
        device = torch.cuda.current_device()
    else:
        device = tensor.device
    data_width = _dtype_width(values.get("dtype"))
    if data_width is None and tensor is not None:
        data_width = tensor.element_size()
    device_index = device.index if isinstance(device, torch.device) else int(device)
    if device_index is None:
        device_index = torch.cuda.current_device()

    cache_key = (device_index, data_width)
    with _hardware_args_cache_lock:
        cached = _hardware_args_cache.get(cache_key)
        if cached is not None:
            return dict(cached)

    major, minor = torch.cuda.get_device_capability(device)
    cc = major * 10 + minor
    result = {
        "cc": cc,
        "smem_size": _smem_size(cc),
        "num_sm": torch.cuda.get_device_properties(device).multi_processor_count,
    }
    if data_width is not None:
        result["data_width"] = data_width
    with _hardware_args_cache_lock:
        _hardware_args_cache[cache_key] = dict(result)
    return result


def _prepare_parallel_compile() -> None:
    if os.environ.get("TVM_FFI_CUDA_ARCH_LIST") or not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    os.environ["TVM_FFI_CUDA_ARCH_LIST"] = f"{major}.{minor}"


@dataclass(frozen=True)
class JitConfig:
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    name: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "kwargs": _stable_value(self.kwargs)}

    @property
    def cache_key(self) -> str:
        return _stable_hash(self.to_json())


@dataclass(frozen=True)
class AutotuneResult:
    best_config: JitConfig
    latency_ms: float
    timings: Mapping[str, float]
    cache_hit: Optional[str]
    key_hash: str
    space_hash: str


class HotColdJsonCache:
    _hot: dict[tuple[str, str, str], dict[str, Any]] = {}
    _lock = threading.RLock()

    def __init__(self, cache_dir: Optional[str | pathlib.Path] = None, version: int = 1):
        self.cache_dir = pathlib.Path(cache_dir).expanduser() if cache_dir else pathlib.Path(os.environ.get("JIT_AUTOTUNE_CACHE_DIR")).expanduser()
        self.version = version

    def _path(self, kernel_id: str, space_hash: str) -> pathlib.Path:
        return self.cache_dir / f"v{self.version}" / _safe_name(kernel_id) / f"{space_hash}.json"

    def lookup(self, kernel_id: str, space_hash: str, key_hash: str) -> tuple[Optional[dict[str, Any]], Optional[str]]:
        hot_key = (kernel_id, space_hash, key_hash)
        with self._lock:
            cached = self._hot.get(hot_key)
            if cached is not None:
                return deepcopy(cached), "hot"

            path = self._path(kernel_id, space_hash)
            if not path.exists():
                return None, None
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                return None, None

            entry = data.get("entries", {}).get(key_hash)
            if entry is None:
                return None, None
            self._hot[hot_key] = deepcopy(entry)
            return deepcopy(entry), "cold"

    def store(self, kernel_id: str, space_hash: str, key_hash: str, entry: Mapping[str, Any]) -> None:
        hot_key = (kernel_id, space_hash, key_hash)
        path = self._path(kernel_id, space_hash)
        with self._lock:
            self._hot[hot_key] = deepcopy(dict(entry))
            path.parent.mkdir(parents=True, exist_ok=True)

            data: dict[str, Any] = {
                "schema_version": self.version,
                "kernel_id": kernel_id,
                "space_hash": space_hash,
                "entries": {},
            }
            if path.exists():
                try:
                    old_data = json.loads(path.read_text())
                    if isinstance(old_data.get("entries"), dict):
                        data["entries"].update(old_data["entries"])
                except (OSError, json.JSONDecodeError):
                    pass

            data["updated_at"] = time.time()
            data["entries"][key_hash] = _stable_value(entry)
            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
            tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp_path, path)


class JitModule:
    def __init__(
        self,
        module: Any,
        *,
        best_config: Optional[JitConfig] = None,
        tune_result: Optional[AutotuneResult] = None,
        run_attr: str = "kernel",
    ):
        self.module = module
        self.best_config = best_config
        self.tune_result = tune_result
        self.run_attr = run_attr

    def run(self, *args: Any, **kwargs: Any) -> Any:
        target = getattr(self.module, "run", None)
        if target is None:
            target = getattr(self.module, self.run_attr)
        return target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


def wrap_jit_module(
    module: Any,
    *,
    best_config: Optional[JitConfig] = None,
    tune_result: Optional[AutotuneResult] = None,
    run_attr: str = "kernel",
) -> JitModule:
    if isinstance(module, JitModule):
        module.best_config = best_config or module.best_config
        module.tune_result = tune_result or module.tune_result
        module.run_attr = run_attr
        return module
    return JitModule(module, best_config=best_config, tune_result=tune_result, run_attr=run_attr)


def _bind_runtime_config(
    module: JitModule,
    config: JitConfig,
    runtime_config_params: Sequence[str],
) -> JitModule:
    if not runtime_config_params:
        return module

    target = getattr(module.module, "run", None)
    if target is None:
        target = getattr(module.module, module.run_attr)
    runtime_config_args = tuple(config.kwargs[name] for name in runtime_config_params)

    def run_with_runtime_config(*args: Any, **kwargs: Any) -> Any:
        return target(*args, *runtime_config_args, **kwargs)

    setattr(module, "run", run_with_runtime_config)
    return module


class JitAutotuner:
    def __init__(
        self,
        kernel_id: str,
        configs: Sequence[JitConfig],
        *,
        key: Optional[Sequence[str]],
        compile_config_params: Optional[Sequence[str]] = None,
        runtime_config_params: Sequence[str] = (),
        cache_dir: Optional[str | pathlib.Path] = None,
        cudagraph: bool = False,
        compile_workers: Optional[int] = None,
        warmup: int = 25,
        rep: int = 100,
    ):
        if not configs:
            raise ValueError("JitAutotuner requires at least one config")
        self.kernel_id = kernel_id
        self.configs = list(configs)
        self.key = tuple(key) if key is not None else None
        self.compile_config_params = (
            tuple(compile_config_params) if compile_config_params is not None else None
        )
        self.runtime_config_params = tuple(runtime_config_params)
        self.cache = HotColdJsonCache(cache_dir=cache_dir)
        self.cudagraph = cudagraph
        self.compile_workers = compile_workers
        self.warmup = warmup
        self.rep = rep

    def tune(
        self,
        named_args: Mapping[str, Any],
        *,
        compile_fn: Callable[[JitConfig], JitModule],
        run_fn: Callable[[JitModule, JitConfig], None],
        outputs: Sequence[Any],
        force_tune: bool = False,
    ) -> tuple[AutotuneResult, JitModule]:
        space = {"kernel_id": self.kernel_id, "configs": self.configs}
        if self.compile_config_params is not None:
            space.update(
                compile_config_params=self.compile_config_params,
                runtime_config_params=self.runtime_config_params,
            )
        space_hash = _stable_hash(space)
        key_payload = self._key_payload(named_args)
        key_hash = _stable_hash(key_payload)
        config_by_key = {config.cache_key: config for config in self.configs}
        compiled: dict[str, JitModule] = {}

        if not force_tune:
            entry, cache_hit = self.cache.lookup(self.kernel_id, space_hash, key_hash)
            if entry is not None:
                cached_config = config_by_key.get(entry.get("config_key"))
                if cached_config is not None:
                    if _autotune_verbose():
                        latency = float(entry.get("latency_ms", 0.0))
                        _report_cache_hit_once(
                            self.kernel_id,
                            space_hash,
                            key_hash,
                            f"[autotune] cache hit on {self._format_key(key_payload)} ({cache_hit}); "
                            f"best_config={dict(cached_config.kwargs)}, latency={latency:.4f} ms",
                        )
                    module = compile_fn(cached_config)
                    result = AutotuneResult(
                        best_config=cached_config,
                        latency_ms=float(entry.get("latency_ms", 0.0)),
                        timings=entry.get("timings", {}),
                        cache_hit=cache_hit,
                        key_hash=key_hash,
                        space_hash=space_hash,
                    )
                    return result, module

        if len(self.configs) == 1:
            best_config = self.configs[0]
            best_latency = 0.0
            timings: dict[str, float] = {}
            best_module = compile_fn(best_config)
        else:
            compiled = self._compile_all(compile_fn)
            timings = self._benchmark(compiled=compiled, run_fn=run_fn, outputs=outputs)
            best_config = min(self.configs, key=lambda config: timings.get(config.cache_key, float("inf")))
            best_latency = timings[best_config.cache_key]
            best_module = compiled[best_config.cache_key]

        entry = {
            "key": key_payload,
            "config_key": best_config.cache_key,
            "best_config": best_config.to_json(),
            "latency_ms": best_latency,
            "timings": timings,
        }
        self.cache.store(self.kernel_id, space_hash, key_hash, entry)
        if _autotune_verbose():
            print(
                f"[autotune] best config on {self._format_key(key_payload)}: "
                f"{dict(best_config.kwargs)}, latency={best_latency:.4f} ms",
                flush=True,
            )
        result = AutotuneResult(
            best_config=best_config,
            latency_ms=best_latency,
            timings=timings,
            cache_hit=None,
            key_hash=key_hash,
            space_hash=space_hash,
        )
        return result, best_module

    def _key_payload(self, named_args: Mapping[str, Any]) -> dict[str, Any]:
        keys = self.key if self.key is not None else tuple(named_args)
        return {key: _stable_value(named_args[key]) for key in keys if key in named_args}

    def _compile_group_key(self, config: JitConfig) -> str:
        if self.compile_config_params is None:
            return config.cache_key
        return _stable_hash(
            {name: config.kwargs[name] for name in self.compile_config_params}
        )

    def _assign_compiled_group(
        self,
        compiled: dict[str, JitModule],
        configs: Sequence[JitConfig],
        module: JitModule,
    ) -> None:
        for config in configs:
            candidate = module
            if self.runtime_config_params:
                candidate = wrap_jit_module(
                    module.module,
                    best_config=config,
                    run_attr=module.run_attr,
                )
                candidate = _bind_runtime_config(
                    candidate,
                    config,
                    self.runtime_config_params,
                )
            compiled[config.cache_key] = candidate

    def _compile_all(self, compile_fn: Callable[[JitConfig], JitModule]) -> dict[str, JitModule]:
        if self.compile_config_params is None:
            groups = [[config] for config in self.configs]
        else:
            compile_groups: dict[str, list[JitConfig]] = {}
            for config in self.configs:
                compile_groups.setdefault(self._compile_group_key(config), []).append(config)
            groups = list(compile_groups.values())

        workers = self.compile_workers if self.compile_workers is not None else os.environ.get("JIT_AUTOTUNE_COMPILE_WORKERS", 8)
        workers = max(1, min(int(workers), len(groups)))

        verbose = _autotune_verbose()
        if verbose:
            if len(groups) == len(self.configs):
                message = f"[autotune] compiling {len(groups)} configs with {workers} workers"
            else:
                message = (
                    f"[autotune] compiling {len(groups)} modules for "
                    f"{len(self.configs)} configs with {workers} workers"
                )
            print(message, flush=True)

        if workers == 1:
            compiled: dict[str, JitModule] = {}
            for done, group in enumerate(groups, start=1):
                module = compile_fn(group[0])
                self._assign_compiled_group(compiled, group, module)
                if verbose:
                    _print_compile_progress(done, len(groups))
            return compiled

        _prepare_parallel_compile()
        compiled: dict[str, JitModule] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(compile_fn, group[0]): group
                for group in groups
            }
            for done, future in enumerate(as_completed(futures), start=1):
                group = futures[future]
                self._assign_compiled_group(compiled, group, future.result())
                if verbose:
                    _print_compile_progress(done, len(groups))
        return compiled

    def _benchmark(
        self,
        *,
        compiled: Mapping[str, JitModule],
        run_fn: Callable[[JitModule, JitConfig], None],
        outputs: Sequence[Any],
    ) -> dict[str, float]:
        snapshots = _clone_outputs(outputs)
        timings: dict[str, float] = {}
        try:
            for config in self.configs:
                module = compiled[config.cache_key]

                def call_once() -> None:
                    _restore_outputs(outputs, snapshots)
                    run_fn(module, config)

                if self.cudagraph:
                    latency = _bench_latency_ms(do_bench_cudagraph(call_once))
                else:
                    latency = _bench_latency_ms(do_bench(call_once, warmup=self.warmup, rep=self.rep))
                timings[config.cache_key] = latency
        finally:
            _restore_outputs(outputs, snapshots)
        return timings

    @staticmethod
    def _format_key(key_payload: Mapping[str, Any]) -> str:
        return ", ".join(f"{key}={_stable_value(value)}" for key, value in key_payload.items())


def autotune(
    *,
    kernel_id: str,
    config_params: Sequence[str],
    configs: Sequence[Any] | Callable[..., Sequence[Any]] = [],
    key: Optional[Sequence[str]] = None,
    prune_configs_by: Optional[ConfigPruner] = None,
    runtime_params: Sequence[str] = (),
    runtime_config_params: Sequence[str] = (),
    restore_params: Sequence[str] = (),
    heuristic: Optional[Callable[..., Any]] = None,
    cache_dir: Optional[str | pathlib.Path] = None,
    enabled: bool = True,
    force_tune: bool = False,
    cudagraph: bool = False,
    compile_workers: Optional[int] = None,
    run_attr: str = "kernel",
    warmup: int = 25,
    rep: int = 100,
) -> Callable[[Callable[..., Any]], Callable[..., JitModule]]:
    def decorator(compile_fn: Callable[..., Any]) -> Callable[..., JitModule]:
        runtime_config_names = tuple(runtime_config_params)
        unknown_runtime_configs = set(runtime_config_names).difference(config_params)
        if unknown_runtime_configs:
            unknown = ", ".join(sorted(unknown_runtime_configs))
            raise ValueError(
                f"runtime_config_params must be included in config_params, got: {unknown}"
            )
        runtime_config_set = set(runtime_config_names)
        compile_config_names = tuple(name for name in config_params if name not in runtime_config_set)
        compile_config_name_set = set(compile_config_names)
        signature = inspect.signature(compile_fn)
        signature_param_names = set(signature.parameters)
        default_arguments = {
            name: param.default
            for name, param in signature.parameters.items()
            if param.default is not inspect.Parameter.empty
        }
        runtime_names = set(runtime_params) | set(restore_params)
        hot_modules: dict[Any, JitModule] = {}
        hot_modules_lock = threading.RLock()

        @wraps(compile_fn)
        def wrapped(*args: Any, **kwargs: Any) -> JitModule:
            do_autotune = bool(kwargs.pop("autotune", enabled))
            runtime_force_tune = bool(kwargs.pop("force_tune", force_tune))
            runtime_cudagraph = bool(kwargs.pop("cudagraph", cudagraph))

            runtime_values = {name: kwargs.pop(name) for name in list(runtime_names) if name in kwargs}
            runtime_values.update({name: kwargs.pop(name) for name in list(kwargs) if name not in signature_param_names})

            if args:
                bound = signature.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                bound_arguments = bound.arguments
            else:
                bound_arguments = dict(default_arguments)
                bound_arguments.update(kwargs)

            available = {**bound_arguments, **runtime_values}
            for name, value in _hardware_args(available).items():
                available.setdefault(name, value)
            named_args = _named_args(available, config_params)
            hot_key: Any = None
            if do_autotune and not runtime_force_tune:
                hot_key = _hot_cache_key(named_args, key)
                with hot_modules_lock:
                    hot_module = hot_modules.get(hot_key)
                if hot_module is not None:
                    return hot_module

            config_space = _call_with_available_args(configs, available) if callable(configs) else configs
            valid_configs = _normalize_configs(config_space, config_params)
            if prune_configs_by is not None:
                valid_configs = list(prune_configs_by(valid_configs, named_args))
            if not valid_configs and heuristic is not None:
                valid_configs = [_config_from_any(_call_with_available_args(heuristic, available), config_params, name="heuristic")]
            if not valid_configs:
                raise ValueError(f"All configs were pruned for kernel '{kernel_id}'")

            def compile_with_config(config: JitConfig) -> JitModule:
                candidate_args = dict(bound_arguments)
                candidate_args.update(
                    {
                        name: value
                        for name, value in config.kwargs.items()
                        if name in compile_config_name_set
                    }
                )
                return wrap_jit_module(compile_fn(**candidate_args), best_config=config, run_attr=run_attr)

            if not do_autotune:
                if heuristic is None:
                    raise ValueError(f"Kernel '{kernel_id}' requires heuristic when autotune is disabled")
                config = _config_from_any(_call_with_available_args(heuristic, available), config_params, name="heuristic")
                return _bind_runtime_config(
                    compile_with_config(config), config, runtime_config_names
                )

            runtime_call_args = tuple(available[name] for name in runtime_params)
            outputs = [available[name] for name in restore_params]

            def run_candidate(module: JitModule, config: JitConfig) -> None:
                module.run(*runtime_call_args)

            tuner = JitAutotuner(
                kernel_id,
                valid_configs,
                key=key,
                compile_config_params=compile_config_names if runtime_config_names else None,
                runtime_config_params=runtime_config_names,
                cache_dir=cache_dir,
                cudagraph=runtime_cudagraph,
                compile_workers=compile_workers,
                warmup=warmup,
                rep=rep,
            )
            result, module = tuner.tune(
                named_args,
                compile_fn=compile_with_config,
                run_fn=run_candidate,
                outputs=outputs,
                force_tune=runtime_force_tune,
            )
            module.best_config = result.best_config
            module.tune_result = result
            module = _bind_runtime_config(module, result.best_config, runtime_config_names)
            if hot_key is not None:
                with hot_modules_lock:
                    hot_modules[hot_key] = module
            return module

        return wrapped

    return decorator
