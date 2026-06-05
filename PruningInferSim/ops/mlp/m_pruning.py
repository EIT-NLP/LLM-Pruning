import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl

from functools import partial
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
from triton.testing import do_bench, do_bench_cudagraph

from .base import _PruningMLPKernel, DenseMLPKernel, ACT2FUNC
from .triton_kernel.mlp import BMSparseMLP

os.environ['TRITON_PRINT_AUTOTUNING']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='0'
os.environ['TRITON_DEBUG']='0'

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

def backend_dispatch(backend: str):
    if backend == 'tilelang':
        raise NotImplementedError('tilelang backend is not supported')
    else:
        return BMSparseMLP

class MSparseMLPKernel(_PruningMLPKernel):
    """
    Kernel Implementation:

    Token-wise pruning across whole ffn layer

    For route mask with shape [M], we feed the sorted mask and indices to kernels, now each CTA can skip current block when sum(mask[bm_id * BM:(bm_id + 1) * BM]) == 0
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def base_prefill(
        cls,
        x: torch.Tensor,
        w_up: torch.Tensor,
        w_gate: Optional[torch.Tensor]=None,
        w_down: Optional[torch.Tensor]=None,
        b_up: Optional[torch.Tensor]=None,
        b_gate: Optional[torch.Tensor]=None,
        route_mask: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        estimated_sparsity: Optional[float]=0,
        impl: Optional[str]='auto',
        **kwargs,
    ):
        """
        An implementation of token sparse mlp.

        Args:
            x: torch.Tensor with shape [batch_size, seqlen, hidden_size]
            w_up: nn.Parameter with shape [intermediate_size, hidden_size], the w_up in glu, which is also the weight for a single mlp layer
            route_mask: torch.Tensor with shape [batch_size, seqlen]
            estimated_sparsity: optinal, float, estimated sparsity of route_mask, for heuristic tiling
        Returns:
            out: torch.Tensor with shape same as x
        """

        if (w_down is None) and (w_gate is None): # single mlp layer
            res, meta = backend_dispatch(backend=kwargs.get('backend', 'triton')).kernel(
                x=x,
                route_mask=route_mask,
                w=w_up,
                b=b_up,
                estimated_sparsity=estimated_sparsity,
                impl=impl,
                **kwargs,
            )
        else:
            up, meta = backend_dispatch(backend=kwargs.get('backend', 'triton')).kernel(
                x=x,
                route_mask=route_mask,
                w=w_up,
                estimated_sparsity=estimated_sparsity,
                impl=impl,
                **kwargs,
            )
            meta.update(**kwargs)
            gate, _ = backend_dispatch(backend=kwargs.get('backend', 'triton')).kernel(
                x=x,
                route_mask=route_mask,
                w=w_gate,
                activation=activation,
                estimated_sparsity=estimated_sparsity,
                impl=impl,
                **meta,
            )
            res = gate * up
            res, _ = backend_dispatch(backend=kwargs.get('backend', 'triton')).kernel(
                x=res,
                route_mask=route_mask,
                w=w_down,
                estimated_sparsity=estimated_sparsity,
                impl=impl,
                **meta,
            )

        return res

    @classmethod
    def base_decode(cls, **kwargs):
        return cls.base_prefill(**kwargs)
    
    @classmethod
    def forward(
        cls,
        x: torch.Tensor,
        w_up: torch.Tensor,
        w_gate: Optional[torch.Tensor]=None,
        w_down: Optional[torch.Tensor]=None,
        b_up: Optional[torch.Tensor]=None,
        b_gate: Optional[torch.Tensor]=None,
        route_mask: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        estimated_sparsity: Optional[float]=0,
        prefill_impl: Optional[str]='auto',
        enable_autotune: Optional[bool]=False,
        **kwargs,
    ):
        assert x.dim() == 3
        assert w_up.shape[1] == x.shape[-1]

        if w_down is not None: assert w_down.shape[0] == x.shape[-1]
        if w_gate is not None: assert w_up.shape == w_gate.shape
        return cls.base_prefill(
            x=x,
            w_up=w_up,
            w_gate=w_gate,
            w_down=w_down,
            b_up=b_up,
            b_gate=b_gate,
            route_mask=route_mask,
            activation=activation,
            estimated_sparsity=estimated_sparsity,
            impl=prefill_impl,
            enable_autotune=enable_autotune,
            **kwargs,
        )
    
    def _ref_forward(
        cls,
        x: torch.Tensor,
        w_up: torch.Tensor,
        w_gate: Optional[torch.Tensor]=None,
        w_down: Optional[torch.Tensor]=None,
        b_up: Optional[torch.Tensor]=None,
        b_gate: Optional[torch.Tensor]=None,
        b_down: Optional[torch.Tensor]=None,
        route_mask: Optional[torch.Tensor]=None,
        activation: Optional[str]='identity',
        **kwargs,
    ):
        if (w_down is None) and (w_gate is None): # single mlp layer
            res = DenseMLPKernel.forward(
                x=x,
                w_up=w_up,
                b_up=b_up,
                **kwargs,
            )
            if route_mask is not None:
                res.masked_fill_(route_mask.logical_not()[:, :, None], 0)
        elif w_down is None: # glu
            res = DenseMLPKernel.forward(
                x=x,
                w_up=w_up,
                w_gate=w_gate,
                b_up=b_up,
                b_gate=b_gate,
                activation=activation,
                **kwargs,
            )
            if route_mask is not None:
                res.masked_fill_(route_mask.logical_not()[:, :, None], 0)
        else: # glu + down_proj fuse
            res = DenseMLPKernel.forward(
                x=x,
                w_up=w_up,
                w_gate=w_gate,
                b_up=b_up,
                b_gate=b_gate,
                activation=activation,
                **kwargs,
            )
            res = DenseMLPKernel.forward(
                x=res,
                w_up=w_down,
                b_up=b_down,
                **kwargs,
            )
            if route_mask is not None:
                res.masked_fill_(route_mask.logical_not()[:, :, None], 0)

        return res
    
    def precision_diff(self, repeat: Optional[int]=3, **kwargs):
        from copy import deepcopy
        dtype = kwargs.get('dtype', torch.bfloat16)
        device = kwargs.get('device', 'cuda:0')
        
        import random
        hidden_size = kwargs.get('hidden_size', 4096)
        intermediate_size = kwargs.get('intermediate_size', 4096)
        up = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        gate = nn.Linear(hidden_size, intermediate_size, bias=False, device=device, dtype=dtype)
        down = nn.Linear(intermediate_size, hidden_size, bias=False, device=device, dtype=dtype)

        for ffn_type in ['mlp']:
            for i in range(repeat):
                seqlen = random.randint(1024, 4096)
                sparsity = min(0.9, random.random())

                x = torch.randn((1, seqlen, hidden_size), dtype=dtype, device=device)
                route_mask = torch.rand([1, seqlen], device=device) > sparsity

                if ffn_type == 'mlp':
                    ref_out = self._ref_forward(deepcopy(x), deepcopy(up.weight), route_mask=deepcopy(route_mask), activation='silu')
                    out = self.forward(deepcopy(x), deepcopy(up.weight), route_mask=deepcopy(route_mask), activation='silu', estimated_sparsity=sparsity)
                elif ffn_type == 'glu':
                    ref_out = self._ref_forward(deepcopy(x), deepcopy(up.weight), w_gate=deepcopy(gate.weight), route_mask=deepcopy(route_mask), activation='silu')
                    out = self.forward(deepcopy(x), deepcopy(up.weight), w_gate=deepcopy(gate.weight), route_mask=deepcopy(route_mask), activation='silu', estimated_sparsity=sparsity)
                elif ffn_type == 'ffn':
                    ref_out = self._ref_forward(deepcopy(x), deepcopy(up.weight), w_gate=deepcopy(gate.weight), w_down=deepcopy(down.weight), route_mask=deepcopy(route_mask), activation='silu')
                    out = self.forward(deepcopy(x), deepcopy(up.weight), w_gate=deepcopy(gate.weight), w_down=deepcopy(down.weight), route_mask=deepcopy(route_mask), activation='silu', estimated_sparsity=sparsity)

                diff = torch.abs(out - ref_out)
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                is_pass = mean_diff < 1e-2 and max_diff < 1e-2

                if is_pass:
                    print(f"✅ pass in [{1}, {seqlen}, {hidden_size}] @ [{intermediate_size}, {hidden_size}] with sparsity {sparsity}, max_diff {max_diff}, mean_diff {mean_diff}")
                else:
                    print(f"❌ fail in [{1}, {seqlen}, {hidden_size}] @ [{intermediate_size}, {hidden_size}] with sparsity {sparsity}, max_diff {max_diff}, mean_diff {mean_diff}")
    
    def get_benchmark(self, **kwargs):
        bench_name = kwargs.get('bench_name', 'M')
        bench_range = kwargs.get('bench_range', [2**i for i in range(0, 15)])
        kwargs.pop(bench_name, None)

        device = kwargs.get('device', 'cuda:0')
        dtype = kwargs.get('dtype', torch.bfloat16)

        M = kwargs.get('M', 4096)
        N = kwargs.get('N', 4096)
        K = kwargs.get('K', 4096)
        sparsity = kwargs.get('sparsity', 0.5)
        mode = kwargs.get('mode', 'glu')
        tag = kwargs.get('tag', '')
        impl = kwargs.get('impl', 'sort')
        x_log = kwargs.get('x_log', True)

        if bench_name == 'M':
            plot_name = f'BM-sparse-mlp-N{N}-K{K}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = dict(
                up=torch.tensor([False, True], device='cpu').repeat(1, bench_range[-1]),
            )
        elif bench_name == 'N':
            plot_name = f'BM-sparse-mlp-M{M}-K{K}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = dict(
                up=torch.tensor([False, True], device='cpu').repeat(1, M),
            )
        elif bench_name == 'K':
            plot_name = f'BM-sparse-mlp-M{M}-N{N}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = dict(
                up=torch.tensor([False, True], device='cpu').repeat(1, M),
            )
        elif bench_name == 'sparsity':
            plot_name = f'BM-sparse-mlp-M{M}-N{N}-K{K}-{mode}'
            active_mask = dict(
                up=torch.tensor([False, True], device='cpu').repeat(1, M),
            )
        else:
            raise ValueError(f'Unknown bench_name: {bench_name}')
        
        if tag != '': plot_name += f'-{tag}'

        if isinstance(impl, list):
            line_vals = ['torch'] + impl
            line_names = ['torch'] + impl
            styles = STYLES[:len(impl) + 1]
        else:
            line_vals = ['torch', 'triton']
            line_names = ['torch', 'triton']
            styles = STYLES[:2]
        
        benchmark_args = dict(
            M=M,
            N=N,
            K=K,
            sparsity=sparsity,
            mode=mode,
            mask=active_mask,
            device=device,
            dtype=dtype,
            impl=impl,
        )
        benchmark_args.pop(bench_name)

        @triton.testing.perf_report((
            triton.testing.Benchmark(
                x_names=[bench_name],
                x_vals=bench_range,
                x_log=x_log,
                line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
                line_vals=line_vals,  # Possible values for `line_arg`.
                line_names=line_names,  # Label name for the lines.
                styles=styles,  # Line styles.
                ylabel='TFLOPS',  # Label name for the y-axis.
                plot_name=plot_name,  # Name for the plot. Used also as a file name for saving the plot.
                args=benchmark_args,  # Values for function arguments not in `x_names` and `y_name`.
            )
        ))
        def benchmark(M, N, K, sparsity, mode, mask, device, dtype, impl, provider):
            up = nn.Linear(K, N, bias=False, device=device, dtype=dtype)
            gate = nn.Linear(K, N, bias=False, device=device, dtype=dtype)
            down = nn.Linear(N, K, bias=False, device=device, dtype=dtype)

            x = torch.randn((1, M, K), dtype=dtype, device=device)

            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                func = partial(
                    self._ref_forward,
                    x=x,
                    w_up=up.weight,
                    b_up=up.bias,
                    w_gate=gate.weight if mode in ['glu', 'glu-cudagraph', 'ffn', 'ffn-cudagraph'] else None,
                    b_gate=gate.bias if mode in ['glu', 'glu-cudagraph', 'ffn', 'ffn-cudagraph'] else None,
                    w_down=down.weight if mode in ['ffn', 'ffn-cudagraph'] else None,
                    b_down=down.bias if mode in ['ffn', 'ffn-cudagraph'] else None,
                    activation='silu',
                )
            else:
                func_kwargs = dict(
                    x=x,
                    w_up=up.weight,
                    b_up=up.bias,
                    w_gate=gate.weight if mode in ['glu', 'glu-cudagraph', 'ffn', 'ffn-cudagraph'] else None,
                    b_gate=gate.bias if mode in ['glu', 'glu-cudagraph', 'ffn', 'ffn-cudagraph'] else None,
                    w_down=down.weight if mode in ['ffn', 'ffn-cudagraph'] else None,
                    b_down=down.bias if mode in ['ffn', 'ffn-cudagraph'] else None,
                    activation='silu',
                    route_mask=mask['up'][:, :M].to(device) > sparsity,
                    estimated_sparsity=sparsity,
                    prefill_impl=provider if provider != 'triton' else impl,
                )

                if '.' in provider:
                    backend = provider.split('.')[-1]
                    func_kwargs['prefill_impl'] = func_kwargs['prefill_impl'].split('.')[0]
                else: backend = 'triton'
                func_kwargs['backend'] = backend

                func = partial(
                    self.forward,
                    **func_kwargs,
                )

            with torch.no_grad():
                try:
                    if '-cudagraph' in mode:
                        ms, min_ms, max_ms = do_bench_cudagraph(func, rep=500, quantiles=quantiles)
                    else:
                        ms, min_ms, max_ms = do_bench(func, warmup=100, rep=500, quantiles=quantiles)
                    print(f"✅ finish {provider} in [M:{M}, N:{N}, K:{K}] with sparsity {sparsity}, {mode} mode")
                except Exception as e:
                    print(e)
                    return 0, 0, 0
            
            if 'mlp' in mode: tflops = lambda ms: (2 * (M * N * K) * 1e-12) / (ms * 1e-3)
            elif 'glu' in mode: tflops = lambda ms: (2 * (M * N * K * 2 + M * N) * 1e-12) / (ms * 1e-3)
            elif 'ffn' in mode: tflops = lambda ms: (2 * (M * N * K * 3 + M * N) * 1e-12) / (ms * 1e-3)

            return tflops(ms), tflops(min_ms), tflops(max_ms)
        
        return benchmark
        

def run_test(*args, **kwargs):
    cc = torch.cuda.get_device_capability("cuda")[0]
    device = kwargs.pop('device', 'cuda:0')
    dtype = kwargs.pop('dtype', torch.float16 if cc < 9 else torch.bfloat16)

    print(f"Detected device capability: {cc}, using dtype {dtype} for testing")

    kernel = MSparseMLPKernel()
    print(f"1. Test precision diff")
    kernel.precision_diff(device=device, dtype=dtype)

    print(f"2. Test throughput")
    bench = kernel.get_benchmark(
        bench_name='M',
        bench_range=[2 ** i for i in range(0, 13)],
        M=4096,
        N=14336,
        K=4096,
        sparsity=0.5,
        device=device,
        dtype=dtype,
        impl=['sort_offline', 'sort_online'],
        mode='mlp-cudagraph',
        tag='triton_v2',
        x_log=True,
    )

    os.makedirs('./triton_benchmark', exist_ok=True)
    bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')