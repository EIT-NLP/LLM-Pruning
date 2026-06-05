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

os.environ['TRITON_PRINT_AUTOTUNING']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='0'
os.environ['TRITON_DEBUG']='0'

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

def rounding(
    size: int,
    sparsity: Optional[float]=0.5,
    rounding: Optional[str|int]='',
) -> int:
    sparsity = max(0, min(1, sparsity))
    k = int(size * sparsity)

    # handle rounding strategy
    if rounding == 'even':
        if k % 2 != 0: k += 1
    elif rounding == 'odd':
        if k % 2 == 0: k -= 1
    elif isinstance(rounding, int) and k > rounding and sparsity > 0:
        # round to the nearest multiple of rounding
        if k % rounding != 0: k = rounding * ((k + rounding - 1) // rounding)
    return k


def get_benchmark(self, **kwargs):
        device = kwargs.get('device', 'cuda:0')
        dtype = kwargs.get('dtype', torch.bfloat16)
        mode = kwargs.get('mode', 'mlp-cudagraph')

        M = kwargs.get('M', [1])
        N = kwargs.get('N', 4096)
        K = kwargs.get('K', 4096)
        sparsity = kwargs.get('sparsity', 0.5)
        tag = kwargs.get('tag', '')
        x_log = kwargs.get('x_log', True)

        plot_name = f'static-sparse-mlp-N{N}-K{K}-sparsity{int(sparsity * 100)}'
        
        if tag != '': plot_name += f'-{tag}'

        line_vals = ['torch', 'static NK', 'static KN']
        line_names = ['torch', 'static NK', 'static KN']
        styles = STYLES[:3]
        
        benchmark_args = dict(
            N=N,
            K=K,
            sparsity=sparsity,
            device=device,
            dtype=dtype,
            mode=mode,
        )

        @triton.testing.perf_report((
            triton.testing.Benchmark(
                x_names=['M'],
                x_vals=M,
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
        def benchmark(M, N, K, sparsity, device, dtype, mode, provider):
            if provider == 'torch':
                A = torch.rand((M, K), dtype=dtype, device=device)
                func = torch.nn.Linear(K, N, bias=False, device=device, dtype=dtype)
            elif provider == 'static NK':
                A = torch.rand((M, K), dtype=dtype, device=device)
                func = torch.nn.Linear(K, rounding(N, sparsity, 16), bias=False, device=device, dtype=dtype)
            elif provider == 'static KN':
                A = torch.rand((M, rounding(K, sparsity, 16)), dtype=dtype, device=device)
                func = torch.nn.Linear(N, rounding(K, sparsity, 16), bias=False, device=device, dtype=dtype)

            quantiles = [0.5, 0.2, 0.8]
            test_func = partial(
                lambda a, b: b(a),
                a=A,
                b=func,
            )

            with torch.no_grad():
                try:
                    if '-cudagraph' in mode:
                        ms, min_ms, max_ms = do_bench_cudagraph(test_func, rep=500, quantiles=quantiles)
                    else:
                        ms, min_ms, max_ms = do_bench(test_func, warmup=100, rep=500, quantiles=quantiles)
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
    bench = get_benchmark(
        M=[1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
        N=4096,
        K=4096,
        sparsity=0.5,
        mode='mlp-cudagraph',
        device=device,
        dtype=dtype,
        tag='',
        x_log=True,
    )

    os.makedirs('./triton_benchmark', exist_ok=True)
    bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')