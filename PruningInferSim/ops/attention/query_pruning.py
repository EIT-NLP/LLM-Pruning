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

from .base import _PruningAttentionKernel, DenseAttentionKernel
from .triton_kernel.prefill import QuerySparsePrefill
from .triton_kernel.decode import QuerySparseDecode

os.environ['TRITON_PRINT_AUTOTUNING']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='0'
os.environ['TRITON_DEBUG']='0'

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

class MSparseAttentionKernel(_PruningAttentionKernel):
    """
    Kernel Implementation:

    Token-wise query sprase attention for prefill & decode

    For a chunk with active query [5, 6, 31, 32]
    we generate a causal mask from position 5 to 32
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def base_prefill(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: Optional[torch.Tensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        estimated_sparsity: Optional[float]=0,
        impl: Optional[str]='sort',
        **kwargs,
    ):
        """
        An implementation of query sparse attention (causal) in prefill.

        The main pipeline is based on FLA's prefill flash-attn kernel.

        Args:
            q: torch.Tensor with shape [batch_size, query_length, num_query_heads, head_dim]
            k: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            v: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            route_mask: torch.Tensor with shape [batch_size, query_length,], the mask for active tokens, which including other mask for query (e.g., left padding mask)
            pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        Returns:
            out: torch.Tensor with shape same as q or kwargs.flatten_q
        """
        return QuerySparsePrefill.kernel(
            q=q,
            k=k,
            v=v,
            route_mask=route_mask,
            pad_offset=pad_offset,
            estimated_sparsity=estimated_sparsity,
            impl=impl,
            **kwargs,
        )
    
    @classmethod
    def base_decode(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: Optional[torch.Tensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        impl: Optional[str]='split',
        **kwargs,
    ):
        """
        An implementation of query sparse attention (causal) in decode with XQA.

        The main pipeline is based on FLA's prefill flash-attn kernel.

        Args:
            q: torch.Tensor with shape [batch_size, 1, num_heads_q, head_dim]
            k: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
            v: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
            route_mask: torch.Tensor with shape [batch_size, seqlen], the route mask for query
            pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        Returns:
            out: torch.Tensor with shape [batch_size, 1, num_heads, head_dim]
        """
        return QuerySparseDecode.kernel(
            q=q,
            k=k,
            v=v,
            route_mask=route_mask,
            pad_offset=pad_offset,
            impl=impl,
            **kwargs,
        )

    @classmethod    
    def forward(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        route_mask: Optional[torch.Tensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        estimated_sparsity: Optional[float]=0,
        prefill_impl: Optional[str]='auto',
        decode_impl: Optional[str]='auto',
        enable_autotune: Optional[bool]=False,
        **kwargs,
    ):
        assert k.dim() == 4 and q.dim() == 4
        if route_mask.shape[1] > 1:
            return cls.base_prefill(q, k, v, route_mask, pad_offset, estimated_sparsity=estimated_sparsity, impl=prefill_impl, enable_autotune=enable_autotune)
        else:
            assert k.dim() == 4 and q.dim() == 4
            return cls.base_decode(q, k, v, route_mask, pad_offset, impl=decode_impl, enable_autotune=enable_autotune)
    
    def _ref_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        route_mask: Optional[torch.Tensor]=None,
    ):
        o = DenseAttentionKernel.forward(q, k, v, None, attention_mask, pad_offset)
        if route_mask is not None: o.masked_fill_(route_mask.logical_not()[:, :, None, None], 0)
        return o
    
    def precision_diff(self, repeat: Optional[int]=3, **kwargs):
        from copy import deepcopy
        dtype = kwargs.get('dtype', torch.bfloat16)
        device = kwargs.get('device', 'cuda:0')
        
        import random
        num_q_heads = kwargs.get('num_q_heads', 32)
        num_kv_heads = kwargs.get('num_kv_heads', 8)
        head_dim = kwargs.get('head_dim', 128)

        for is_prefill in [True, False]:
            for i in range(repeat):
                bsz = random.randint(1, 16)
                seqlen = random.randint(1024, 4096)
                sparsity = min(0.9, random.random())

                seqlen_q = seqlen if is_prefill else 1
                seqlen_kv = seqlen

                q = torch.randn((bsz, seqlen_q, num_q_heads, head_dim), dtype=dtype, device=device)
                k = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)
                v = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)

                route_mask = torch.rand([bsz, seqlen_q], device=device) > sparsity
                attention_mask = torch.ones((bsz, seqlen_kv), dtype=torch.bool, device=device)
                for j in range(bsz): attention_mask[j, :j] = False

                if is_prefill: route_mask = route_mask & attention_mask
                pad_offset = seqlen_kv - attention_mask.sum(-1)

                ref_out = self._ref_forward(deepcopy(q), deepcopy(k), deepcopy(v), attention_mask, pad_offset, route_mask)
                out = self.forward(deepcopy(q), deepcopy(k), deepcopy(v), route_mask, pad_offset, estimated_sparsity=sparsity)
                if out.dim() == 3:
                    tmp = torch.zeros_like(ref_out).flatten(0, 1)
                    tmp[route_mask.flatten()] = out
                    out = tmp.reshape(bsz, seqlen_q, num_q_heads, head_dim)

                diff = torch.abs(out - ref_out)
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                is_pass = mean_diff < 1e-2 and max_diff < 1e-2

                if is_pass:
                    print(f"✅ pass in [{bsz}, {seqlen_q}] @ [{bsz}, {seqlen_kv}] with sparsity {sparsity}, max_diff {max_diff}, mean_diff {mean_diff}")
                else:
                    print(f"❌ fail in [{bsz}, {seqlen_q}] @ [{bsz}, {seqlen_kv}] with sparsity {sparsity}, max_diff {max_diff}, mean_diff {mean_diff}")
    
    def get_benchmark(self, **kwargs):
        bench_name = kwargs.get('bench_name', 'seqlen')
        bench_range = kwargs.get('bench_range', [2**i for i in range(3, 15)])
        kwargs.pop(bench_name, None)

        device = kwargs.get('device', 'cuda:0')
        dtype = kwargs.get('dtype', torch.bfloat16)

        bsz = kwargs.get('bsz', 1)
        seqlen = kwargs.get('seqlen', 1024)
        sparsity = kwargs.get('sparsity', 0.5)
        num_q_heads = kwargs.get('num_q_heads', 32)
        num_kv_heads = kwargs.get('num_kv_heads', 8)
        head_dim = kwargs.get('head_dim', 128)
        mode = kwargs.get('mode', 'prefill')
        tag = kwargs.get('tag', '')
        impl = kwargs.get('impl', 'sort')
        x_log = kwargs.get('x_log', True)

        if bench_name == 'seqlen':
            plot_name = f'query-sparse-attn-bsz{bsz}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = torch.tensor([False, True], device='cpu').repeat(bsz, bench_range[-1] // 2)
            if 'decode' in mode: active_mask = torch.tensor([False, True], device='cpu').repeat(bsz // 2).unsqueeze(1)
        elif bench_name == 'bsz':
            plot_name = f'query-sparse-attn-seqlen{seqlen}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = torch.tensor([False, True], device='cpu').repeat(bench_range[-1], seqlen // 2)
            if 'decode' in mode: active_mask = torch.tensor([False, True], device='cpu').repeat(bench_range[-1] // 2).unsqueeze(1)
        elif bench_name == 'sparsity':
            plot_name = f'query-sparse-attn-bsz{bsz}-seqlen{seqlen}-{mode}'
            active_mask = torch.tensor([False, True], device='cpu').repeat(bsz, seqlen // 2)
            if 'decode' in mode: active_mask = torch.tensor([False, True], device='cpu').repeat(bsz // 2).unsqueeze(1)
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
            bsz=bsz,
            seqlen=seqlen,
            sparsity=sparsity,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
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
        def benchmark(bsz, seqlen, sparsity, num_q_heads, num_kv_heads, head_dim, mode, mask, device, dtype, impl, provider):
            seqlen_q = 1 if 'decode' in mode else seqlen
            seqlen_kv = seqlen
            G = num_q_heads // num_kv_heads

            q = torch.randn((bsz, seqlen_q, num_q_heads, head_dim), dtype=dtype, device=device)
            k = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)
            v = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)

            quantiles = [0.5, 0.2, 0.8]
            if provider == 'torch':
                func = partial(
                    self._ref_forward,
                    q=q,
                    k=k,
                    v=v,
                    attention_mask=None,
                    pad_offset=torch.zeros((bsz,), dtype=torch.int32, device=device),
                    route_mask=None,
                )
            else:
                func_kwargs = dict(
                    q=q,
                    k=k,
                    v=v,
                    route_mask=(mask[:bsz, :seqlen_q] > sparsity).to(device),
                    pad_offset=torch.zeros((bsz,), dtype=torch.int32, device=device),
                    estimated_sparsity=sparsity,
                )
                if 'prefill' in mode: func_kwargs['prefill_impl'] = provider if provider != 'triton' else impl
                elif 'decode' in mode: func_kwargs['decode_impl'] = provider if provider != 'triton' else impl

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
                    print(f"✅ finish {provider} in [{bsz}, {seqlen}] with sparsity {sparsity}, {mode} mode")
                except Exception as e:
                    print(e)
                    return 0, 0, 0
            
            tflops = lambda ms: (2 * bsz * num_q_heads * head_dim * seqlen_q * seqlen_kv * 1e-12) / (ms * 1e-3)

            return tflops(ms), tflops(min_ms), tflops(max_ms)
        
        return benchmark

def run_test(*args, **kwargs):
    cc = torch.cuda.get_device_capability("cuda")[0]
    device = kwargs.pop('device', 'cuda:0')
    dtype = kwargs.pop('dtype', torch.float16 if cc < 9 else torch.bfloat16)

    print(f"Detected device capability: {cc}, using dtype {dtype} for testing")

    kernel = MSparseAttentionKernel()
    print(f"1. Test precision diff")
    kernel.precision_diff(device=device, dtype=dtype)

    print(f"2. Test prefill throughput")
    bench = kernel.get_benchmark(
        bench_name='seqlen',
        bench_range=[2 ** i for i in range(10, 16)],
        bsz=1,
        seqlen=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        sparsity=0.5,
        mode='prefill',
        device=device,
        dtype=dtype,
        impl=['ragged', 'dense', 'sort'],
        tag='',
        x_log=True,
    )

    os.makedirs('./triton_benchmark', exist_ok=True)
    bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')

    print(f"3. Test decode throughput")
    bench = kernel.get_benchmark(
        bench_name='seqlen',
        bench_range=[2 ** i for i in range(10, 15)],
        bsz=32,
        seqlen=4096,
        num_q_heads=32,
        num_kv_heads=8,
        head_dim=128,
        sparsity=0.5,
        mode='decode-cudagraph',
        device=device,
        dtype=dtype,
        impl=['split', 'fuse'],
        tag='',
        x_log=True,
    )

    os.makedirs('./triton_benchmark', exist_ok=True)
    bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')

if __name__ == '__main__':
    run_test()