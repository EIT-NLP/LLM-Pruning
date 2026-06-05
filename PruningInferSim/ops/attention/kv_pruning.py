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
from .triton_kernel.prefill_sparse import SparseAttentionPrefill
from .triton_kernel.decode_sparse import SparseAttentionDecode

from .tilelang_kernel.prefill_sparse import SparseAttentionTilelangPrefill
from .tilelang_kernel.decode_sparse import SparseAttentionTilelangDecode

os.environ['TRITON_PRINT_AUTOTUNING']='0'
os.environ['CUDA_LAUNCH_BLOCKING']='0'
os.environ['TRITON_DEBUG']='0'

STYLES = [('blue', '-'), ('red', '-'), ('green', '-'), ('orange', '-'), ('purple', '-'), ('brown', '-'), ('pink', '-'), ('gray', '-'), ('olive', '-'), ('cyan', '-')]

def backend_dispatch(backend: str):
    if backend == 'tilelang':
        return dict(
            prefill=SparseAttentionTilelangPrefill,
            decode=SparseAttentionTilelangDecode,
        )
    else:
        return dict(
            prefill=SparseAttentionPrefill,
            decode=SparseAttentionDecode,
        )

class BlockSparseAttentionKernel(_PruningAttentionKernel):
    """
    Kernel Implementation:

    P@V sparse attention for prefill & decode
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @classmethod
    def base_prefill(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        threshold: float,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        impl: Optional[str]='seer',
        **kwargs,
    ):
        """
        An implementation of sparse attention (causal) in prefill.

        The main pipeline is based on FLA's prefill flash-attn kernel.

        Args:
            q: torch.Tensor with shape [batch_size, query_length, num_query_heads, head_dim]
            k: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            v: torch.Tensor with shape [batch_size, key_length, num_key_heads, head_dim]
            threshold: the BLASST style threshold for local max
            pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        Returns:
            execute_block: torch.Tensor with shape [cdiv(key_length, 16)], manually specified execute block
            out: torch.Tensor with shape same as q or kwargs.flatten_q
        """
        return backend_dispatch(backend=kwargs.get('backend', 'triton'))['prefill'].kernel(
            q=q,
            k=k,
            v=v,
            threshold=threshold,
            pad_offset=pad_offset,
            execute_block=execute_block,
            impl=impl,
            **kwargs,
        )
    
    @classmethod
    def base_decode(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        threshold: float,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        impl: Optional[str]='seer',
        **kwargs,
    ):
        """
        An implementation of sparse attention (causal) in decode with XQA.

        The main pipeline is based on FLA's prefill flash-attn kernel.

        Args:
            q: torch.Tensor with shape [batch_size, 1, num_heads_q, head_dim]
            k: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
            v: torch.Tensor with shape [batch_size, seqlen, num_heads_k, head_dim]
            threshold: the BLASST style threshold for local max
            pad_offset: torch.Tensor with shape [batch_size,], the left padding offset for key and value
        Returns:
            execute_block: torch.Tensor with shape [cdiv(key_length, 16)], manually specified execute block
            out: torch.Tensor with shape [batch_size, 1, num_heads, head_dim]
        """
        return backend_dispatch(backend=kwargs.get('backend', 'triton'))['decode'].kernel(
            q=q,
            k=k,
            v=v,
            threshold=threshold,
            pad_offset=pad_offset,
            execute_block=execute_block,
            impl=impl,
            **kwargs,
        )

    @classmethod    
    def forward(
        cls,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        threshold: float,
        block_size: Optional[int]=64,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        prefill_impl: Optional[str]='seer',
        decode_impl: Optional[str]='seer',
        enable_autotune: Optional[bool]=False,
        **kwargs,
    ):
        assert k.dim() == 4 and q.dim() == 4
        if not q.is_contiguous(): q = q.contiguous()
        if not k.is_contiguous(): k = k.contiguous()
        if not v.is_contiguous(): v = v.contiguous()

        if q.shape[1] > 1:
            return cls.base_prefill(q, k, v, threshold, pad_offset, execute_block, block_size=block_size, impl=prefill_impl, enable_autotune=enable_autotune, **kwargs)
        else:
            assert k.dim() == 4 and q.dim() == 4
            return cls.base_decode(q, k, v, threshold, pad_offset, execute_block, block_size=block_size, impl=decode_impl, enable_autotune=enable_autotune, **kwargs)
    
    def _ref_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        execute_block: Optional[torch.Tensor]=None,
        block_size: Optional[int]=16,
    ):
        if execute_block is not None: # eager mode
            indices = execute_block.nonzero(as_tuple=True)[0]
            ks = rearrange(k, 'b (n s) h d -> b n s h d', s=block_size)[:, indices].flatten(1, 2)
            vs = rearrange(v, 'b (n s) h d -> b n s h d', s=block_size)[:, indices].flatten(1, 2)
        else:
            ks = k
            vs = v

        o = DenseAttentionKernel.forward(q, ks, vs, None, None, pad_offset)
        return o
    
    def precision_diff(self, repeat: Optional[int]=3, **kwargs):
        from copy import deepcopy
        dtype = kwargs.get('dtype', torch.bfloat16)
        device = kwargs.get('device', 'cuda:0')
        
        import random
        num_q_heads = kwargs.get('num_q_heads', 32)
        num_kv_heads = kwargs.get('num_kv_heads', 8)
        head_dim = kwargs.get('head_dim', 128)

        for is_prefill in [True]:
            for i in range(repeat):
                bsz = random.randint(1, 16)
                seqlen = random.sample(list(range(1024, 4097, 128)), 1)[0]
                # execute_block = torch.randint(0, 2, (triton.cdiv(seqlen, 16),), device=device)
                execute_block = torch.ones((bsz, num_kv_heads, triton.cdiv(seqlen, 16), triton.cdiv(seqlen, 16)), device=device, dtype=torch.int8)

                seqlen_q = seqlen if is_prefill else 1
                seqlen_kv = seqlen

                q = torch.randn((bsz, seqlen_q, num_q_heads, head_dim), dtype=dtype, device=device)
                k = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)
                v = torch.randn((bsz, seqlen_kv, num_kv_heads, head_dim), dtype=dtype, device=device)

                attention_mask = torch.ones((bsz, seqlen_kv), dtype=torch.bool, device=device)
                # for j in range(bsz): attention_mask[j, :j] = False

                pad_offset = seqlen_kv - attention_mask.sum(-1)

                ref_out = self._ref_forward(deepcopy(q), deepcopy(k), deepcopy(v), attention_mask, pad_offset, None)
                out = self.forward(deepcopy(q), deepcopy(k), deepcopy(v), -1, 64, pad_offset, execute_block, enable_autotune=False, prefill_impl='seer', backend='tilelang')

                diff = torch.abs(out - ref_out)
                mean_diff = diff.mean().item()
                max_diff = diff.max().item()
                is_pass = mean_diff < 1e-2 and max_diff < 1e-2

                if is_pass:
                    print(f"✅ pass in [{bsz}, {seqlen_q}] @ [{bsz}, {seqlen_kv}], max_diff {max_diff}, mean_diff {mean_diff}")
                else:
                    print(f"❌ fail in [{bsz}, {seqlen_q}] @ [{bsz}, {seqlen_kv}], max_diff {max_diff}, mean_diff {mean_diff}")
    
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
        impl = kwargs.get('impl', 'blasst')
        x_log = kwargs.get('x_log', True)

        if bench_name == 'seqlen':
            plot_name = f'query-sparse-attn-bsz{bsz}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = torch.rand((bsz, num_kv_heads, bench_range[-1] // 16, bench_range[-1] // 16), device='cpu')
        elif bench_name == 'bsz':
            plot_name = f'query-sparse-attn-seqlen{seqlen}-sparsity{int(sparsity * 100)}-{mode}'
            active_mask = torch.rand((bench_range[-1], num_kv_heads, seqlen // 16, seqlen // 16), device='cpu')
        elif bench_name == 'sparsity':
            plot_name = f'query-sparse-attn-bsz{bsz}-seqlen{seqlen}-{mode}'
            active_mask = torch.rand((bsz, num_kv_heads, seqlen // 16, seqlen // 16), device='cpu')
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
                    execute_block=None,
                )
            else:
                if 'blasst' in provider: matched_mask = mask[0, 0, :, 0]
                elif 'seer' in provider: matched_mask = mask[:, :, :(seqlen_q - 1 + 16) // 16, :(seqlen_kv - 1 + 16) // 16]
                else: matched_mask = mask
                func_kwargs = dict(
                    q=q,
                    k=k,
                    v=v,
                    threshold=-1,
                    block_size=64,
                    pad_offset=torch.zeros((bsz,), dtype=torch.int32, device=device),
                    execute_block=(matched_mask > sparsity).to(device),
                )
                if 'prefill' in mode: func_kwargs['prefill_impl'] = provider if provider != 'triton' else impl
                elif 'decode' in mode: func_kwargs['decode_impl'] = provider if provider != 'triton' else impl

                if '.' in provider:
                    backend = provider.split('.')[-1]
                    if 'prefill' in mode: func_kwargs['prefill_impl'] = func_kwargs['prefill_impl'].split('.')[0]
                    if 'decode' in mode: func_kwargs['decode_impl'] = func_kwargs['decode_impl'].split('.')[0]
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

    kernel = BlockSparseAttentionKernel()
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
        impl=['seer.triton', 'seer.tilelang'],
        tag='',
        x_log=True,
    )

    os.makedirs('./triton_benchmark', exist_ok=True)
    bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')

    # print(f"3. Test decode throughput")
    # bench = kernel.get_benchmark(
    #     bench_name='seqlen',
    #     bench_range=[2 ** i for i in range(10, 15)],
    #     bsz=32,
    #     seqlen=4096,
    #     num_q_heads=32,
    #     num_kv_heads=8,
    #     head_dim=128,
    #     sparsity=0.5,
    #     mode='decode-cudagraph',
    #     device=device,
    #     dtype=dtype,
    #     impl=['blasst', 'seer'],
    #     tag='',
    #     x_log=True,
    # )

    # os.makedirs('./triton_benchmark', exist_ok=True)
    # bench.run(print_data=True, show_plots=False, save_path='./triton_benchmark')

if __name__ == '__main__':
    run_test()