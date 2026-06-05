import os
import argparse
import torch

from transformers import set_seed

from prologue import init_model, wrap_model
from benchmark.profiler import ModelProfiler
from utils import record_results
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser(description="Pruning Wrapper")
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument("--model_name", type=str, default='', help="Model name")
    parser.add_argument("--model_path", type=str, default='', help="Path to the model")
    parser.add_argument("--dynamic", type=str, default="token_dynamic", help="Dynamic type")
    parser.add_argument("--style", type=str, default="skipgpt", help="Wrapper style")
    parser.add_argument("--config_name", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")

    parser.add_argument("--benchmark_metric", type=str, default="ttft", help="Benchmark metric")
    parser.add_argument("--batch_size", type=int, default=[1], nargs='+', help="Batch size")
    parser.add_argument("--seq_len", type=int, default=[2048], nargs='+', help="Sequence length")
    parser.add_argument("--sparsity", type=float, default=-1, nargs='+', help="Benchmark sparsity")
    parser.add_argument("--num_warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--num_repeat", type=int, default=500, help="Number of repeat iterations")
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA graph")
    parser.add_argument("--liger_kernel", action="store_true", help="Enable Liger Kernel")
    parser.add_argument("--inplace_update_kvcache", action="store_true", help="Enable inplace update kvcache")
    parser.add_argument("--torch_profiler", action="store_true", help="Enable torch profiler")
    parser.add_argument("--ncu_profiler", action="store_true", help="Enable ncu profiler")

    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.liger_kernel:
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen3, apply_liger_kernel_to_qwen2
        liger_kernel_kwargs = {'rope': False, 'rms_norm': True, 'swiglu': False}
        apply_liger_kernel_to_llama(**liger_kernel_kwargs)
        apply_liger_kernel_to_qwen2(**liger_kernel_kwargs)
        apply_liger_kernel_to_qwen3(**liger_kernel_kwargs)
    
    cc = torch.cuda.get_device_capability("cuda")[0]
    dtype = torch.float16 if cc < 9 else torch.bfloat16
    print(f"[INFO] CUDA capability: {cc}, using dtype {dtype}")

    model, tokenizer, config = init_model(args)

    profiler_name = f"{args.benchmark_metric}_{args.batch_size[-1]}_{args.seq_len[-1]}_{args.dynamic}_{args.style}_{args.config_name}_{args.sparsity[-1]}_{args.cuda_graph}"
    config_path = ''
    res_dict = {}
    for i, sparsity in enumerate(args.sparsity):
        set_seed(args.seed + i)
        bench_model, config_path = wrap_model(deepcopy(model), config, args, sparsity)
        profiler = ModelProfiler(
            model=bench_model,
            tokenizer=tokenizer,
            args=args,
            dtype=dtype,
            profiler_name=profiler_name,
        )
        if args.ncu_profiler:
            if args.benchmark_metric == 'ttft': profiler.ncu_ttft(batch_size=args.batch_size[0], seq_len=args.seq_len[0])
        else:
            for batch_size in args.batch_size:
                for seq_len in args.seq_len:
                    try:
                        if args.benchmark_metric == "ttft":
                            token_num = batch_size * seq_len
                            ms, min_ms, max_ms = profiler.profile_ttft(
                                batch_size=batch_size,
                                seq_len=seq_len,
                                warmup=args.num_warmup,
                                repeat=args.num_repeat,
                                cuda_graph=args.cuda_graph,
                                sparsity=sparsity,
                            )
                            print(f"[INFO] TTFT: {ms:.4f} ms, min: {min_ms:.4f} ms, max: {max_ms:.4f} ms")
                            print(f"[INFO] Throughput: {(token_num / ms) * 1000.0:.4f} tokens/sec, min: {(token_num / max_ms) * 1000.0:.4f} tokens/sec, max: {(token_num / min_ms) * 1000.0:.4f} tokens/sec")
                        elif args.benchmark_metric == "tpot":
                            token_num = batch_size
                            ms, min_ms, max_ms = profiler.profile_tpot(
                                batch_size=batch_size,
                                seq_len=seq_len,
                                warmup=args.num_warmup,
                                repeat=args.num_repeat,
                                cuda_graph=args.cuda_graph,
                                sparsity=sparsity,
                            )
                            print(f"[INFO] TPOT: {ms:.4f} ms, min: {min_ms:.4f} ms, max: {max_ms:.4f} ms")
                            print(f"[INFO] Throughput: {(token_num / ms) * 1000.0:.4f} tokens/sec, min: {(token_num / max_ms) * 1000.0:.4f} tokens/sec, max: {(token_num / min_ms) * 1000.0:.4f} tokens/sec")
                    
                        res_dict[(batch_size, seq_len, f"{sparsity:.2f}", i)] = [(token_num / ms) * 1000.0, (token_num / max_ms) * 1000.0, (token_num / min_ms) * 1000.0]
                    except Exception as e:
                        print(f"[ERROR] {e}")
                        res_dict[(batch_size, seq_len, f"{sparsity:.2f}", i)] = [0, 0, 0]
        
    prefix = f'[INFO] Model: {args.model_name}, Config: {config_path}, {args.benchmark_metric} results:'
    record_results(res_dict, f"{profiler_name}.txt", prefix)

if __name__ == "__main__":
    main()