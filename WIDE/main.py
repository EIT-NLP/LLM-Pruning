import os
import torch
import argparse
import regex as re

from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from lightning import seed_everything
from typing import List, Optional, Tuple

from wrapper.dispatch import apply_wrapper
from mainloop import mainloop_func

torch.set_float32_matmul_precision("high")


def parse_benchmark_batch_size(value: str):
    normalized = value.strip()
    if normalized.startswith("auto"):
        if normalized == "auto":
            return normalized
        prefix, _, suffix = normalized.partition(":")
        if prefix != "auto" or suffix == "":
            raise argparse.ArgumentTypeError(
                "benchmark_batch_size must be an integer, 'auto', or 'auto:N'."
            )
        try:
            float(suffix)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "benchmark_batch_size auto schedule must be numeric, e.g. 'auto:1.5'."
            ) from exc
        return normalized

    try:
        return int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "benchmark_batch_size must be an integer, 'auto', or 'auto:N'."
        ) from exc


def get_task_num_fewshot(task_name: str) -> Optional[int]:
    if task_name == "mmlu" or task_name.startswith("mmlu_"):
        return 5
    return None


def group_tasks_by_num_fewshot(tasks: List[str]) -> List[Tuple[Optional[int], List[str]]]:
    grouped_tasks = {}
    for task in tasks:
        num_fewshot = get_task_num_fewshot(task)
        grouped_tasks.setdefault(num_fewshot, []).append(task)
    return list(grouped_tasks.items())

def parse_args():
    parser = argparse.ArgumentParser(description="dynamic layer skipping.")
    
    parser.add_argument("--seed", type=int, default=17, help="Seed for reproducibility")
    parser.add_argument("--lora", action="store_true", help="Use LoRA")
    parser.add_argument("--sparsity", type=float, default=0.2, help="Sparsity of the routers")
    parser.add_argument("--sparsity_attention", type=float, default=-1, help="Sparsity of the attention")
    parser.add_argument("--sparsity_ffn", type=float, default=-1, help="Sparsity of the ffn")
    parser.add_argument("--model_name", type=str, default="Llama-2-13b", help="Name of the model to load") # 模型名称
    parser.add_argument("--model_path", type=str, default="Llama-2-13b", help="Path to the model to load") # 模型路径
    parser.add_argument("--project_name", type=str, default="SkipGPT-train", help="Name of the swanlab project")
    parser.add_argument("--save_dir", type=str, default="output", help="Path to save the model")
    parser.add_argument("--wrapper_type", type=str, default="attn_group_ffn_block", help="choosed method") # 动态层跳过的方法
    parser.add_argument("--train_dataset", type=str, default="datasets/RedPajama-Data-1T-Sample", help="Path to the dataset") # 数据集路径
    parser.add_argument("--eval_dataset", type=str, default="datasets/RedPajama-Data-1T-Sample", help="Path to the dataset") # 数据集路径
    parser.add_argument("--initial_temperature", type=float, default=5.0, help="Initial temperature for the gumbel softmax") # gumbe softmax的初始温度
    parser.add_argument("--final_temperature", type=float, default=1.0, help="Final temperature for the gumbel softmax") # gumbe softmax的最终温度
    parser.add_argument("--warmup_ratio", type=float, default=0, help="Warmup ratio for the temperature") # 温度的预热比例
    parser.add_argument("--max_steps", type=int, default=10000, help="Maximum training steps") # 最大训练步数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2, help="Evaluation batch size")
    parser.add_argument("--train_data_mode", type=str, default="sample", choices=['sample', 'concat'], help="Data mode for training") # 训练数据切分模式
    parser.add_argument("--max_lr", type=float, default=2e-3, help="Max learning rate")
    parser.add_argument("--min_lr", type=float, default=0, help="Min learning rate") # 最小学习率
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation steps") # 评估步数
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps") # 梯度累积步数
    parser.add_argument("--eval_dataset_ratio", type=float, default=0.001, help="Ratio of the dataset to use for evaluation") # 用于评估的数据集比例
    parser.add_argument("--max_length", type=int, default=4096, help="the max context length") # 最大上下文长度
    parser.add_argument("--sparse_target", type=str, default="all", choices=['all', 'attention', 'ffn']) # 训练方法
    parser.add_argument("--attn_group_size", type=int, default=1, help="Number of query heads controlled by one attention router group")
    parser.add_argument("--ffn_group_size", type=int, default=32, help="Number of intermediate channels controlled by one FFN router group")
    parser.add_argument("--route_rank_size", type=int, default=16)
    parser.add_argument("--tag", type=str, default="", help="Tag for the run")
    parser.add_argument("--router_ckpt_path", type=str, default="")
    parser.add_argument("--lora_ckpt_path", type=str, default="")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing") # 梯度检查点
    parser.add_argument("--profiling", action="store_true", help="Use PyTorch Profiler to record performance metrics") # 使用PyTorch Profiler记录性能指标
    parser.add_argument("--cpu_offload", action="store_true", help="Use CPU offload") # 使用CPU卸载
    parser.add_argument("--freeze_router", action="store_true", help="Freeze the router in lora turning")

    # lm_eval config
    parser.add_argument("--benchmark", action="store_true", help="Train or evaluate the model") # 训练或评估模型
    parser.add_argument("--benchmark_tasks", type=str, nargs="+", default=["wikitext"])
    parser.add_argument(
        "--benchmark_batch_size",
        type=parse_benchmark_batch_size,
        default='auto',
        help="lm-eval batch size. Supports integer, 'auto', or 'auto:N'.",
    )
    parser.add_argument(
        "--benchmark_max_batch_size",
        type=int,
        default=None,
        help="Upper bound used by lm-eval when benchmark_batch_size is auto.",
    )
    parser.add_argument("--benchmark_max_length", type=int, default=-1)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    seed_everything(args.seed)

    # model import
    if not args.benchmark:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.use_cache = False

        # apply liger wrapper
        from liger_kernel.transformers import apply_liger_kernel_to_llama, apply_liger_kernel_to_qwen3

        apply_liger_kernel_to_llama(rope=False, swiglu=False)
        apply_liger_kernel_to_qwen3(rope=False, swiglu=False)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        config.use_cache = True

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if args.lora_ckpt_path != "":
        if os.path.exists(args.lora_ckpt_path):
            lora_dict = torch.load(args.lora_ckpt_path, map_location='cpu', weights_only=True)
            new_lora = {}
            for k, v in lora_dict.items():
                name = re.findall(r'[\w\W]*?(model.layers.[\d]+.[\w\W]+)', k)[-1]
                component = re.findall(r'[\w\W]*?(lora_A|lora_B)', name)[-1]
                prefix = name.split('.lora')[0]
                if prefix not in new_lora:
                    new_lora[prefix] = {}
                new_lora[prefix][component] = v
                print(f"|---> Sucess load lora: {name}")

            param_dict = {}
            for k, v in model.named_parameters(): param_dict[k] = v
            for k, v in new_lora.items():
                if k + '.weight' in param_dict:
                    lora_weight = ((v['lora_B'].float() @ v['lora_A'].float()) * 2).to(v['lora_A'].dtype) # alpha / r
                    param_dict[k + '.weight'].data.add_(lora_weight)
        else:
            print(f"|---> Failed to load lora: {args.lora_ckpt_path}")

    model = apply_wrapper(model, config, args)

    if args.router_ckpt_path != "":
        if os.path.exists(args.router_ckpt_path):
            router_dict = torch.load(args.router_ckpt_path, map_location='cpu', weights_only=True)
            new_router = {}
            for k, v in router_dict.items():
                name = re.findall(r'[\w\W]*?(model.layers.[\d]+.[\w\W]+)', k)[-1]
                new_router[name] = v
                print(f"|---> Sucess load router: {name}")
            try:
                model.load_state_dict(new_router, strict=True)
            except Exception as e:
                # print(e)
                print("Use flexible loading...")
                model.load_state_dict(new_router, strict=False)
        else:
            print(f"|---> Failed to load router: {args.router_ckpt_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Maximum context size: {args.max_length}")

    if args.benchmark:
        import time
        import swanlab
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        swanlab.init(
            project=args.project_name,
            experiment_name=f"{args.model_name.replace('/', '_')}-{args.wrapper_type}-{args.sparse_target}-{args.sparsity}-A{args.attn_group_size}-F{args.ffn_group_size}-tag:{args.tag}",
            config=vars(args),
        )

        benchmark_max_length = None
        if args.benchmark_max_length > 0: benchmark_max_length = args.benchmark_max_length

        model = model.eval()
        model = model.to(device='cuda', dtype=torch.bfloat16)
        llm = HFLM(
            model,
            tokenizer=tokenizer,
            batch_size=args.benchmark_batch_size,
            max_batch_size=args.benchmark_max_batch_size,
            max_length=benchmark_max_length,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        tasks = args.benchmark_tasks
        total_duration = 0.0

        for num_fewshot, grouped_tasks in group_tasks_by_num_fewshot(tasks):

            torch.cuda.synchronize()
            start = time.perf_counter()
            results = simple_evaluate(
                llm,
                tasks=grouped_tasks,
                num_fewshot=num_fewshot,
            )
            torch.cuda.synchronize()
            duration = time.perf_counter() - start
            total_duration += duration

            swanlab_results = {}
            for k, v in results['results'].items():
                if isinstance(v, dict):
                    if 'alias' in v: v.pop('alias')
                    for kk, vv in v.items():
                        if not isinstance(vv, str):
                            metric, _, _ = kk.partition(",")
                            swanlab_results[f"{k}:{metric}"] = vv

            swanlab.log(swanlab_results)
            swanlab.log({
                f"duration/{'+'.join(grouped_tasks)}": duration,
            })

            if results is not None:
                res = make_table(results)

                print(f"Tasks: {', '.join(grouped_tasks)}")
                print(f"Few-shot: {num_fewshot}")
                print(res)
                print(f"Duration: {duration:.4f}s")

        swanlab.log({"duration_total": total_duration})
        print(f"Total benchmark duration: {total_duration:.4f}s")
        
        swanlab.finish()

    else:
        mainloop_func(
            model=model,
            tokenizer=tokenizer,
            config=config,
            args=args,
        )

if __name__ == "__main__":
    main()
