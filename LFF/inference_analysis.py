from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import argparse
import numpy as np

def main(args):
    # 设置设备
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型和分词器
    print(f"加载模型: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_mode,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print("模型加载完成")

    # 测试的序列长度列表（prefill阶段的输入长度）
    seq_lengths = [1024, 2048, 4096]
    # 每个长度的推理次数（包含热身）
    warmup_runs = 2
    test_runs = 10

    # 存储结果的字典
    results = {length: [] for length in seq_lengths}

    for seq_len in seq_lengths:
        print(f"\n开始测试序列长度: {seq_len}")
        
        # 生成固定长度的输入（batch_size=1）
        input_ids = torch.full(
            (1, seq_len),  # 形状: [batch_size, seq_len]
            fill_value=tokenizer.eos_token_id,
            dtype=torch.long,
            device=device
        )

        # 热身推理（不计入结果，排除首次编译耗时）
        print(f"进行 {warmup_runs} 次热身推理...")
        with torch.no_grad():
            for _ in range(warmup_runs):
                outputs = model(input_ids=input_ids)  # 直接调用模型进行prefill
        torch.cuda.synchronize()  # 确保GPU操作完成

        # 正式测试prefill阶段耗时
        print(f"进行 {test_runs} 次正式推理...")
        with torch.no_grad():
            for i in range(test_runs):
                # 同步后开始计时，确保测量的是纯推理时间
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # 直接调用模型，仅进行prefill阶段计算
                outputs = model(input_ids=input_ids)
                
                # 同步后结束计时，确保推理完成
                torch.cuda.synchronize()
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                results[seq_len].append(elapsed)
                print(f"测试 {i+1}/{test_runs} 完成，耗时: {elapsed:.4f}秒")

    # 输出统计结果（重点关注prefill阶段的速度）
    print("\n===== Prefill阶段速度测试结果 =====")
    for seq_len in seq_lengths:
        times = results[seq_len]
        avg_time = np.mean(times)
        std_time = np.std(times)
        tokens_per_sec = (seq_len * test_runs) / sum(times)  # 总token数/总时间
        print(f"序列长度: {seq_len}")
        print(f"  平均耗时: {avg_time:.4f}秒 ± {std_time:.4f}秒")
        print(f"  Prefill速度: {tokens_per_sec:.2f} tokens/秒")
        print(f"  单次耗时范围: {min(times):.4f} - {max(times):.4f}秒")
        print("----------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data/chao_han/model/Meta-Llama-3_1-8B")
    parser.add_argument("--gpu_id", type=int, default=1, help="GPU设备ID")
    parser.add_argument("--attn_mode", type=str, default='eager', help="GPU设备ID") # flash_attention_2, eager，sdpa
    args = parser.parse_args()
    main(args)


