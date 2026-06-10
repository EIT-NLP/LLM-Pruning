from .eval_utils.custom_llama import CustomLlama





# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")  # 根据你的模型选择合适的tokenizer

# 创建自定义模型实例
custom_model = CustomLlama(
    model=modified_model,
    tokenizer=tokenizer,
    batch_size=1
)

# 选择要评估的任务
task_names = ["lambada", "piqa", "winogrande"]  # 可以选择其他任务

# 运行评估
results = evaluator.simple_evaluate(
    model="custom_llama_mod",
    model_args=f"model={custom_model},tokenizer_path=meta-llama/Llama-2-7b-hf",
    tasks=task_names,
    batch_size=1,
    device="cuda" if torch.cuda.is_available() else "cpu",
    no_cache=True
)

# 打印结果
print(evaluator.make_table(results))