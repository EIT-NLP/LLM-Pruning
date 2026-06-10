from eval_utils.custom_llama import CustomLM
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import lm_eval


def parse_args():
    parser = argparse.ArgumentParser(description="eval")
    # parser.add_argument("--tokenizer_path", type=str, default="/data/chao_han/model/Meta-Llama-3_1-8B")
    parser.add_argument("--tokenizer_path", type=str, default="/data/chao_han/model/Llama-3.2-3b")
    # parser.add_argument("--model_path", type=str, default="/data/chao_han/code/skipgpt/skipgpt/saved_models/llama3_8b_router_lora.pt") # 注意在保存时需要完整保存模型
    parser.add_argument("--model_path", type=str, default="/data/chao_han/code/skipgpt/skipgpt/saved_models/llama3_8b_router.pt")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)  # 根据你的模型选择合适的tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    model = torch.load(args.model_path, weights_only=False).to('cuda').to(torch.bfloat16)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.tokenizer_path,
    #     device_map="auto",  # 自动分配设备（CPU/GPU）
    #     torch_dtype="auto",  # 自动选择数据类型（推荐GPU时使用float16/bfloat16）
    #     trust_remote_code=True  # 必要时启用，加载模型定义代码
    # )

    # 创建自定义模型实例
    custom_model = CustomLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=1
    )

    ''' fast eval '''
    results = lm_eval.evaluator.simple_evaluate(
            model=custom_model,
            tasks=['wikitext'],
            num_fewshot=0,  # 设置为0表示不使用fewshot
            batch_size='auto:1',
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    print(lm_eval.utils.make_table(results))
    exit()
    # fast eval 

    
    task_0 = ["wikitext", "piqa", "boolq", "openbookqa"]
    task_5 = ["winogrande"]
    task_10 = ["hellaswag"]
    task_25 = ["arc_challenge", "arc_easy"]

    task = {}
    task["0"] = task_0
    task["5"] = task_5
    task["10"] = task_10
    task["25"] = task_25

    # 运行评估
    res = []
    for i in [0,5,10,25]:
        results = lm_eval.evaluator.simple_evaluate(
            model=custom_model,
            tasks=task[str(i)],
            num_fewshot=i,  # 设置为0表示不使用fewshot
            batch_size='auto:1',
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
        res.append(results)


    # 打印结果
    for i in range(len(res)):
        print(lm_eval.utils.make_table(res[i]))


main()