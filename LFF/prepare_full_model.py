from transformers import AutoModelForCausalLM, AutoConfig
import torch
from router_attn_mlp import apply_router_attn_mlp


# modified these 

model_path = "/data/chao_han/model/Meta-Llama-3_1-8B"
lw_path = None
router_path = None
name = None

device = 'cpu'
orignal_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                ).to(device)

# torch.save(orignal_model, "saved_models/llama3_8b.pt")

# 加载自定义模型
from main import parse_args
default_args = parse_args()
default_args.post_norm_router = True
model = apply_router_attn_mlp(orignal_model, default_args).to(device)

# 加载权重
lw_weight = torch.load("./checkpoint_lw/checkpoint-2000/model.pth")
missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
print(f"load lw net unexpected keys: {unexpected_keys}")
router_weight = torch.load("./checkpoint_router/checkpoint-2000/model.pth")
missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
print(f"load router net unexpected keys: {unexpected_keys}")

torch.save(model, "saved_models/llama3_8b_router.pt")

def lora(model):
    lora_config = LoraConfig(
        r=16,  # 低秩矩阵的秩
        lora_alpha=32,
        target_modules=["q_proj", "v_proj","gate_proj"],  # 指定应用 LoRA 的模块
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    return model
from peft import get_peft_model, LoraConfig

lora_model = lora(model)
lora_weight = torch.load("./checkpoint_lora/checkpoint-lora-2000/model.pth")
missing_keys, unexpected_keys = model.load_state_dict(lora_weight, strict=False)
print(f"load lora net unexpected keys: {unexpected_keys}")
torch.save(model, "saved_models/llama3_8b_router_lora.pt")





