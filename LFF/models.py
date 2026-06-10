from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from router_attn_mlp import apply_router_attn_mlp
# from dllm import apply_router_attn_mlp
import torch
def model_import(device,args):
    if args.eval:
        if args.finetuned:
            orignal_model=torch.load(args.model_path).to(device)    
            tokenizer= AutoTokenizer.from_pretrained(args.model_name)
            tokenizer.pad_token = tokenizer.eos_token
        else:
            # orignal_model = AutoModelForCausalLM.from_pretrained(
            #     args.model_path,
            #     torch_dtype=torch.bfloat16,
            #     attn_implementation="flash_attention_2",
            #     ).to(device)

            orignal_model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                ).to(device)
            tokenizer= AutoTokenizer.from_pretrained(args.model_path)
            tokenizer.pad_token = tokenizer.eos_token
        return  orignal_model, tokenizer
    elif args.eval_stage2:
        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                )
        
        model = apply_router_attn_mlp(model, args).to(device)
        lw_weight = torch.load("/home/user/chao_han/code/skipgpt/checkpoint_lw/checkpoint-2000/model.pth")
        missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
        print(f"load lw net unexpected keys: {unexpected_keys}")

        router_weight = torch.load("/home/user/chao_han/code/skipgpt/checkpoint_router/checkpoint-2000/model.pth")
        missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
        print(f"load router net unexpected keys: {unexpected_keys}")

        tokenizer= AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    else:
        model = AutoModelForCausalLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                )
        
        if args.method == "router_attn_mlp":
            model = apply_router_attn_mlp(model, args).to(device).to(torch.bfloat16)
            if args.train_lw_only:
                if args.init_router:
                    print('init router, load pre-trained lw weights')
                    lw_weight = torch.load("./checkpoint_lw/checkpoint-2000/model.pth")
                    missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                    print(f"load lw net unexpected keys: {unexpected_keys}")
                else:
                    pass
                # lw_weight = torch.load("/home/user/chao_han/code/skipgpt/checkpoint_lw/checkpoint-2000/model.pth")
                # missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                # print(f"load lw net unexpected keys: {unexpected_keys}")
                # router_weight = torch.load("/home/user/chao_han/code/skipgpt/checkpoint_router/checkpoint-2000/model.pth")
                # missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
                # print(f"load router net unexpected keys: {unexpected_keys}")

            if args.retrain_lw:
                lw_weight = torch.load("./checkpoint_lw/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                print(f"load lw net unexpected keys: {unexpected_keys}")
                router_weight = torch.load("./checkpoint_router/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
                print(f"load router net unexpected keys: {unexpected_keys}")
            
            if args.retrain_router:
                lw_weight = torch.load("./checkpoint_retrain_lw/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                print(f"load lw net unexpected keys: {unexpected_keys}")
                router_weight = torch.load("./checkpoint_router/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
                print(f"load router net unexpected keys: {unexpected_keys}")


            if args.train_router_only:
                # pass
                lw_weight = torch.load("./checkpoint_lw/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                print(f"load lw net unexpected keys: {unexpected_keys}")

                # init router weight
                # router_weight = torch.load("./checkpoint_lw/checkpoint-2000/init_router_model.pth")
                # missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
                # print(f"load init router net unexpected keys: {unexpected_keys}")
                
            if args.train_lw_and_router or args.lora_hc:
                lw_weight = torch.load("./checkpoint_lw/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(lw_weight, strict=False)
                print(f"load lw net unexpected keys: {unexpected_keys}")
                router_weight = torch.load("./checkpoint_router/checkpoint-2000/model.pth")
                missing_keys, unexpected_keys = model.load_state_dict(router_weight, strict=False)
                print(f"load router net unexpected keys: {unexpected_keys}")
            
        elif args.method == "router_all":
            model = apply_router_all(model, args).to(device)

        elif args.method == "mod_twice":
            model = apply_mod_twice(model, args).to(device)

        elif args.method == "mod":
            model = apply_mod(model, args).to(device)

        if args.method == "joint":
            model = apply_router_attn_mlp(model, args).to(device)

        elif args.method == "post_training":
            model = torch.load("/code/anhao_zhao/mod/Llama-2-7b_router_attn_mlp_0_loraFalse/checkpoint-3200/model.pth").to(device)

        elif args.method == "post_training_deepspeed":
            model = apply_router_attn_mlp(model, args).to(device)
            model.load_state_dict(torch.load("/code/anhao_zhao/mod/Llama-2-7b_router_attn_mlp_0_loraFalse/checkpoint-3200/model.pth"), strict=False)

        tokenizer= AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.eos_token
    
        return model, tokenizer