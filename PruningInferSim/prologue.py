import os
import torch
import torch.nn as nn

from typing import Optional, Dict, List
from argparse import Namespace
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from config import load_config_and_class

def patch_estimated_sparsity(node, target: float, sentinel: float = -1) -> int:
    changed = 0
    if isinstance(node, Dict):
        for k, v in node.items():
            if k == "estimated_sparsity" and isinstance(v, (int, float)) and float(v) == sentinel:
                node[k] = target
                changed += 1
            else:
                changed += patch_estimated_sparsity(v, target, sentinel)
    elif isinstance(node, List):
        for item in node:
            changed += patch_estimated_sparsity(item, target, sentinel)
    return changed


def init_model(args: Namespace) -> tuple:
    """
    Initialize model wrapper and tokenizer based on arguments
    
    Args:
        args: Command line arguments containing 'model_path' key
        
    Returns:
        Dict containing 'model' and 'tokenizer'
    """
    model_path = getattr(args, "model_path", None)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    return model, tokenizer, config

def wrap_model(model, config, args: Namespace, sparsity: Optional[float]=-1):
    # apply wrapper
    wrapper_config, wrapper_cls, config_path = load_config_and_class(args)
    # set benchmark sparsity
    if sparsity >= 0:
        patch_estimated_sparsity(wrapper_config, sparsity)

    model = wrapper_cls(
        config=config,
        pruning_config=wrapper_config,
        block=model,
    )

    model.post_load()
    
    return model, config_path