import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from transformers import PreTrainedModel, PretrainedConfig

from .dense import WrapperModel as DenseModel
from .attn_group_ffn_block import WrapperModel as AttnGroupFFNBlockModel
from .attn_token_ffn_token import WrapperModel as AttnTokenFFNTokenModel
from .layer_skip import WrapperModel as LayerSkipModel

WRAPPER = {
    'attn_group_ffn_block': AttnGroupFFNBlockModel,
    'attn_token_ffn_token': AttnTokenFFNTokenModel,
    'layer_skip': LayerSkipModel,
    'dense': DenseModel,
}

def apply_wrapper(
    model: PreTrainedModel,
    config: PretrainedConfig,
    args: Namespace,
):
    wrapper_type = args.wrapper_type
    model.model = WRAPPER[wrapper_type](
        block=model.model,
        config=config,
        args=args,
    )

    raw_class_name = model.__class__.__name__
    if "ForCausalLM" in raw_class_name:
        model.__class__.__name__ = raw_class_name.replace("ForCausalLM", "WrapperForCausalLM")
    
    return model