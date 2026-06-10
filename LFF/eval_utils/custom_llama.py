import numpy as np
import torch
from lm_eval.api.model import TemplateLM as LM
from lm_eval.models.huggingface import HFLM
from lm_eval import utils
from transformers import AutoTokenizer
from lm_eval.api.registry import register_model


@register_model('custom_lm')
class CustomLM(HFLM):
    def __init__(
        self,
        pretrained,
        tokenizer=None,
        subfolder=None,
        revision="main",
        **kwargs,
    ):

        super().__init__(
            pretrained=pretrained,
            tokenizer=tokenizer,
            **kwargs,
        )
# from lm_eval import tasks, evaluator, models
# # 注册自定义模型
# models.registry["custom_llama"] = CustomLlama