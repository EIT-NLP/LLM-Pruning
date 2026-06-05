import torch
import torch.nn as nn

from typing import Optional, Tuple, Dict, List, Union, Any
from einops import rearrange

from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from torch.profiler import record_function

from ops import __ATTENTION__, __MLP__, __ROUTER__, __KV_CACHE__, __APPROXIMATOR__, __MASK__
from ops.utils import triton_rmsnorm, triton_rope_qk_align
from wrapper.base import PrunedModelForCausalLM
from wrapper.static.dense import DenseAttention, DenseDecoderLayer, DenseModel, DenseForCausalLM

#################### Layer ####################
class UnstructuredDecoderLayer(DenseDecoderLayer):
    """
    HF style decoder layer
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        layer_idx: int,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, layer_idx, **kwargs)
        self._support_pruning_components = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj']
        self.is_pruned = False

        # pruning impl
        for key in self._support_pruning_components:
            prefix, suffix = key.split('.')
            if suffix == '': setattr(self, f'{prefix}_kwargs', self.pruning_config.get(prefix, {}))
            else: setattr(self, f'{suffix}_kwargs', self.pruning_config[prefix].get(suffix, {}))
    
    def generate_pruning_kwargs(
        self,
        pruning_targets: List[str]=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj', 'mlp.up_proj', 'mlp.gate_proj', 'mlp.down_proj'],
        estimated_sparsity: Optional[float]=-1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        super().generate_pruning_kwargs(**kwargs)

        if self.is_pruned: return {}
        with record_function("**Initialization**"):
            pruning_kwargs = {}
            for name in pruning_targets:
                prefix, suffix = name.split('.')
                settings = getattr(self, f'{suffix}_kwargs')
                pruning_type = settings.get('pruning_type', 'base')
                block_size = settings.get('block_size', None)

                component: nn.Linear = getattr(getattr(self, prefix), suffix)

                mask = __MASK__[pruning_type].random_sample(
                    shape=component.weight.shape,
                    sparsity=estimated_sparsity if estimated_sparsity > 0 else settings.get('estimated_sparsity', 0),
                    device=component.weight.device,
                    block_size=block_size,
                )
                kwargs.update(settings)
                setattr(getattr(self, prefix), suffix, __MASK__[pruning_type].monkey_patch(component, mask, **kwargs))
                pruning_kwargs[name] = mask
        
                self.is_pruned = True

        return {}

class UnstructuredPretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["UnstructuredDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": UnstructuredDecoderLayer,
        "attentions": DenseAttention,
    }

class UnstructuredModel(UnstructuredPretrainedModel):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config)
        self.pruning_config = pruning_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = block.embed_tokens
        self.layers = nn.ModuleList([UnstructuredDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
        self.norm = block.norm
        self.rotary_emb = block.rotary_emb
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.LongTensor]=None,
        past_key_values: Optional[Cache]=None,
        inputs_embeds: Optional[torch.FloatTensor]=None,
        cache_position: Optional[torch.LongTensor]=None,
        use_cache: Optional[bool]=None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        if use_cache and past_key_values is None:
            past_key_values = __KV_CACHE__[self.pruning_config.get('cache_type', 'base')](self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        pad_offset = attention_mask.shape[1] - attention_mask.sum(-1)

        for i, decoder_layer in enumerate(self.layers[:self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                pad_offset=pad_offset,
                **kwargs,
            )
        
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )

class UnstructuredForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, **kwargs)
        self.model = UnstructuredModel(config, pruning_config, block.model, **kwargs)
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        pruning_kwargs = {}
        for i in range(self.config.num_hidden_layers):
            pruning_kwargs[f'layer_{i}'] = self.model.layers[i].generate_pruning_kwargs(**kwargs)
        
        return {}
    
    def post_load(
        self,
        router_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        lora_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        full_ckpt: Optional[Dict[str, torch.Tensor]]=None,
        lora_rank: Optional[int]=16,
        lora_alpha: Optional[float]=32,
        **kwargs,
    ):
        """
        SkipGPT-style post load:
        - First, we load all the router from ckpt
        - Then, we load all the lora adapter from ckpt
        """
        pass