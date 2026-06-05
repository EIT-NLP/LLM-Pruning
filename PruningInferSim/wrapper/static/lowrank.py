import random
import torch
import torch.nn as nn
import nvtx

from typing import Optional, Tuple, Dict, List, Union, Any, Sequence
from einops import rearrange

from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerateOutput
from torch.profiler import record_function

from ops import __ATTENTION__, __MLP__, __ROUTER__, __KV_CACHE__, __APPROXIMATOR__, __LOW_RANK__
from ops.utils import triton_rmsnorm, triton_rope_qk_align
from wrapper.base import PrunedModelForCausalLM
from wrapper.static.dense import DenseAttention, DenseMLP, DenseDecoderLayer, DenseModel, DenseForCausalLM

def generate_sparsity(pruning_config: Dict[str, Any]) -> float:
    estimated_sparsity = pruning_config.get("estimated_sparsity", 0.0)
    offset = pruning_config.get("offset", 0.0)
    sampled_offset = random.uniform(-offset, offset)
    return estimated_sparsity + sampled_offset

#################### FFN ####################
class LowRankMLP(DenseMLP):
    """
    HF style FFN
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        post_attention_layernorm: nn.Module,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, post_attention_layernorm, **kwargs)
        self.mlp_impl = __MLP__['low_rank']()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        with nvtx.annotate("ffn", color='green'):
            with record_function("ffn"):
                if self.up_proj.weight.numel() > 0:
                    ffn_output = self.mlp_impl(
                        hidden_states,
                        w_up_r1=self.up_proj_r1.weight,
                        w_up_r2=self.up_proj_r2.weight,
                        w_up=self.up_proj.weight,
                        w_gate=self.gate_proj.weight,
                        w_down=self.down_proj.weight,
                        b_up=self.up_proj.bias,
                        b_gate=self.gate_proj.bias,
                        b_down=self.down_proj.bias,
                        w_gate_r1=self.gate_proj_r1.weight,
                        w_gate_r2=self.gate_proj_r2.weight,
                        w_down_r1=self.down_proj_r1.weight,
                        w_down_r2=self.down_proj_r2.weight,
                        activation=self.activation,
                    )
                else:
                    up_output = self.mlp_impl(
                        hidden_states,
                        w_up_r1=self.up_proj_r1.weight,
                        w_up_r2=self.up_proj_r2.weight,
                        w_up=self.up_proj.weight,
                        b_up=self.up_proj.bias,
                    )
                    gate_output = self.mlp_impl(
                        hidden_states,
                        w_up_r1=self.gate_proj_r1.weight,
                        w_up_r2=self.gate_proj_r2.weight,
                        w_up=self.gate_proj.weight,
                        b_up=self.gate_proj.bias,
                    )
                    ffn_output = up_output * self.act_fn(gate_output)
                    ffn_output = self.mlp_impl(
                        ffn_output,
                        w_up_r1=self.down_proj_r1.weight,
                        w_up_r2=self.down_proj_r2.weight,
                        w_up=self.down_proj.weight,
                        b_up=self.down_proj.bias,
                    )
        return ffn_output + residual

#################### ATTN ####################
class LowRankAttention(DenseAttention):
    """
    HF style attention with FA2
    """
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: nn.Module,
        input_layernorm: nn.Module,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, input_layernorm, **kwargs)
        self.mlp_impl = __MLP__['low_rank']()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor]=None,
        attention_mask: Optional[torch.Tensor]=None,
        past_key_values: Optional[Cache]=None,
        cache_position: Optional[torch.LongTensor]=None,
        pad_offset: Optional[torch.Tensor]=None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        with record_function("qkv_proj"):
            with nvtx.annotate("q_proj", color='blue'):
                q = self.mlp_impl(
                    hidden_states,
                    w_up_r1=self.q_proj_r1.weight,
                    w_up_r2=self.q_proj_r2.weight,
                    w_up=self.q_proj.weight,
                    b_up=self.q_proj.bias,
                )
            with nvtx.annotate("k_proj", color='blue'):
                k = self.mlp_impl(
                    hidden_states,
                    w_up_r1=self.k_proj_r1.weight,
                    w_up_r2=self.k_proj_r2.weight,
                    w_up=self.k_proj.weight,
                    b_up=self.k_proj.bias,
                )
            with nvtx.annotate("v_proj", color='blue'):
                v = self.mlp_impl(
                    hidden_states,
                    w_up_r1=self.v_proj_r1.weight,
                    w_up_r2=self.v_proj_r2.weight,
                    w_up=self.v_proj.weight,
                    b_up=self.v_proj.bias,
                )

        q, k, v = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [q, k, v]))
        if self.q_norm: q = self.q_norm(q)
        if self.k_norm: k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = triton_rope_qk_align(q, k, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
                "inplace_update_kvcache": kwargs.get('inplace_update_kvcache', False),
            }
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        
        input_kwargs = dict(
            q=q, k=k, v=v,
            attention_mask=attention_mask,
            pad_offset=pad_offset,
            execute_block=getattr(self, 'execute_block', None),
            prefill_impl=self.attention_kwargs.get('pruning_type'),
            decode_impl=self.attention_kwargs.get('pruning_type'),
            backend=self.attention_kwargs.get('backend', 'triton'),
        )
        input_kwargs.update(self.threshold_impl.get_threshold_kwargs())
        
        with nvtx.annotate("attention", color='blue'):
            with record_function("attention"):
                attn_output = self.attention_impl(**input_kwargs)
        attn_output = rearrange(attn_output, '... h d -> ... (h d)')
        with nvtx.annotate("o_proj", color='blue'):
            with record_function("o_proj"):
                attn_output = self.mlp_impl(
                    attn_output,
                    w_up_r1=self.o_proj_r1.weight,
                    w_up_r2=self.o_proj_r2.weight,
                    w_up=self.o_proj.weight,
                    b_up=self.o_proj.bias,
                )

        return attn_output + residual

#################### Layer ####################
class LowRankDecoderLayer(DenseDecoderLayer):
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

        self.self_attn = LowRankAttention(config, pruning_config, block.self_attn, block.input_layernorm, **kwargs)
        self.mlp = LowRankMLP(config, pruning_config, block.mlp, block.post_attention_layernorm, **kwargs)

class LowRankPretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["LowRankDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": LowRankDecoderLayer,
        "attentions": LowRankAttention,
    }

class LowRankModel(LowRankPretrainedModel):
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
        self.layers = nn.ModuleList([LowRankDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
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

class LowRankForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, **kwargs)
        self.model = LowRankModel(config, pruning_config, block.model, **kwargs)

        self.is_pruned = False
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        for layer in self.model.layers:
            layer.generate_pruning_kwargs(**kwargs)
            
        if self.is_pruned: return {}

        attention_pruning_config = self.pruning_config.get('self_attn', {})
        ffn_pruning_config = self.pruning_config.get('mlp', {})
        retained_ratio = ffn_pruning_config.get('retained_ratio', 0)

        all_module_names = [name for name, module in self.named_modules()]

        for name in all_module_names:
            if name.endswith(('q_proj', 'k_proj', 'v_proj', 'o_proj')):
                component = name.split('.')[-1]
                low_rank_type = attention_pruning_config.get(component).get('pruning_type', 'base')

                __LOW_RANK__[low_rank_type].monkey_patch(
                    self,
                    name,
                    sparsity=generate_sparsity(attention_pruning_config.get(component)),
                    retained_ratio=0,
                    device=self.device,
                )
            
            elif name.endswith(('up_proj', 'gate_proj', 'down_proj')):
                component = name.split('.')[-1]
                low_rank_type = ffn_pruning_config.get(component).get('pruning_type', 'base')

                __LOW_RANK__[low_rank_type].monkey_patch(
                    self,
                    name,
                    sparsity=generate_sparsity(ffn_pruning_config.get(component)),
                    retained_ratio=retained_ratio,
                    device=self.device,
                )
        
        self.is_pruned = True
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