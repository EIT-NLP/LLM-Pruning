import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional
from argparse import Namespace
from typing import Optional, Tuple, Any
from transformers import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask
from transformers import PretrainedConfig
from einops import rearrange

from .base_layer import TokenRouter, apply_rotary_pos_emb, apply_mask, get_group_num
from .base_layer import WrapperMLP as BaseWrapperMLP
from .base_layer import WrapperAttention as BaseWrapperAttention
from .base_layer import WrapperDecoderLayer as BaseWrapperDecoderLayer
from .base_layer import WrapperModel as BaseWrapperModel

_use_top_left_mask = flash_attn_supports_top_left_mask()

def ensure_group_dim(mask: torch.Tensor) -> torch.Tensor:
    return mask[..., None] if mask.dim() == 2 else mask

class WrapperMLP(BaseWrapperMLP):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args: Namespace):
        super().__init__(block=block, config=config, args=args)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        execute_ffn: Optional[torch.Tensor]=None,
    ):
        hidden_states = self.swiglu_func.apply(self.gate_proj(hidden_states), self.up_proj(hidden_states))
        # equal to apply N-axis pruning for glu & apply K-axis pruning for down_proj
        if execute_ffn is not None:
            hidden_states = apply_mask(hidden_states, execute_ffn, self.ffn_group_size)

        hidden_states = self.down_proj(hidden_states)
        return hidden_states

class WrapperAttention(BaseWrapperAttention):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args: Namespace):
        super().__init__(block=block, config=config, args=args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        execute_attn: Optional[torch.Tensor]=None,
        past_key_values: Optional[Cache]=None,
        cache_position: Optional[torch.LongTensor]=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states, key_states, value_states = list(map(lambda x: rearrange(x, '... (h d) -> ... h d', d=self.head_dim), [query_states, key_states, value_states]))

        if self.q_norm is not None:
            query_states = self.q_norm(query_states)
        if self.k_norm is not None:
            key_states = self.k_norm(key_states)
        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length=query_states.shape[1],
            is_causal=self.is_causal,
            dropout=0.0 if not self.training else self.attention_dropout,
            softmax_scale=self.scaling,
            use_top_left_mask=_use_top_left_mask,
            attn_implementation='flash_attention_2',
            layer_idx=self.layer_idx,
            **kwargs,
        )
        if execute_attn is not None:
            head_mask = execute_attn.repeat_interleave(self.attn_group_size, dim=-1)
            if head_mask.shape[-1] != attn_output.shape[-2]:
                raise RuntimeError(
                    f"Attention mask expands to {head_mask.shape[-1]} heads, "
                    f"but attention output has {attn_output.shape[-2]} heads"
                )
            attn_output = attn_output * head_mask[..., None]
        
        attn_output = attn_output.flatten(-2, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

class WrapperDecoderLayer(BaseWrapperDecoderLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        block: nn.Module,
        args: Namespace,
    ):
        super().__init__(config=config, block=block, args=args)
        self._router_names = ['attention', 'ffn']

        self.group_nums = {
            'attention': get_group_num(
                self.num_heads_q,
                self.attn_group_size,
                min_group_size=1,
                name='attention',
            ),
            'ffn': get_group_num(
                self.intermediate_size,
                self.ffn_group_size,
                min_group_size=1,
                name='ffn',
            ),
        }

        self.self_attn = WrapperAttention(block.self_attn, config, args)
        self.mlp = WrapperMLP(block.mlp, config, args)

        self.router_attention = None
        self.router_ffn = None
        if self.sparse_target in ['all', 'attention']:
            self.router_attention = TokenRouter(
                self.hidden_size,
                group_num=self.group_nums['attention'],
                rank_size=self.rank_size,
            )

        if self.sparse_target in ['all', 'ffn']:
            self.router_ffn = TokenRouter(
                self.hidden_size,
                group_num=self.group_nums['ffn'],
                rank_size=self.rank_size,
            )
        
        # freeze the parameters of the block
        for param in self.parameters():
            param.requires_grad = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        b, s, _ = hidden_states.shape

        if self.training and (self.args.initial_temperature != self.args.final_temperature):
            if self.training_step <  self.args.gradient_accumulation_steps * self.args.max_steps:
                self.training_step += 1
            temperature = self.args.initial_temperature - (self.args.initial_temperature - self.args.final_temperature) * ((self.training_step - 1) // self.args.gradient_accumulation_steps) / (self.args.max_steps)
        else:
            temperature = self.args.final_temperature
        
        skipped_tokens = kwargs.get('skipped_tokens', None)
        router_zero_prob = kwargs.get('router_zero_prob', None)
        valid_token_num = attention_mask.sum()
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.router_attention is not None:
            weights_attn = self.router_attention(residual)
            execute_attn, skip_attn = self._gumbel_softmax(weights_attn, tau=temperature)
            execute_attn = ensure_group_dim(execute_attn)
            skip_attn = ensure_group_dim(skip_attn)
            selected_mask_attn = skip_attn * attention_mask[..., None]
            if skipped_tokens is not None: self._update_skipped_tokens('attention', valid_token_num, selected_mask_attn, skipped_tokens, router_zero_prob)
        else:
            execute_attn = None
            skip_attn = None
        
        # attention in FA2
        hidden_states = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            use_cache=use_cache,
            attention_mask=attention_mask,
            execute_attn=execute_attn,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        if self.router_ffn is not None:
            weights_ffn = self.router_ffn(residual)
            execute_ffn, skip_ffn = self._gumbel_softmax(weights_ffn, tau=temperature)
            execute_ffn = ensure_group_dim(execute_ffn)
            skip_ffn = ensure_group_dim(skip_ffn)
            selected_mask_ffn = skip_ffn * attention_mask[..., None]
            if skipped_tokens is not None: self._update_skipped_tokens('ffn', valid_token_num, selected_mask_ffn, skipped_tokens, router_zero_prob)
        else:
            execute_ffn = None
            skip_ffn = None
        
        hidden_states = self.mlp(
            hidden_states,
            execute_ffn=execute_ffn,
        )
        hidden_states = hidden_states + residual
        return hidden_states, skipped_tokens, router_zero_prob

class WrapperModel(BaseWrapperModel):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args):
        super().__init__(block=block, config=config, args=args)
        self.layers = nn.ModuleList(
            [WrapperDecoderLayer(config, block.layers[layer_idx], args) for layer_idx in range(config.num_hidden_layers)]
        )

        self._router_names = self.layers[0]._router_names
        self.group_nums = self.layers[0].group_nums

        # freeze the parameters of the block
        for param in self.parameters():
            param.requires_grad = False
    
    def compute_sparsity(self) -> dict[str, torch.Tensor]:
        sparsity = {}
        for name, skipped in self.skipped_tokens.items():
            if name in self.group_nums:
                sparsity[name] = (
                    skipped
                    / self.config.num_hidden_layers
                    / (self.total_tokens * self.group_nums[name])
                    if self.total_tokens > 0
                    else skipped.new_zeros(())
                )

        if sparsity:
            sparsity['total'] = sum(sparsity.values()) / len(sparsity)
        
        for k in sparsity: sparsity[k] = sparsity[k].item()
        self.skipped_tokens = {}
        self.total_tokens = 0
        return sparsity
