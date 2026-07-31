import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List, Optional
from argparse import Namespace
from typing import Optional, Tuple, Any
from transformers import PreTrainedModel, DynamicCache, Cache
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_layers import GradientCheckpointingLayer
from peft.tuners.lora import Linear as LoraLinear
from einops import rearrange
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from .cache import DynamicCacheSeqFirst

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

@torch.compile
def apply_rotary_pos_emb(q, k, cos, sin):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor with size [batch_size, seq_len, num_heads, head_dim].
        k (`torch.Tensor`): The key tensor with size [batch_size, seq_len, num_heads, head_dim].
        cos (`torch.Tensor`): The cosine part of the rotary embedding, with size [1, seq_len, head_dim].
        sin (`torch.Tensor`): The sine part of the rotary embedding, with size [1, seq_len, head_dim].
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_mask(hidden_states: torch.Tensor, mask: torch.Tensor, group_size: int):
    feature_size = hidden_states.shape[-1]
    if feature_size % group_size:
        raise ValueError(
            f"group_size={group_size} must divide feature size {feature_size}"
        )

    # Token-level routers produce [B, S], while grouped routers produce
    # [B, S, num_groups]. Keep both callers supported by the same helper.
    if mask.dim() == hidden_states.dim() - 1:
        if group_size != feature_size:
            raise ValueError(
                "A token-level mask can only control the complete feature axis"
            )
        return hidden_states * mask[..., None]

    group_num = feature_size // group_size
    if mask.shape[-1] != group_num:
        raise ValueError(
            f"Mask has {mask.shape[-1]} groups, expected {group_num}"
        )

    hidden_states = rearrange(
        hidden_states,
        'b l (n h) -> b l n h',
        h=group_size,
    )
    hidden_states = hidden_states * mask[..., None]
    return rearrange(hidden_states, 'b l n h -> b l (n h)')

def get_group_num(
    total_size: int,
    group_size: int,
    min_group_size: int,
    name: str,
) -> int:
    if group_size < min_group_size:
        raise ValueError(
            f"{name}_group_size must greater than {min_group_size}, but got {group_size}"
        )
    if total_size % group_size:
        raise ValueError(
            f"{name}_group_size={group_size} must divide {name} size "
            f"{total_size}"
        )
    return total_size // group_size

class TokenRouter(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        group_num: Optional[int]=-1,
        rank_size: Optional[int]=-1,
        **kwargs,
    ):
        super().__init__()
        assert group_num > 0 and rank_size > 0
        self.compressor = nn.Linear(hidden_size, rank_size)
        self.router = nn.Linear(rank_size, group_num * 2)
        
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        if kwargs.get('lora_init', False): nn.init.zeros_(self.router.weight)

    def forward(self, x):
        res = self.router(self.compressor(x))
        if res.shape[-1] > 2: res = rearrange(res, 'b s (h d) -> b s h d', d=2)
        return res

class WrapperMLP(nn.Module):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args: Namespace):
        super().__init__()
        self.config = config
        self.args = args

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.ffn_group_size = args.ffn_group_size
        self.gate_proj: nn.Linear | LoraLinear = block.gate_proj
        self.up_proj: nn.Linear | LoraLinear = block.up_proj
        self.down_proj: nn.Linear | LoraLinear = block.down_proj
        self.act_fn = block.act_fn

        assert self.config.hidden_act in ['silu']
        self.swiglu_func = LigerSiLUMulFunction()

class WrapperAttention(nn.Module):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args: Namespace):
        super().__init__()
        self.config = config
        self.args = args
        self.layer_idx = block.layer_idx

        self.hidden_size = config.hidden_size
        self.attn_group_size = args.attn_group_size
        self.head_dim = block.head_dim
        self.scaling = block.scaling
        self.attention_dropout = block.attention_dropout
        self.is_causal = block.is_causal

        self.q_proj = block.q_proj
        self.k_proj = block.k_proj
        self.v_proj = block.v_proj
        self.o_proj = block.o_proj

        self.q_norm = getattr(block, 'q_norm', None)
        self.k_norm = getattr(block, 'k_norm', None)

class WrapperDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: PretrainedConfig,
        block: nn.Module,
        args: Namespace,
    ):
        super().__init__()
        self.args = args
        self.config = config
        self._router_names = []

        self.sparse_target = args.sparse_target
        self.hidden_size = config.hidden_size
        self.num_heads_q = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads_q
        self.attn_group_size = args.attn_group_size
        self.ffn_group_size = args.ffn_group_size
        self.rank_size = getattr(args, 'route_rank_size', 16)
        self.intermediate_size = config.intermediate_size

        self.input_layernorm = block.input_layernorm
        self.post_attention_layernorm = block.post_attention_layernorm

        self.training_step = 0

        assert self.config.hidden_act in ['silu']
        self.swiglu_func = LigerSiLUMulFunction()
    
    def _gumbel_sigmoid(self, logits: torch.Tensor, tau: float, threshold: float = 0.5):
        if self.training:
            logistic = (
                torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log() - 
                torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
            )
            soft = (logits + logistic) / tau
            soft = F.sigmoid(soft)

            # STE
            hard = (soft > threshold).to(soft.dtype)
            return (hard - soft.detach() + soft).to(logits.dtype)
        else:
            soft = F.sigmoid(logits)
            return (soft > threshold).to(logits.dtype)
    
    def _gumbel_softmax(self, logits: torch.Tensor, tau: float):
        if self.training:
            res = F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1).to(logits.dtype)
            return res[..., 0], res[..., 1]
        else:
            skip_mask = logits.argmax(dim=-1).to(torch.bool)
            execute_mask = skip_mask.logical_not()
            return execute_mask.to(logits.dtype), skip_mask.to(logits.dtype)
    
    def _update_skipped_tokens(
        self,
        tgt: str,
        valid_token_num: torch.Tensor,
        skip_mask: torch.Tensor,
        skipped_tokens: Dict[str, torch.Tensor],
        router_zero_prob: Dict[str, torch.Tensor],
    ):
        assert tgt in self._router_names, f'{tgt} is not a valid router name'
        group_num = 1 if skip_mask.dim() < 3 else skip_mask.shape[-1]

        if tgt not in skipped_tokens:
            skipped_tokens[tgt] = torch.zeros((1,), device=skip_mask.device)
        skipped_tokens[tgt] = skipped_tokens[tgt] + skip_mask.sum()

        if tgt not in router_zero_prob:
            router_zero_prob[tgt] = torch.zeros((1,), device=skip_mask.device)
        router_zero_prob[tgt] = router_zero_prob[tgt] + skip_mask.sum() / (valid_token_num * group_num)


class WrapperPretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WrapperDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": WrapperDecoderLayer,
        "attentions": WrapperAttention,
    }


class WrapperModel(WrapperPretrainedModel):
    def __init__(self, block: nn.Module, config: PretrainedConfig, args):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = block.embed_tokens
        self.layers = nn.ModuleList()
        self.norm = block.norm
        self.rotary_emb = block.rotary_emb
        self.gradient_checkpointing = block.gradient_checkpointing

        # initialize the total tokens and skipped tokens
        self._router_names = []
        self.total_tokens = 0
        self.skipped_tokens = {}
        self.router_zero_prob = {}

        self.hidden_size = config.hidden_size
        self.num_heads_q = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads_q
        self.attn_group_size = args.attn_group_size
        self.ffn_group_size = args.ffn_group_size
        self.intermediate_size = config.intermediate_size

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCacheSeqFirst(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if attention_mask is None:
            b, s, _ = inputs_embeds.shape
            attention_mask = torch.ones((b, s), device=inputs_embeds.device)
        self.total_tokens += attention_mask.sum()

        hidden_states = inputs_embeds
        # hidden_states.requires_grad_() # for gradient checkpointing in reentrant mode
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        router_zero_prob = {}
        skipped_tokens = {}
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states, skipped_tokens, router_zero_prob = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                skipped_tokens=skipped_tokens,
                router_zero_prob=router_zero_prob,
                **kwargs,
            )
        
        for key in router_zero_prob:
            self.router_zero_prob[key] = router_zero_prob[key] / self.config.num_hidden_layers
        
        for key in skipped_tokens:
            if key not in self.skipped_tokens: self.skipped_tokens[key] = skipped_tokens[key]
            else: self.skipped_tokens[key] = self.skipped_tokens[key] + skipped_tokens[key]

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
    
    def compute_sparsity(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("compute_sparsity is not implemented for WrapperModel")
