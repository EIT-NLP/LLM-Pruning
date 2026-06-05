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

from ops import __ATTENTION__, __MLP__, __ROUTER__, __KV_CACHE__, __APPROXIMATOR__, __INDEX__
from ops.utils import triton_rmsnorm, triton_rope_qk_align
from wrapper.base import PrunedModelForCausalLM
from wrapper.static.dense import DenseAttention, DenseMLP, DenseDecoderLayer, DenseModel, DenseForCausalLM

def generate_sparsity(pruning_config: Dict[str, Any]) -> float:
    estimated_sparsity = pruning_config.get("estimated_sparsity", 0.0)
    offset = pruning_config.get("offset", 0.0)
    sampled_offset = random.uniform(-offset, offset)
    return estimated_sparsity + sampled_offset

#################### FFN ####################
class PropagateMLP(DenseMLP):
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
        self.residual_map = nn.Identity()

    def _residual_mapping(self):
        # self.residual_map = nn.Identity()
        # if self.up_proj.in_features != self.down_proj.out_features:
        self.residual_map = nn.Linear(
            self.up_proj.in_features, self.down_proj.out_features,
            bias=False, device=self.up_proj.weight.device, dtype=self.up_proj.weight.dtype
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        with nvtx.annotate("glu", color='green'):
            with record_function("glu"):
                glu_output = self.mlp_impl(
                    hidden_states,
                    w_up=self.up_proj.weight,
                    w_gate=self.gate_proj.weight,
                    b_up=self.up_proj.bias,
                    b_gate=self.gate_proj.bias,
                    activation=self.activation,
                )
        with nvtx.annotate("down_proj", color='green'):
            with record_function("down_proj"):
                ffn_output = self.mlp_impl(
                    glu_output,
                    w_up=self.down_proj.weight,
                    b_down=self.down_proj.bias,
                )
        return ffn_output + self.residual_map(residual)

#################### ATTN ####################
class PropagateAttention(DenseAttention):
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
        self.residual_map = nn.Identity()
    
    def _residual_mapping(self):
        # self.residual_map = nn.Identity()
        # if self.q_proj.in_features != self.o_proj.out_features:
        self.residual_map = nn.Linear(
            self.q_proj.in_features, self.o_proj.out_features,
            bias=False, device=self.q_proj.weight.device, dtype=self.q_proj.weight.dtype
        )
    
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
                q = self.q_proj(hidden_states)
            with nvtx.annotate("k_proj", color='blue'):
                k = self.k_proj(hidden_states)
            with nvtx.annotate("v_proj", color='blue'):
                v = self.v_proj(hidden_states)

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
                attn_output = self.o_proj(attn_output)

        return attn_output + self.residual_map(residual)

#################### Layer ####################
class PropagateDecoderLayer(DenseDecoderLayer):
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
        self._propagate_group = tuple([
            ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",),
            ("self_attn.o_proj",),
            ("mlp.up_proj", "mlp.gate_proj",),
            ("mlp.down_proj",),
        ])

        self.self_attn = PropagateAttention(config, pruning_config, block.self_attn, block.input_layernorm, **kwargs)
        self.mlp = PropagateMLP(config, pruning_config, block.mlp, block.post_attention_layernorm, **kwargs)

class PropagatePretrainedModel(PreTrainedModel):
    config: PretrainedConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["PropagateDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = False
    _can_record_outputs = {
        "hidden_states": PropagateDecoderLayer,
        "attentions": PropagateAttention,
    }

class PropagateModel(PropagatePretrainedModel):
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
        self.layers = nn.ModuleList([PropagateDecoderLayer(config, pruning_config, block.layers[i], i) for i in range(config.num_hidden_layers)])
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

class PropagateForCausalLM(PrunedModelForCausalLM):
    def __init__(
        self,
        config: PretrainedConfig,
        pruning_config: Dict[str, Any],
        block: PreTrainedModel,
        **kwargs,
    ):
        super().__init__(config, pruning_config, block, **kwargs)
        self.model = PropagateModel(config, pruning_config, block.model, **kwargs)

        # initialize pruning chain
        self.pruning_chain = []
        if pruning_config.get('prune_embedding', False): self.pruning_chain.append(('model.embed_tokens',))
        for i, layer in enumerate(self.model.layers):
            layer_chain = layer._propagate_group
            real_chain = []
            for group in layer_chain:
                real_chain.append(tuple([f'model.layers.{i}.{component}' for component in group]))
            self.pruning_chain.extend(real_chain)
        
        if self.pruning_config.get('prune_lm_head', False): self.pruning_chain.append(('lm_head',))

        self.is_pruned = False
    
    # Generate random route mask for benchmark
    def generate_pruning_kwargs(self, **kwargs) -> Dict[str, torch.Tensor]:
        for layer in self.model.layers:
            layer.generate_pruning_kwargs(**kwargs)
        
        if self.is_pruned: return {}

        patch_type = self.pruning_config.get('patch_type', 'structured')

        # layer & sublayer first
        layer_pruning_config = self.pruning_config.get('layer', {})
        attention_pruning_config = self.pruning_config.get('self_attn', {})
        ffn_pruning_config = self.pruning_config.get('mlp', {})

        layer_skip_ids = set()
        if layer_pruning_config.get('pruning_type', None) == 'M':
            layer_skip_ids = set(__INDEX__[patch_type].monkey_patch_layer(
                root_module=self,
                sparsity=generate_sparsity(layer_pruning_config)
            ))

        attention_skip_ids = set()
        if attention_pruning_config.get('pruning_type', None) == 'M':
            attention_skip_ids = set(__INDEX__[patch_type].monkey_patch_sublayer(
                root_module=self,
                target='self_attn',
                sparsity=generate_sparsity(attention_pruning_config)
            ))
        
        ffn_skip_ids = set()
        if ffn_pruning_config.get('pruning_type', None) == 'M':
            ffn_skip_ids = set(__INDEX__[patch_type].monkey_patch_sublayer(
                root_module=self,
                target='mlp',
                sparsity=generate_sparsity(ffn_pruning_config)
            ))
        
        deleted_layers = []
        for layer_ids in layer_skip_ids:
            chains = self.model.layers[layer_ids]._propagate_group
            real_chain = []
            for group in chains:
                real_chain.append(tuple([f'model.layers.{layer_ids}.{component}' for component in group]))
            deleted_layers.extend(real_chain)
        
        for layer_ids in attention_skip_ids:
            chains = self.model.layers[layer_ids]._propagate_group[:2]
            real_chain = []
            for group in chains:
                real_chain.append(tuple([f'model.layers.{layer_ids}.{component}' for component in group]))
            deleted_layers.extend(real_chain)
        
        for layer_ids in ffn_skip_ids:
            chains = self.model.layers[layer_ids]._propagate_group[2:]
            real_chain = []
            for group in chains:
                real_chain.append(tuple([f'model.layers.{layer_ids}.{component}' for component in group]))
            deleted_layers.extend(real_chain)
        
        # remove deleted layers
        deleted_layers = set(deleted_layers)
        self.pruning_chain = [group for group in self.pruning_chain if group not in deleted_layers]
        
        # then for each layer's row & col
        i = 0
        while i < len(self.pruning_chain):
            group = self.pruning_chain[i]

            if not isinstance(group, Sequence): 
                i += 1
                continue

            component = group[0]
            component_name = '.'.join(component.split('.')[-2:])

            prefix, suffix = component_name.split('.') if len(component_name.split('.')) == 2 else ('', component_name)
            component_config = self.pruning_config.get(prefix, {}).get(suffix, {})
            pruning_type = component_config.get('pruning_type', None)
            src_targets = dst_targets = ()
            if pruning_type == 'N':
                src_targets = group
                dst_targets = self.pruning_chain[i+1] if i+1 < len(self.pruning_chain) else ()
            elif pruning_type == 'K':
                src_targets = self.pruning_chain[0] if i == 1 and 'model.embed_tokens' in self.pruning_chain[0] else ()
                dst_targets = group
            else:
                i += 1
                continue

            component_skip_ids = set(__INDEX__[patch_type].monkey_patch_nk(
                root_module=self,
                src_targets=src_targets,
                dst_targets=dst_targets,
                sparsity=generate_sparsity(component_config),
                num_query_heads=self.config.num_attention_heads,
                num_key_value_heads=self.config.num_key_value_heads,
                head_dim=self.config.hidden_size // self.config.num_attention_heads,
            ))

            i += 2 if pruning_type == 'N' else 1
        
        # epilogue
        if self.pruning_config.get('residual_mapping', False):
            for layer in self.model.layers:
                layer.self_attn._residual_mapping()
                layer.mlp._residual_mapping()
        
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