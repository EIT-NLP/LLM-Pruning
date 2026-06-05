import types
import random
import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Dict, Sequence

###########################
#   Index for Structured
###########################

def get_module_recursive(start_module: nn.Module, target: str) -> nn.Module:
    module = start_module
    for part in target.split('.'):
        if len(part) == 0: continue
        module = getattr(module, part)
    return module

def identity_mapping(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return hidden_states

class BaseIndex:
    __propagate_targets__ = [
        'attention.q_proj',
        'attention.o_proj',
        'ffn.up_proj',
        'ffn.down_proj'
    ]

    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        dim: Optional[int]=0,
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
    @classmethod
    def monkey_patch_layer(
        cls,
        root_module: nn.Linear,
        sparsity: Optional[float]=0.5,
        **kwargs,
    ):
        pass

    @classmethod
    def monkey_patch_sublayer(
        cls,
        root_module: nn.Linear,
        target: str,
        sparsity: Optional[float]=0.5,
        ignore_idx: Optional[Sequence[int]]=None,
        **kwargs,
    ):
        pass

    @classmethod
    def check_if_pruneable(
        cls,
        root_module: nn.Module,
        module_name: Optional[str]='',
        **kwargs,
    ):
        tgt_module = get_module_recursive(root_module, module_name)
        pruneable = (not hasattr(tgt_module, '_applied_pruning')) or len(tgt_module._applied_pruning) == 0
        if hasattr(tgt_module, '_leaf_modules'):
            for leaf_module in tgt_module._leaf_modules:
                pruneable &= cls.check_if_pruneable(tgt_module, leaf_module, **kwargs)
        return pruneable
    
    @classmethod
    def mark_pruned(
        cls,
        root_module: nn.Module,
        pruning_type: str,
        module_name: Optional[str]='',
        **kwargs,
    ):
        tgt_module = get_module_recursive(root_module, module_name)
        if not hasattr(tgt_module, '_applied_pruning'): tgt_module._applied_pruning = []
        tgt_module._applied_pruning.append(pruning_type)

        if hasattr(tgt_module, '_leaf_modules'):
            for leaf_module in tgt_module._leaf_modules:
                cls.mark_pruned(tgt_module, pruning_type, leaf_module, **kwargs)
    
    @classmethod
    def delete_pruned(
        cls,
        root_module: nn.Module,
        module_name: Optional[str]='',
        **kwargs,
    ):
        tgt_module = get_module_recursive(root_module, module_name)
        for name, param in tgt_module.named_parameters():
            empty_param = nn.Parameter(torch.empty(0, device=param.device, dtype=param.dtype))
            module = get_module_recursive(tgt_module, '.'.join(name.split('.')[:-1]))
            setattr(module, name.split('.')[-1], empty_param)

            if isinstance(module, nn.Linear):
                module.in_features = module.out_features = 0

class StructuredIndex(BaseIndex):
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        dim: Optional[int]=0,
        k: Optional[int]=-1,
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        rounding: Optional[str|int]='',
        **kwargs,
    ) -> torch.Tensor:
        sparsity = max(0, min(1, sparsity))
        k = int(shape[dim] * sparsity) if (k < 1 or k > shape[dim]) else k
        sparsity = sparsity if sparsity >= 0 and sparsity <= 1 else k / shape[dim]

        # handle rounding strategy
        if rounding == 'even':
            if k % 2 != 0: k += 1
        elif rounding == 'odd':
            if k % 2 == 0: k -= 1
        elif isinstance(rounding, int) and k > rounding and sparsity > 0:
            # round to the nearest multiple of rounding
            if k % rounding != 0: k = rounding * ((k + rounding - 1) // rounding)

        indices = random.sample(range(shape[dim]), k=k)
        indices = sorted(indices)
        indices = torch.tensor(indices, device=device) if kwargs.get('return_tensor', True) else indices
        return indices
    
    @classmethod
    def monkey_patch_layer(
        cls,
        root_module: nn.Module,
        sparsity: Optional[float]=0.5,
        indices: Optional[Sequence[int]]=None,
        **kwargs,
    ):
        module = root_module.model.layers
        device = module[0].self_attn.q_proj.weight.device

        if indices is None:
            candidate_indices = [i for i in range(len(module)) if cls.check_if_pruneable(module[i])]
            indices = cls.random_sample(
                shape=(len(module),),
                dim=0,
                sparsity=sparsity,
                return_tensor=False,
                device=device,
            )
        
        for layer_id in indices:
            module[layer_id].forward = types.MethodType(identity_mapping, module[layer_id])
            cls.mark_pruned(module[layer_id], 'M')
            cls.delete_pruned(module[layer_id])
        
        return indices
    
    @classmethod
    def monkey_patch_sublayer(
        cls,
        root_module: nn.Module,
        target: str,
        sparsity: Optional[float]=0.5,
        indices: Optional[Sequence[int]]=None,
        **kwargs,
    ):
        module = root_module.model.layers
        device = module[0].self_attn.q_proj.weight.device

        if indices is None:
            candidate_indices = [i for i in range(len(module)) if cls.check_if_pruneable(module[i], target)]
            indices = cls.random_sample(
                shape=(len(module),),
                dim=0,
                sparsity=sparsity,
                return_tensor=False,
                device=device,
            )
        
        for layer_id in indices:
            setattr(getattr(module[layer_id], target), 'forward', types.MethodType(identity_mapping, getattr(module[layer_id], target)))
            cls.mark_pruned(module[layer_id], 'M', target)
            cls.delete_pruned(module[layer_id], target)
        
        return indices
    
    @classmethod
    def monkey_patch_nk(
        cls,
        root_module: nn.Module,
        src_targets: Sequence[str],
        dst_targets: Sequence[str],
        sparsity: Optional[float]=0.5,
        indices: Optional[torch.Tensor]=None,
        num_query_heads: Optional[int]=0,
        num_key_value_heads: Optional[int]=0,
        head_dim: Optional[int]=0,
        **kwargs,
    ):
        device = root_module.model.layers[0].self_attn.q_proj.weight.device

        if len(src_targets) > 0:
            start_module = get_module_recursive(root_module, src_targets[0])
            prune_dim = 1 if src_targets[0] == 'model.embed_tokens' else 0
        else:
            start_module = get_module_recursive(root_module, dst_targets[0])
            prune_dim = 1

        start_module = get_module_recursive(root_module, src_targets[0] if len(src_targets) > 0 else dst_targets[0])
        is_src_prunable = all([cls.check_if_pruneable(root_module, _) for _ in src_targets]) if len(src_targets) > 0 else True
        is_tgt_prunable = all([cls.check_if_pruneable(root_module, _) for _ in dst_targets]) if len(dst_targets) > 0 else True

        if not (is_src_prunable and is_tgt_prunable): return []

        is_multi_group = False
        if (len(src_targets[0].split('.')) != len(dst_targets[0].split('.'))) or (src_targets[0].split('.')[:-1] != dst_targets[0].split('.')[:-1]): is_multi_group = True # need to handle the dimension of norm
        
        is_head_pruning = len(src_targets) * len(dst_targets) != 0 and src_targets[0].split('.')[-1] == 'q_proj' and dst_targets[0].split('.')[-1] == 'o_proj'
        if is_head_pruning:
            group_size = num_query_heads // num_key_value_heads
            if indices is None:
                indices = cls.random_sample(
                    shape=(num_key_value_heads,),
                    dim=0,
                    sparsity=1 - sparsity,
                    device=device,
                )
                indices = (indices[:, None] * group_size + torch.arange(group_size, device=device)[None, :]).flatten()
            
            for module_name in src_targets:
                module = get_module_recursive(root_module, module_name)
                weight = rearrange(module.weight, '(h g) d -> h g d', g=head_dim)

                module.weight = nn.Parameter(weight[indices if 'q_proj' in module_name else indices.reshape(-1, group_size)[:, 0] // group_size, :].flatten(0, 1))
                module.out_features = module.weight.shape[0]
                cls.mark_pruned(module, 'N')
            
            for module_name in dst_targets:
                module = get_module_recursive(root_module, module_name)
                weight = rearrange(module.weight, 'd (h g) -> d h g', g=head_dim)

                module.weight = nn.Parameter(weight[:, indices, :].flatten(1, 2))
                module.in_features = module.weight.shape[1]
                cls.mark_pruned(module, 'K')
        
        else:
            if indices is None:
                indices = cls.random_sample(
                    shape=start_module.weight.shape,
                    dim=prune_dim,
                    sparsity=1 - sparsity,
                    device=device,
                    rounding=16,
                )
            
            for module_name in src_targets:
                module = get_module_recursive(root_module, module_name)
                if isinstance(module, nn.Linear):
                    module.weight = nn.Parameter(module.weight[indices, :])
                    module.out_features = module.weight.shape[0]
                elif isinstance(module, nn.Embedding):
                    module.weight = nn.Parameter(module.weight[:, indices])
                    module.embedding_dim = module.weight.shape[1]
                
                cls.mark_pruned(module, 'N')
        
            for module_name in dst_targets:
                module = get_module_recursive(root_module, module_name)
                if isinstance(module, nn.Linear):
                    module.weight = nn.Parameter(module.weight[:, indices])
                    module.in_features = module.weight.shape[1]
                
                cls.mark_pruned(module, 'K')
            
            if is_multi_group:
                # norm
                if dst_targets[0] == 'lm_head':
                    root_module.model.norm.weight = nn.Parameter(root_module.model.norm.weight[indices])
                elif dst_targets[0].split('.')[-2] == 'self_attn':
                    norm_type = 'input_layernorm'
                    norm = get_module_recursive(root_module, '.'.join(dst_targets[0].split('.')[:-1] + [norm_type]))
                    norm.weight = nn.Parameter(norm.weight[indices])
                elif dst_targets[0].split('.')[-2] == 'mlp':
                    norm_type = 'post_attention_layernorm'
                    norm = get_module_recursive(root_module, '.'.join(dst_targets[0].split('.')[:-1] + [norm_type]))
                    norm.weight = nn.Parameter(norm.weight[indices])
        
        return indices
