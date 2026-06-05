import types
import random
import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Dict, Sequence

###########################
#   Index for LowRank
###########################
def get_module_recursive(start_module: nn.Module, target: str) -> nn.Module:
    module = start_module
    for part in target.split('.'):
        if len(part) == 0: continue
        module = getattr(module, part)
    return module

class BaseLowRank:
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
    def calculate_rank(
        cls,
        sparsity: Optional[float]=0.5,
        retained_ratio: Optional[float]=0.0,
        N_raw: Optional[int]=None,
        K_raw: Optional[int]=None,
        rounding: Optional[str|int]='',
        **kwargs,
    ):
        if kwargs.get('is_ffn', False) == False or retained_ratio == 0: # single linear
            base_r = int(sparsity * (N_raw * K_raw / (N_raw + K_raw)))
        else:
            base_r = int(N_raw * K_raw * (sparsity - retained_ratio) / (N_raw + K_raw - N_raw * retained_ratio))
        
        # handle rounding strategy
        if rounding == 'even':
            if base_r % 2 != 0: base_r += 1
        elif rounding == 'odd':
            if base_r % 2 == 0: base_r -= 1
        elif isinstance(rounding, int) and base_r > rounding and sparsity > 0:
            # round to the nearest multiple of rounding
            if base_r % rounding != 0: base_r = rounding * ((base_r + rounding - 1) // rounding)

        return base_r
    
    @classmethod
    def slice_weight(
        cls,
        module: nn.Linear,
        indices: Optional[torch.Tensor]=None,
        dim: Optional[int]=0,
    ):
        if indices is None: # set to empty
            empty_param = nn.Parameter(torch.empty(0, device=module.weight.device, dtype=module.weight.dtype))
            setattr(module, 'weight', empty_param)
            module.in_features = module.out_features = 0
        else:
            assert dim in [0, 1]
            setattr(module, 'weight', nn.Parameter(module.weight[indices, :] if dim == 0 else module.weight[:, indices]))
            if dim == 0: module.out_features = indices.numel()
            else: module.in_features = indices.numel()
    
    @classmethod
    def monkey_patch(
        cls,
        root_module: nn.Module,
        target: str,
        sparsity: Optional[float]=0.5,
        retained_ratio: Optional[float]=0.0,
        indices: Optional[torch.Tensor]=None,
        device: Optional[torch.device]=None,
        **kwargs,
    ):
        module = get_module_recursive(root_module, target)
        prev_module = get_module_recursive(root_module, '.'.join(target.split('.')[:-1]))
        component = target.split('.')[-1]

        if component in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj'] and retained_ratio == 0:
            rank = cls.calculate_rank(
                sparsity=1 - sparsity,
                retained_ratio=retained_ratio,
                N_raw=module.weight.shape[0],
                K_raw=module.weight.shape[1],
                rounding=16,
            )
            setattr(
                prev_module,
                f"{component}_r1",
                nn.Linear(
                    module.weight.shape[1], rank,
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            setattr(
                prev_module,
                f"{component}_r2",
                nn.Linear(
                    rank, module.weight.shape[0],
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            cls.slice_weight(
                module,
                indices=None,
                dim=0,
            )
        elif retained_ratio > 0 and component in ['up_proj', 'gate_proj']:
            # handle mlp
            if indices is None and retained_ratio > 0:
                indices = cls.random_sample(
                    shape=module.weight.shape,
                    dim=0,
                    sparsity=retained_ratio,
                    device=device,
                    rounding=16,
                )
            
            N, K = module.weight.shape
            
            rank = cls.calculate_rank(
                sparsity=1 - sparsity,
                retained_ratio=retained_ratio,
                N_raw=N,
                K_raw=K,
                rounding=16,
                is_ffn=True,
            )

            N0, N1 = indices.numel(), N - indices.numel()
            setattr(
                prev_module,
                f"{component}_r1",
                nn.Linear(
                    K, rank,
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            setattr(
                prev_module,
                f"{component}_r2",
                nn.Linear(
                    rank, N1,
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            cls.slice_weight(
                module,
                indices=indices,
                dim=0,
            )

        elif retained_ratio > 0 and component in ['down_proj']:
            # handle mlp
            if indices is None and retained_ratio > 0:
                indices = cls.random_sample(
                    shape=module.weight.shape,
                    dim=1,
                    sparsity=retained_ratio,
                    device=device,
                    rounding=16,
                )
            
            K, N = module.weight.shape
            
            rank = cls.calculate_rank(
                sparsity=1 - sparsity,
                retained_ratio=retained_ratio,
                N_raw=N,
                K_raw=K,
                rounding=16,
                is_ffn=True,
            )

            N0, N1 = indices.numel(), N - indices.numel()
            setattr(
                prev_module,
                f"{component}_r1",
                nn.Linear(
                    N1, rank,
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            setattr(
                prev_module,
                f"{component}_r2",
                nn.Linear(
                    rank, K,
                    bias=False, device=device, dtype=module.weight.dtype,
                )
            )
            cls.slice_weight(
                module,
                indices=indices,
                dim=1,
            )
