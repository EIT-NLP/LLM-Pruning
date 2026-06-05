import random
import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Dict, Sequence
from torch.sparse.semi_structured import SparseSemiStructuredTensor

from .mlp.semi_sparse_pruning import CuSPARSELtLinear, TorchCuSPARSELtLinear

###########################
#   Mask for Unstructured
###########################

class BaseMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        pass
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        pass


class UnstructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        return (torch.rand(shape, device=device) < sparsity).to(torch.bool)
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        module.weight *= mask.to(module.weight.dtype)
        return module

class SemiStructuredMask:
    @classmethod
    def random_sample(
        cls,
        shape: Sequence[int],
        sparsity: Optional[float]=0.5,
        block_size: Optional[int]=4,
        device: Optional[torch.device]=None,
        **kwargs,
    ) -> torch.Tensor:
        left = int(sparsity * block_size)
        right = block_size
        assert left > 0

        mask = torch.ones(shape, device=device).flatten()
        mask = rearrange(mask, '(a b) -> a b', b=right)
        indices = torch.multinomial(mask, left, replacement=False)

        rows = torch.arange(mask.shape[0], device=device).view(-1, 1).expand(-1, left)
        mask[rows, indices] = False
        mask = mask.reshape(shape)
        return mask.to(torch.bool)
    
    @classmethod
    def monkey_patch(
        cls,
        module: nn.Linear,
        mask: Optional[torch.Tensor]=None,
        **kwargs,
    ):
        assert module.bias is None
        if isinstance(module.weight.data, SparseSemiStructuredTensor): return
        if mask is None:
            mask = cls.random_sample(module.weight.shape, device=module.weight.device, **kwargs)
        
        backend = kwargs.get('backend', 'torch')
        if backend == 'torch':
            return TorchCuSPARSELtLinear(module.weight, mask)
        elif backend == 'jit':
            return CuSPARSELtLinear(module.weight, mask)
        else:
            raise ValueError(f"Unknown semi_structured backend: {backend}")
