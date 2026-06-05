import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any

###################
# Various Approximator for efficient compensating accuracy
###################
class Approximator(nn.Module):
    """
    Basic Approximator Class
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        if self.training:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
                    if module.bias is not None: nn.init.zeros_(module.bias, 0)
    
    def load_approximator_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: List[str],
        **kwargs,
    ):
        assert len(prefix) == len(self.components)
        for i, local_prefix in enumerate(prefix):
            local_state_dict = {k.replace(local_prefix, self.components[i]): v for k, v in state_dict.items() if k.startswith(local_prefix)}
            getattr(self, self.components[i]).load_state_dict(local_state_dict, strict=True)


class BottleneckApproximator(Approximator):
    """
    Use a bottleneck projection matrix for approximating the original hidden states:

    x[..., D], w1[r, D], w2[D, r] -> (x @ w1^T) @ w2^T
    """
    def __init__(
        self,
        hidden_size: int,
        rank_size: int,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank_size = rank_size

        self.down_proj = nn.Linear(self.hidden_size, self.rank_size)
        self.approximator = nn.Linear(self.rank_size, self.hidden_size)

        self.components = ['down_proj', 'approximator']

        super().__init__()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        approx = self.down_proj(hidden_states)
        approx = self.approximator(approx)
        return approx
