import torch
import torch.nn as nn

from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any

###################
# Various Router for Dynamic Pruning
###################
class Router(nn.Module):
    """
    Basic Router Class
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def load_router_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefix: List[str],
        **kwargs,
    ):
        assert len(prefix) == len(self.components)
        for i, local_prefix in enumerate(prefix):
            local_state_dict = {k.replace(local_prefix, self.components[i]): v for k, v in state_dict.items() if k.startswith(local_prefix)}
            getattr(self, self.components[i]).load_state_dict(local_state_dict, strict=True)
    
    def epilogue(
        self,
        route: torch.Tensor, # [..., 2]
        skip_dim: Optional[int]=1,
    ):
        """
        Return `Execution` and `Skip` mask
        """
        route = route.argmax(dim=-1).to(torch.bool)
        route_neg = route.logical_not()
        return (route_neg, route) if skip_dim == 1 else (route, route_neg)

class LinearRouter(Router):
    """
    Use a single projection matrix for generating route indices:

    x[..., D], w[D // NG, 2] -> x reshape to [..., NG, D // NG] -> o = x @ w + b, where o[..., 0] = 1 means execute, o[..., 1] = 1 means skip, based on gumbel softmax
    """
    def __init__(
        self,
        hidden_size: int,
        num_groups: Optional[int]=1,
        update_freq: Optional[str]='token',
        skip_dim: Optional[int]=1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.group_size = hidden_size // num_groups
        self.exact_hidden_size = self.group_size * self.num_groups

        assert self.exact_hidden_size <= hidden_size

        self.router = nn.Linear(self.group_size, 2)
        self.update_freq = update_freq
        self.skip_dim = skip_dim

        self.components = ['router']

        if self.training:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
                    if module.bias is not None: nn.init.zeros_(module.bias)
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        route = hidden_states if self.exact_hidden_size == self.hidden_size else hidden_states[..., :self.exact_hidden_size]
        route = self.router(rearrange(route, '... (ng g) -> ... ng g', ng=self.num_groups))
        if self.num_groups == 1: route = route.squeeze(-2)
        return self.epilogue(route, skip_dim=self.skip_dim)

class BottleneckRouter(Router):
    """
    Use an addition down-scale matrix for generating route indices:

    x[..., D], w1[D, r], w2[r, NG * 2]
     
    -> o = (x @ w1 + b1) @ w2 + b2 -> reshape o to [..., NG, 2], where o[..., 0] = 1 means execute, o[..., 1] = 0 means skip, based on gumbel softmax
    """
    def __init__(
        self,
        hidden_size: int,
        rank_size: int,
        num_groups: Optional[int]=1,
        update_freq: Optional[str]='token',
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank_size = rank_size
        self.num_groups = num_groups

        self.down_proj = nn.Linear(self.hidden_size, self.rank_size)
        self.router = nn.Linear(self.rank_size, self.num_groups * 2)
        self.update_freq = update_freq
        
        self.components = ['down_proj', 'router']

        if self.training:
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='linear')
                    if module.bias is not None: nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        route = hidden_states
        route = self.router(self.down_proj(route))
        if self.num_groups > 1: route = rearrange(route, '... (ng g) -> ... ng g', ng=self.num_groups, g=2)
        return self.epilogue(route, skip_dim=1)
