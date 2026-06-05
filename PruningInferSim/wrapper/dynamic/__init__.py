from .propagate import PropagateForCausalLM
from .sparse_attention import SparseAttentionForCausalLM

from config import register_wrapper

register_wrapper("propagate", "dynamic")(PropagateForCausalLM)
register_wrapper("sparse_attention", "dynamic")(SparseAttentionForCausalLM)

__all__ = ["PropagateForCausalLM", "SparseAttentionForCausalLM"]
