from .dense import DenseForCausalLM
from .lowrank import LowRankForCausalLM
from .unstructured import UnstructuredForCausalLM
from .propagate import PropagateForCausalLM
from config import register_wrapper

register_wrapper("dense", "static")(DenseForCausalLM)
register_wrapper("propagate", "static")(PropagateForCausalLM)
register_wrapper("lowrank", "static")(LowRankForCausalLM)
register_wrapper("unstructured", "static")(UnstructuredForCausalLM)

__all__ = [
    "DenseForCausalLM",
    "PropagateForCausalLM",
    "LowRankForCausalLM",
    "UnstructuredForCausalLM",
]
