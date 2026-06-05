from .cache import DynamicCacheSeqFirst
from .utils import triton_rmsnorm, triton_rope_qk_align

from .router import LinearRouter, BottleneckRouter
from .approximator import BottleneckApproximator

from .mask import BaseMask, UnstructuredMask, SemiStructuredMask

from .index import BaseIndex, StructuredIndex

from .lowrank import BaseLowRank

from .attention_threshold import BaseThreshold, BlasstThreshold, SeerThreshold

from .attention.base import DenseAttentionKernel
from .attention.query_pruning import MSparseAttentionKernel
from .attention.kv_pruning import BlockSparseAttentionKernel

from .mlp.base import DenseMLPKernel, LowRankMLPKernel
from .mlp.m_pruning import MSparseMLPKernel

__all__ = [
    "DynamicCacheSeqFirst",
    "triton_rmsnorm",
    "triton_rope_qk_align",
    "DenseAttentionKernel",
    "MSparseAttentionKernel",
    "BlockSparseAttentionKernel",
    "DenseMLPKernel",
    "MSparseMLPKernel",
]

__ROUTER__ = {
    "linear": LinearRouter,
    "bottleneck": BottleneckRouter,
}

__APPROXIMATOR__ = {
    "bottleneck": BottleneckApproximator,
}

__KV_CACHE__ = {
    "base": DynamicCacheSeqFirst,
}

__MASK__ = {
    "base": BaseMask,
    "unstructured": UnstructuredMask,
    "semi_structured": SemiStructuredMask,
}

__INDEX__ = {
    "base": BaseIndex,
    "structured": StructuredIndex,
}

__LOW_RANK__ = {
    "base": BaseLowRank,
}

__THRESHOLD__ = {
    "base": BaseThreshold,
    "blasst": BlasstThreshold,
    "seer": SeerThreshold,
}

__ATTENTION__ = {
    "base": DenseAttentionKernel,
    "m": MSparseAttentionKernel,
}

__SPARSE_ATTENTION__ = {
    "base": DenseAttentionKernel,
    "blasst": BlockSparseAttentionKernel,
    "seer": BlockSparseAttentionKernel,
}

__MLP__ = {
    "base": DenseMLPKernel,
    "low_rank": LowRankMLPKernel,
    "m": MSparseMLPKernel,
}