"""Dependency-light PyTorch correctness references for public backends."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _route_mask(mask: torch.Tensor, batch: int, tokens: int) -> torch.Tensor:
    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    if mask.dim() != 3 or mask.shape[:2] != (batch, tokens):
        raise ValueError(
            "Mask must have shape [batch, tokens] or [batch, tokens, groups]"
        )
    return mask.to(torch.bool)


def _store_output(result: torch.Tensor, output: Optional[torch.Tensor]) -> torch.Tensor:
    if output is None:
        return result
    output_view = output.reshape_as(result)
    output_view.copy_(result)
    return output_view


def gemm_mn(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    activation: str = "identity",
    **_: object,
) -> torch.Tensor:
    """Reference output-width-pruned linear projection."""

    if A.dim() != 3 or B.dim() != 2:
        raise ValueError("A must be [B, T, K] and B must be [N, K]")
    batch, tokens, _ = A.shape
    mask = _route_mask(Mask, batch, tokens)
    groups = mask.shape[-1]
    output_width = B.shape[0]
    if output_width % groups:
        raise ValueError(
            f"Output width {output_width} must be divisible by groups {groups}"
        )

    result = A @ B.transpose(0, 1)
    grouped = result.reshape(batch, tokens, groups, output_width // groups)
    grouped.masked_fill_(mask.logical_not().unsqueeze(-1), 0)
    result = grouped.reshape(batch, tokens, output_width)

    if activation == "silu":
        result = F.silu(result)
    elif activation == "relu":
        result = F.relu(result)
    elif activation != "identity":
        raise ValueError(f"Unsupported activation: {activation}")
    return _store_output(result, D)


def gemm_k(
    A: torch.Tensor,
    B: torch.Tensor,
    Mask: torch.Tensor,
    D: Optional[torch.Tensor] = None,
    **_: object,
) -> torch.Tensor:
    """Reference input-width-pruned linear projection."""

    if A.dim() != 3 or B.dim() != 2:
        raise ValueError("A must be [B, T, K] and B must be [N, K]")
    batch, tokens, input_width = A.shape
    mask = _route_mask(Mask, batch, tokens)
    groups = mask.shape[-1]
    if input_width % groups:
        raise ValueError(
            f"Input width {input_width} must be divisible by groups {groups}"
        )

    grouped = A.reshape(batch, tokens, groups, input_width // groups)
    grouped = grouped.masked_fill(mask.logical_not().unsqueeze(-1), 0)
    result = grouped.reshape(batch, tokens, input_width) @ B.transpose(0, 1)
    return _store_output(result, D)


def _attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    O: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    Leftpad: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    **_: object,
) -> torch.Tensor:
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError("Q, K, and V must have shape [B, T, H, D]")
    if K.shape != V.shape:
        raise ValueError("K and V must have matching shapes")

    batch, tokens, query_heads, head_dim = Q.shape
    mask = _route_mask(Mask, batch, tokens)
    groups = mask.shape[-1]
    if query_heads % groups:
        raise ValueError(
            f"Query heads {query_heads} must be divisible by groups {groups}"
        )

    query = Q.transpose(1, 2)
    key = K.transpose(1, 2)
    value = V.transpose(1, 2)
    attention_mask = None
    if Leftpad is not None:
        leftpad = Leftpad.to(device=Q.device, dtype=torch.long)
        if leftpad.shape != (batch,):
            raise ValueError("Leftpad must have shape [batch]")
        key_positions = torch.arange(K.shape[1], device=Q.device)
        attention_mask = key_positions[None, :] >= leftpad[:, None]
        query_positions = torch.arange(tokens, device=Q.device)
        query_positions = query_positions + K.shape[1] - tokens
        attention_mask = attention_mask[:, None, None, :]
        if is_causal:
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            attention_mask = attention_mask & causal_mask[None, None, :, :]
            is_causal = False

    result = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=is_causal,
        scale=scale if scale is not None else head_dim**-0.5,
        enable_gqa=query_heads != K.shape[2],
    ).transpose(1, 2)
    grouped = result.reshape(
        batch,
        tokens,
        groups,
        query_heads // groups,
        head_dim,
    )
    grouped.masked_fill_(mask.logical_not()[..., None, None], 0)
    result = grouped.reshape(batch, tokens, query_heads, head_dim)
    return _store_output(result, O)


def attention_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    """Reference causal prefill attention with query/head routing."""

    kwargs.setdefault("is_causal", True)
    return _attention(Q, K, V, Mask, **kwargs)


def attention_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Mask: torch.Tensor,
    **kwargs: object,
) -> torch.Tensor:
    """Reference decode attention with query/head routing."""

    kwargs.setdefault("is_causal", False)
    return _attention(Q, K, V, Mask, **kwargs)
