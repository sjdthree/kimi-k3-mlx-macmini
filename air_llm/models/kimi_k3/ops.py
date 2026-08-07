"""Shared MLX ops for Kimi-K3.

Each op is pinned to the source it must reproduce; see k3ref/layers.py for the torch
oracle and P0_RESULTS.md for how the KDA-side semantics were established.
"""
from __future__ import annotations

import mlx.core as mx


def rmsnorm(x: mx.array, w: mx.array, eps: float) -> mx.array:
    """KimiRMSNorm (modeling_kimi_linear.py:232). Computed in fp32."""
    xf = x.astype(mx.float32)
    return w * (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps))


def situ(gate_up: mx.array, beta: float, linear_beta: float | None) -> mx.array:
    """SituAndMul (modeling_kimi_linear.py:75).

    beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)
    """
    d = gate_up.shape[-1] // 2
    g = gate_up[..., :d].astype(mx.float32)
    u = gate_up[..., d:].astype(mx.float32)
    a = beta * mx.tanh(g / beta) * mx.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * mx.tanh(u / linear_beta)
    return a * u


def l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    """fla/modules/l2norm.py:105 -- x / sqrt(sum(x^2) + eps). Note: NOT x/(||x||+eps)."""
    return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)


def gated_rmsnorm(x: mx.array, w: mx.array, g: mx.array, eps: float) -> mx.array:
    """FusedRMSNormGated(activation='sigmoid') -- fla/modules/fused_norm_gate.py:104."""
    xf = x.astype(mx.float32)
    return (xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)) * w * mx.sigmoid(g)


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def short_conv(x: mx.array, w: mx.array, k: int, state: mx.array | None = None):
    """Causal depthwise conv (kernel k, no bias) then silu.

    x     [B, T, C]
    w     [C, k, 1]   (MLX conv1d is channels-last; the checkpoint ships [C, 1, k])
    state [B, k-1, C] carried across decode steps
    Returns (y [B, T, C], new_state [B, k-1, C]).
    """
    B, T, C = x.shape
    pad = state if state is not None else mx.zeros((B, k - 1, C), dtype=x.dtype)
    xp = mx.concatenate([pad, x], axis=1)                 # [B, T+k-1, C]
    y = mx.conv1d(xp, w, stride=1, padding=0, groups=C)   # [B, T, C]
    return silu(y), xp[:, -(k - 1):, :]


def apply_attn_res(prefix_sum: mx.array, block_residual: mx.array,
                   proj_w: mx.array, norm_w: mx.array, eps: float) -> mx.array:
    """_apply_attn_res (modeling_kimi_linear.py:1075).

    A softmax attention over the accumulated block checkpoints plus the current state --
    not a plain additive residual.

    prefix_sum [N, H]   block_residual [N, B, H]   proj_w [1, H]   norm_w [H]
    """
    v = mx.concatenate([block_residual, mx.expand_dims(prefix_sum, 1)], axis=1).astype(mx.float32)
    k = v * mx.rsqrt(mx.mean(v * v, axis=-1, keepdims=True) + eps)
    score_w = norm_w.astype(mx.float32) * proj_w.reshape(-1).astype(mx.float32)
    probs = mx.softmax(mx.sum(k * score_w, axis=-1), axis=-1)
    return mx.sum(mx.expand_dims(probs, -1) * v, axis=1)


def causal_mask(q_len: int, k_len: int) -> mx.array:
    """Additive -inf mask; queries occupy the LAST q_len positions of the key axis."""
    qi = mx.arange(k_len - q_len, k_len).reshape(-1, 1)
    ki = mx.arange(k_len).reshape(1, -1)
    return mx.where(ki > qi, mx.array(-mx.inf, mx.float32), mx.array(0.0, mx.float32))
