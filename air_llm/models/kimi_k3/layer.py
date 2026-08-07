"""M7: full Kimi-K3 decoder layer in MLX, including the block attention residuals.

Mirrors KimiDecoderLayer._forward_attn_residual (modeling_kimi_linear.py:973).

The attention-residual structure is the part most easily got wrong, so spelled out:

    prefix_sum = x
    if block_residual is non-empty:
        x = attn_res(prefix_sum, block_residual, self_attention_res_{proj,norm})
    if layer_idx % attn_res_block_size == 0:          # 0, 12, 24, ... 84  -> 8 checkpoints
        block_residual = concat(block_residual, prefix_sum)
        prefix_sum = None                             # this layer OPENS a new block
    x = input_layernorm(x)
    x = self_attn(x)
    prefix_sum = x if prefix_sum is None else prefix_sum + x
    x = attn_res(prefix_sum, block_residual, mlp_res_{proj,norm})
    x = post_attention_layernorm(x)
    x = moe(x) | mlp(x)
    prefix_sum = prefix_sum + x
    return prefix_sum, block_residual

Note `attn_res` is a softmax attention over the accumulated block checkpoints plus the
current running sum -- not an additive residual. `*_res_proj` is [1, hidden], i.e. it
emits ONE scalar logit per token per checkpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx

from .attention_kda import KDAAttention
from .attention_mla import MLAAttention, MLACache
from .moe import LatentMoE
from .ops import apply_attn_res, rmsnorm, situ


@dataclass
class LayerState:
    """Everything a single layer must carry between tokens."""
    kda_state: mx.array | None = None
    kda_conv: tuple | None = None
    mla_cache: MLACache | None = None


@dataclass
class LayerDiag:
    router_indices: mx.array | None = None
    router_weights: mx.array | None = None
    attn_checksum: float | None = None
    moe_checksum: float | None = None
    out_checksum: float | None = None
    experts_loaded: int = 0


class DenseMLP:
    """Layer 0 only (first_k_dense_replace=1). Unquantised bf16."""

    def __init__(self, cfg, w: dict[str, mx.array]):
        self.cfg = cfg
        self.gate, self.up, self.down = w["gate_proj.weight"], w["up_proj.weight"], w["down_proj.weight"]

    def __call__(self, x, return_router=False):
        gu = mx.concatenate([x @ self.gate.T, x @ self.up.T], axis=-1)
        y = situ(gu, self.cfg.situ_beta, self.cfg.situ_linear_beta) @ self.down.T
        return (y, None) if return_router else y


class DecoderLayer:
    NORM_KEYS = ("input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attention_res_norm.weight", "self_attention_res_proj.weight",
                 "mlp_res_norm.weight", "mlp_res_proj.weight")

    def __init__(self, cfg, layer: int, norms: dict[str, mx.array],
                 attn, mlp):
        self.cfg, self.L = cfg, layer
        self.is_kda = cfg.is_kda(layer)
        self.attn, self.mlp = attn, mlp
        self.in_norm = norms["input_layernorm.weight"]
        self.post_norm = norms["post_attention_layernorm.weight"]
        self.ares_norm = norms["self_attention_res_norm.weight"]
        self.ares_proj = norms["self_attention_res_proj.weight"]
        self.mres_norm = norms["mlp_res_norm.weight"]
        self.mres_proj = norms["mlp_res_proj.weight"]

    def __call__(self, x: mx.array, block_residual: mx.array,
                 state: LayerState | None = None, diag: LayerDiag | None = None):
        c = self.cfg
        B, T, H = x.shape
        eps = c.rms_norm_eps
        st = state or LayerState()
        prefix_sum = x

        if block_residual.shape[1] > 0:
            x = apply_attn_res(prefix_sum.reshape(-1, H), block_residual,
                               self.ares_proj, self.ares_norm, eps).reshape(B, T, H)

        opens_block = (self.L % c.attn_res_block_size == 0)
        if opens_block:
            block_residual = mx.concatenate(
                [block_residual, mx.expand_dims(prefix_sum.reshape(-1, H), 1)], axis=1)
            prefix_sum = None

        x = rmsnorm(x, self.in_norm, eps)
        if self.is_kda:
            x, st.kda_state, st.kda_conv = self.attn(x, state=st.kda_state, conv=st.kda_conv)
        else:
            x, st.mla_cache = self.attn(x, cache=st.mla_cache)

        if diag is not None:
            diag.attn_checksum = float(mx.sum(x.astype(mx.float32)))

        prefix_sum = x if prefix_sum is None else prefix_sum + x

        x = apply_attn_res(prefix_sum.reshape(-1, H), block_residual,
                           self.mres_proj, self.mres_norm, eps).reshape(B, T, H)
        x = rmsnorm(x, self.post_norm, eps)
        x, router = self.mlp(x, return_router=True)

        if diag is not None:
            diag.moe_checksum = float(mx.sum(x.astype(mx.float32)))
            if router is not None:
                diag.router_indices = router.expert_indices
                diag.router_weights = router.expert_weights
                diag.experts_loaded = int(mx.max(router.expert_indices).item() >= 0) * \
                    router.expert_indices.size

        prefix_sum = prefix_sum + x
        if diag is not None:
            diag.out_checksum = float(mx.sum(prefix_sum.astype(mx.float32)))
        return prefix_sum, block_residual, st


def empty_block_residual(num_tokens: int, hidden: int) -> mx.array:
    """KimiLinearModel.forward:1190 -- starts as [N, 0, H]."""
    return mx.zeros((num_tokens, 0, hidden), dtype=mx.float32)
