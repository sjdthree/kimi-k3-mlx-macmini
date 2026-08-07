"""M4: Multi-head Latent Attention for Kimi-K3, MLX. 24 of the 93 layers.

Mirrors KimiMLAAttention (modeling_kimi_linear.py:405).

NoPE: `mla_use_nope=true`, `self.rotary_emb = None`, and the shipped forward never calls
apply_rotary_pos_emb -- the q_rot/k_rot slices are carried through UNROTATED. There is no
RoPE to implement for these layers. The rope-dim slice is still MQA-shared: k_rot is
[B,1,T,64] broadcast across all 96 heads.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .ops import causal_mask, rmsnorm


@dataclass
class MLACache:
    key: mx.array          # [B, H, T, q_head_dim]
    value: mx.array        # [B, H, T, v_head_dim]

    @property
    def length(self) -> int:
        return self.key.shape[2]


class MLAAttention:
    """Weights are the checkpoint tensors as-is: [out, in], applied as x @ W.T."""

    KEYS = ("q_a_proj.weight", "q_a_layernorm.weight", "q_b_proj.weight",
            "kv_a_proj_with_mqa.weight", "kv_a_layernorm.weight", "kv_b_proj.weight",
            "g_proj.weight", "o_proj.weight")

    def __init__(self, cfg, w: dict[str, mx.array]):
        self.cfg = cfg
        self.q_a, self.q_an, self.q_b = w["q_a_proj.weight"], w["q_a_layernorm.weight"], w["q_b_proj.weight"]
        self.kv_a, self.kv_an = w["kv_a_proj_with_mqa.weight"], w["kv_a_layernorm.weight"]
        self.kv_b = w["kv_b_proj.weight"]
        self.g_w, self.o_w = w["g_proj.weight"], w["o_proj.weight"]

    def __call__(self, x: mx.array, cache: MLACache | None = None):
        c = self.cfg
        B, T, _ = x.shape
        nh, qhd, vhd = c.num_attention_heads, c.q_head_dim, c.v_head_dim
        nope, rope = c.qk_nope_head_dim, c.qk_rope_head_dim
        eps = c.rms_norm_eps

        # ---- queries: hidden -> q_lora -> RMSNorm -> heads
        q = rmsnorm(x @ self.q_a.T, self.q_an, eps) @ self.q_b.T
        q = q.reshape(B, T, nh, qhd).transpose(0, 2, 1, 3)          # [B,H,T,qhd]

        # ---- keys/values: hidden -> (kv_lora | rope) ; kv_lora -> RMSNorm -> heads
        ckv = x @ self.kv_a.T                                        # [B,T,kv_lora+rope]
        k_lat, k_rot = ckv[..., :c.kv_lora_rank], ckv[..., c.kv_lora_rank:]
        kv = rmsnorm(k_lat, self.kv_an, eps) @ self.kv_b.T
        kv = kv.reshape(B, T, nh, nope + vhd).transpose(0, 2, 1, 3)  # [B,H,T,nope+vhd]
        k_pass, v = kv[..., :nope], kv[..., nope:]

        # rope slice is MQA-shared across heads and left unrotated (NoPE)
        k_rot = mx.broadcast_to(k_rot.reshape(B, 1, T, rope), (B, nh, T, rope))
        key = mx.concatenate([k_pass, k_rot], axis=-1)               # [B,H,T,qhd]

        if cache is not None:
            key = mx.concatenate([cache.key, key], axis=2)
            v = mx.concatenate([cache.value, v], axis=2)
        new_cache = MLACache(key, v)

        scores = (q @ key.transpose(0, 1, 3, 2)) * (qhd ** -0.5)
        scores = scores + causal_mask(T, key.shape[2])
        probs = mx.softmax(scores.astype(mx.float32), axis=-1)
        o = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * vhd)

        if c.mla_use_output_gate:
            o = o * mx.sigmoid(x @ self.g_w.T)
        return o @ self.o_w.T, new_cache
