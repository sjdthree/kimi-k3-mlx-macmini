"""M4: Multi-head Latent Attention for Kimi-K3, MLX. 24 of the 93 layers.

Mirrors KimiMLAAttention (modeling_kimi_linear.py:405).

NoPE: `mla_use_nope=true`, `self.rotary_emb = None`, and the shipped forward never calls
apply_rotary_pos_emb -- the q_rot/k_rot slices are carried through UNROTATED. There is no
RoPE to implement for these layers. The rope-dim slice is MQA-shared: k_rot is [B,1,T,64],
conceptually broadcast across all 96 heads.

LATENT CACHING
--------------
The obvious implementation caches what attention consumes -- the expanded per-head key
and value:

    key [B,96,T,192] + value [B,96,T,128]  =  30,720 floats/token/layer

But both are a FIXED linear map of the 512-wide latent that produced them, plus the
64-wide MQA rope slice:

    c_kv [B,T,512] + k_rot [B,T,64]        =       576 floats/token/layer

so caching the latent and folding `kv_b` into the query and output projections is
algebraically identical for 53.3x less memory. Writing kv_b per head as

    k_pass[h,t] = W_k[h] @ c_t             W_k[h] = kv_b[h, :nope, :]     [128, 512]
    v[h,t]      = W_v[h] @ c_t             W_v[h] = kv_b[h, nope:, :]     [128, 512]

the two places kv_b appears both absorb:

    scores[h,i,j] = q_nope[h,i] . (W_k[h] @ c_j)  =  (W_k[h]^T @ q_nope[h,i]) . c_j
    out[h,i]      = sum_j p[i,j] (W_v[h] @ c_j)   =  W_v[h] @ (sum_j p[i,j] c_j)

Neither step approximates anything; the only difference from the expanded form is fp32
reassociation, which `tests/test_mla_cache.py` gate N3 holds to 1e-5.

The trade is memory for arithmetic: the score and context matmuls widen from 192/128 to
512. That is the right trade here by a wide margin -- decode is bound by streaming 2.8T
parameters off disk, and on master the cache had grown to 6.45 GiB at 2.3k tokens on a
machine already 27 GiB into swap.

The cache also grows in CHUNKS. Master appended with `mx.concatenate`, which reallocates
and copies the entire cache on every decode step -- O(T^2) traffic, and a transient
double allocation exactly when memory is tightest.
"""
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx

from .ops import causal_mask, rmsnorm

# Tokens of headroom added per growth. Chosen so a growth at real K3 scale moves ~14 MiB
# across all 24 MLA layers, i.e. cheap enough to be invisible next to one decode step.
CACHE_STEP = 256


def _round_up(n: int, step: int) -> int:
    return ((n + step - 1) // step) * step


@dataclass
class MLACache:
    """Latent KV cache. `latent`/`rot` are backing buffers with spare capacity; only the
    first `length` positions along axis 1 are valid.

    `latent` is normally a plain mx.array. When `mla_cache_bits` is configured it is a
    mla_quant.QuantLatent instead, which exposes the same write/grow/read surface -- see
    that module for why quantisation is off by default.
    """
    latent: object            # mx.array [B, cap, kv_lora_rank] | QuantLatent
    rot: mx.array             # [B, capacity, qk_rope_head_dim]
    length: int = 0

    @property
    def quantised(self) -> bool:
        return not isinstance(self.latent, mx.array)

    @property
    def capacity(self) -> int:
        return self.latent.capacity if self.quantised else self.latent.shape[1]

    def view(self) -> tuple[mx.array, mx.array]:
        """The valid prefix, as fp32 for the attention math."""
        lat = (self.latent.read(self.length) if self.quantised
               else self.latent[:, :self.length].astype(mx.float32))
        return lat, self.rot[:, :self.length].astype(mx.float32)

    def append(self, c_new: mx.array, r_new: mx.array) -> "MLACache":
        """Write T new positions, growing the buffer only when a chunk fills."""
        T = c_new.shape[1]
        need = self.length + T
        if need > self.capacity:
            # Double (with a floor) so total copy stays amortised O(1) per token rather
            # than O(T) per growth * O(T/step) growths.
            cap = _round_up(max(need, 2 * self.capacity, CACHE_STEP), CACHE_STEP)
            if self.quantised:
                self.latent.grow(cap, self.length)
            else:
                self.latent = _grown(self.latent, cap, self.length)
            self.rot = _grown(self.rot, cap, self.length)
        if self.quantised:
            self.latent.write(self.length, c_new)
        else:
            self.latent[:, self.length:need] = c_new.astype(self.latent.dtype)
        self.rot[:, self.length:need] = r_new.astype(self.rot.dtype)
        self.length = need
        return self

    def bytes_per_token(self) -> float:
        lat = (self.latent.bytes_per_token() if self.quantised
               else self.latent.shape[2] * self.latent.itemsize)
        return lat + self.rot.shape[2] * self.rot.itemsize

    def nbytes_used(self) -> int:
        """Bytes for the VALID prefix -- what the cache would cost if trimmed."""
        return int(self.rot.shape[0] * self.length * self.bytes_per_token())


def _grown(buf: mx.array, cap: int, keep: int) -> mx.array:
    out = mx.zeros((buf.shape[0], cap, buf.shape[2]), dtype=buf.dtype)
    if keep:
        out[:, :keep] = buf[:, :keep]
    return out


def _new_cache(B: int, klr: int, rope: int, T: int, dtype, bits, group_size) -> MLACache:
    cap = _round_up(max(T, CACHE_STEP), CACHE_STEP)
    if bits:
        from .mla_quant import QuantLatent, pick_group_size
        lat = QuantLatent(B, cap, klr, bits, group_size or pick_group_size(klr))
    else:
        lat = mx.zeros((B, cap, klr), dtype=dtype)
    return MLACache(lat, mx.zeros((B, cap, rope), dtype=dtype), 0)


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

        # kv_b is [H*(nope+vhd), kv_lora_rank]; split it into the two absorbable halves.
        # Kept in the CHECKPOINT dtype (bf16) and promoted by the matmuls, exactly as
        # master's `rmsnorm(...) @ kv_b.T` did. Upcasting here instead would copy 50 MiB
        # per MLA layer per token -- 1.2 GiB of transient churn per token across 24
        # layers -- because streaming rebuilds the layer on every step.
        nh, nope = cfg.num_attention_heads, cfg.qk_nope_head_dim
        kb = self.kv_b.reshape(nh, nope + cfg.v_head_dim, cfg.kv_lora_rank)
        self.wk = mx.expand_dims(kb[:, :nope, :], 0)          # [1,H,nope,r]
        self.wv = mx.expand_dims(kb[:, nope:, :], 0)          # [1,H,vhd,r]

        # fp32 by default: latent caching is exactly lossless, and nothing here needs to
        # spend that. `mla_cache_dtype` on the config opts into bf16 (a further 2x), and
        # `mla_cache_bits` into quantised storage (see mla_quant.py -- off by default).
        self.cache_dtype = getattr(cfg, "mla_cache_dtype", mx.float32)
        self.cache_bits = getattr(cfg, "mla_cache_bits", None)
        self.cache_group = getattr(cfg, "mla_cache_group_size", None)

    def __call__(self, x: mx.array, cache: MLACache | None = None):
        c = self.cfg
        B, T, _ = x.shape
        nh, qhd, vhd = c.num_attention_heads, c.q_head_dim, c.v_head_dim
        nope, rope, klr = c.qk_nope_head_dim, c.qk_rope_head_dim, c.kv_lora_rank
        eps = c.rms_norm_eps

        # ---- queries: hidden -> q_lora -> RMSNorm -> heads
        q = rmsnorm(x @ self.q_a.T, self.q_an, eps) @ self.q_b.T
        q = q.reshape(B, T, nh, qhd).transpose(0, 2, 1, 3)          # [B,H,T,qhd]
        q_nope, q_rot = q[..., :nope], q[..., nope:]

        # ---- keys/values: hidden -> (kv_lora | rope) ; kv_lora -> RMSNorm.
        # kv_b is NOT applied -- it is absorbed into the query and output paths below.
        ckv = x @ self.kv_a.T                                        # [B,T,klr+rope]
        c_new = rmsnorm(ckv[..., :klr], self.kv_an, eps)             # [B,T,klr]
        r_new = ckv[..., klr:]                                       # [B,T,rope]

        if cache is None:
            cache = _new_cache(B, klr, rope, T, self.cache_dtype,
                               self.cache_bits, self.cache_group)
        cache.append(c_new, r_new)
        lat, rot = cache.view()                                      # [B,Tk,klr], [B,Tk,rope]
        Tk = cache.length

        # ---- scores, absorbed. q_nope @ W_k^T lands the query in latent space, so the
        # nope half of the dot product runs against the cached latent directly.
        qa = q_nope @ self.wk                                        # [B,H,T,klr]
        s = qa @ mx.expand_dims(lat.transpose(0, 2, 1), 1)           # [B,H,T,Tk]
        # rope half is MQA-shared: one [B,1,rope,Tk] operand for all heads
        s = s + q_rot @ mx.expand_dims(rot.transpose(0, 2, 1), 1)
        s = s * (qhd ** -0.5)
        if T > 1:
            # Decode (T == 1) needs no mask: causal_mask places the queries at the LAST
            # T positions, so a lone query sits at Tk-1 and every cached position is
            # visible -- the mask is all zeros. Skipping it drops an allocation and a
            # [B,H,1,Tk] add per MLA layer per token. Gate N2 (incremental decode ==
            # one-shot) is what proves this is not a masking bug.
            s = s + causal_mask(T, Tk)

        probs = mx.softmax(s.astype(mx.float32), axis=-1)

        # ---- output, absorbed. Mix the latents first, then project once per head.
        ctx = probs @ mx.expand_dims(lat, 1)                         # [B,H,T,klr]
        o = (ctx @ self.wv.transpose(0, 1, 3, 2)).transpose(0, 2, 1, 3).reshape(B, T, nh * vhd)

        if c.mla_use_output_gate:
            o = o * mx.sigmoid(x @ self.g_w.T)
        return o @ self.o_w.T, cache
