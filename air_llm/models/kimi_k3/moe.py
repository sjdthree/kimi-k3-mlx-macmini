"""M6: Kimi-K3 router + latent MoE + serial expert streaming, MLX.

Mirrors KimiMoEGate (modeling_kimi_linear.py:703) and KimiSparseMoeBlock (:815).

The LATENT wrapper is the part the original spec omitted (K3_MLX_PLAN.md section 1 row 2).
Experts do not act on the 7168-dim hidden state; they act in a 3584-dim latent space:

    hidden(7168) --routed_expert_down_proj--> latent(3584)
                 --routed_expert_norm (RMSNorm)-->
                 --top-16 experts, w1/w3: 3584->3072, w2: 3072->3584-->
                 --routed_expert_up_proj--> hidden(7168)
                 + shared_experts(ORIGINAL hidden, not latent)

Two router details are easy to get backwards and both silently corrupt every MoE output:
  * e_score_correction_bias affects SELECTION ONLY; the returned weights are gathered
    from the RAW sigmoid scores.
  * renormalisation divides by the sum of those raw top-k scores (+1e-20), then scales
    by routed_scaling_factor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mlx.core as mx

from ...kernels.metal.mxfp4_matvec import mxfp4_matvec_fused
from .mxfp4 import decode_mxfp4_mlx
from .ops import rmsnorm, situ


@dataclass
class RouterResult:
    expert_indices: mx.array      # [N, top_k]
    expert_weights: mx.array      # [N, top_k]
    raw_scores: mx.array          # [N, num_experts] = sigmoid(logits)
    corrected_scores: mx.array    # raw + e_score_correction_bias


class LatentMoE:
    """`expert_source(layer, expert, which) -> (packed, scales)` streams one projection.

    Keeping the source behind a callable is what lets the runtime load-compute-evict a
    single expert at a time without this class knowing about disk, mmap, or caching.
    """

    def __init__(self, cfg, layer: int, w: dict[str, mx.array],
                 expert_source: Callable[[int, int, str], tuple[mx.array, mx.array]],
                 fused: bool = True):
        self.cfg, self.L = cfg, layer
        self.src = expert_source
        # Spec section 13: validation mode disables fused experimental kernels. Note the
        # fused path is also the MORE accurate one (exact weights + fp32 accumulate vs
        # bf16 weights unfused), so `fused=False` is slower AND looser -- it exists to
        # isolate the kernel during debugging, not because it is the safer default.
        self.fused = fused
        self.gate_w, self.gate_b = w["gate.weight"], w["gate.e_score_correction_bias"]
        self.down, self.up = w["routed_expert_down_proj.weight"], w["routed_expert_up_proj.weight"]
        self.norm_w = w["routed_expert_norm.weight"]
        self.sh_gate = w["shared_experts.gate_proj.weight"]
        self.sh_up = w["shared_experts.up_proj.weight"]
        self.sh_down = w["shared_experts.down_proj.weight"]

    # -------------------------------------------------------------------- routing
    def route(self, x: mx.array) -> RouterResult:
        c = self.cfg
        flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        scores = mx.sigmoid(flat @ self.gate_w.astype(mx.float32).T)
        corrected = scores + self.gate_b

        k = c.num_experts_per_token
        idx = mx.argpartition(-corrected, k - 1, axis=-1)[:, :k]
        w = mx.take_along_axis(scores, idx, axis=-1)        # RAW scores, not corrected
        if k > 1 and c.moe_renormalize:
            w = w / (mx.sum(w, axis=-1, keepdims=True) + 1e-20)
        return RouterResult(idx, w * c.routed_scaling_factor, scores, corrected)

    # -------------------------------------------------------------------- experts
    def _proj(self, h: mx.array, e: int, which: str) -> mx.array:
        """h @ W.T for one MXFP4 projection.

        Fused: reads the 5.25 MiB packed row set once and unpacks in-register -- the full
        matrix is never materialised. Unfused: expands to bf16 first (4x the traffic).
        """
        p, s = self.src(self.L, e, which)
        if self.fused:
            return mxfp4_matvec_fused(h, p, s)
        return h @ decode_mxfp4_mlx(p, s, out_dtype=mx.bfloat16).T.astype(h.dtype)

    def expert_forward(self, h: mx.array, e: int) -> mx.array:
        """One routed expert, in latent space. Weights are consumed then dropped.

        Ordered so w1/w3 are released before w2 is touched -- only one projection is
        resident at a time (spec section 10: never a full persistent copy of an expert).
        """
        c = self.cfg
        gate = self._proj(h, e, "w1")
        up = self._proj(h, e, "w3")
        act = situ(mx.concatenate([gate, up], axis=-1), c.situ_beta, c.situ_linear_beta)
        del gate, up
        return self._proj(act, e, "w2")

    def shared_forward(self, x: mx.array) -> mx.array:
        gu = mx.concatenate([x @ self.sh_gate.T, x @ self.sh_up.T], axis=-1)
        return situ(gu, self.cfg.situ_beta, self.cfg.situ_linear_beta) @ self.sh_down.T

    # -------------------------------------------------------------------- forward
    def __call__(self, x: mx.array, return_router: bool = False):
        c = self.cfg
        orig = x.shape
        r = self.route(x)
        flat = x.reshape(-1, orig[-1])

        h = flat @ self.down.T                                  # hidden -> latent
        y = mx.zeros((flat.shape[0], c.routed_expert_hidden_size), dtype=mx.float32)

        idx = np_indices(r.expert_indices)
        for e in sorted({int(v) for row in idx for v in row}):
            tok = [t for t, row in enumerate(idx) if e in row]
            slot = [list(idx[t]).index(e) for t in tok]
            ti = mx.array(tok)
            out = self.expert_forward(h[ti], e)
            wts = mx.take_along_axis(r.expert_weights[ti], mx.array(slot).reshape(-1, 1), axis=-1)
            y = y.at[ti].add(out * wts)
            mx.eval(y)                                          # force release of expert bufs

        y = rmsnorm(y, self.norm_w, c.rms_norm_eps)             # latent_moe_use_norm
        y = (y @ self.up.T).reshape(*orig)                      # latent -> hidden
        y = y + self.shared_forward(x)                          # shared act on ORIGINAL hidden
        return (y, r) if return_router else y


def np_indices(a: mx.array):
    """Materialise routing indices host-side; they drive control flow, not math."""
    import numpy as np
    return np.array(a, copy=False).tolist()
