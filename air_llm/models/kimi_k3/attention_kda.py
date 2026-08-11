"""M5: Kimi Delta Attention for Kimi-K3, MLX. 69 of the 93 layers.

Stage 1 of the plan: a pure-MLX reference path built from standard ops. The fused Metal
kernel (M10) must match this.

Semantics (resolved in P0-2, see P0_RESULTS.md; each line pinned to upstream source):
    A_log      checkpoint ships [128]; entries 96..127 are exactly 0 in ALL 69 KDA
               layers -> a zero-padded per-value-head [96] vector. Slice [:96].
    dt_bias    [12288] -> reshape(96, 128) == (head, k_channel)
    gate       safe_gate + lower_bound=-5.0 (fla/ops/kda/gate.py):
                 g = -5.0 * sigmoid(exp(A_log)[96,1] * (g_raw + dt_bias[96,128]))
    q, k       L2-normalised (use_qk_l2norm_in_kernel)
    beta       sigmoid(b_proj(x))   (use_beta_sigmoid_in_kernel, allow_neg_eigval=False)
    recurrence fla/ops/kda/naive.py::naive_recurrent_kda, scale = head_dim**-0.5
                 S = S * exp(g)
                 S = S + (beta*k) (x) (v - (k . S))
                 o = q . S

State layout: [B, H, K, V]. The checkpoint's config passes transpose_state_layout=True,
which upstream renamed to state_v_first -- that affects serialisation order only, not the
math. We keep [B,H,K,V] internally and convert at the serialisation boundary.
"""
from __future__ import annotations

import mlx.core as mx

from .ops import gated_rmsnorm, l2norm, short_conv


class KDAAttention:
    KEYS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "g_proj.weight",
            "o_proj.weight", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
            "f_a_proj.weight", "f_b_proj.weight", "b_proj.weight", "o_norm.weight",
            "A_log", "dt_bias")

    def __init__(self, cfg, w: dict[str, mx.array]):
        self.cfg = cfg
        H, D = cfg.kda_num_heads, cfg.kda_head_dim
        self.q_w, self.k_w, self.v_w = w["q_proj.weight"], w["k_proj.weight"], w["v_proj.weight"]
        self.g_w, self.o_w = w["g_proj.weight"], w["o_proj.weight"]
        # checkpoint conv weights are [C,1,k]; MLX conv1d wants [C,k,1] (channels-last)
        self.qc = w["q_conv1d.weight"].transpose(0, 2, 1)
        self.kc = w["k_conv1d.weight"].transpose(0, 2, 1)
        self.vc = w["v_conv1d.weight"].transpose(0, 2, 1)
        self.fa, self.fb, self.b_w = w["f_a_proj.weight"], w["f_b_proj.weight"], w["b_proj.weight"]
        self.o_norm_w = w["o_norm.weight"]
        self.dt_bias = w["dt_bias"].reshape(H, D)

        A = w["A_log"]
        assert A.shape == (D,), f"A_log expected [{D}], got {tuple(A.shape)}"
        assert float(mx.max(mx.abs(A[H:]))) == 0.0, (
            "A_log[96:128] is not zero -- the zero-padding assumption from P0-2 no "
            "longer holds for this checkpoint; re-derive before trusting any output.")
        self.A_exp = mx.exp(A[:H]).reshape(H, 1)          # exp(A_log), per value head

    # How many recurrence steps to build before forcing evaluation. MLX is lazy, so
    # without this the whole T-step chain stays live as an unevaluated graph holding one
    # [B,H,D,D] state per step -- 6 MiB at 96 heads x 128 x 128 fp32. That is 1.7 GiB for
    # a 291-token prompt and 11.5 GiB for a 1,955-token one, which is how the first
    # long-prompt run blew the 48 GiB ceiling at layer 0 (a KDA layer). Decode is T=1 and
    # never reaches the check, so this costs nothing there.
    EVAL_EVERY = 64

    # ------------------------------------------------------------------ recurrence
    @staticmethod
    def _recurrent(q, k, v, g, beta, scale, state):
        """naive_recurrent_kda, fp32. q,k,v,g [B,T,H,D]; beta [B,T,H]; state [B,H,K,V]."""
        B, T, H, D = q.shape
        q = q.astype(mx.float32) * scale
        k, v, g = k.astype(mx.float32), v.astype(mx.float32), g.astype(mx.float32)
        beta = beta.astype(mx.float32)

        S = mx.zeros((B, H, D, D), dtype=mx.float32) if state is None else state.astype(mx.float32)
        outs = []
        for i in range(T):
            q_i, k_i, v_i = q[:, i], k[:, i], v[:, i]          # [B,H,D]
            g_i, b_i = g[:, i], beta[:, i]                     # [B,H,D], [B,H]

            S = S * mx.expand_dims(mx.exp(g_i), -1)            # decay: [B,H,K,1]
            # delta rule: v - (k . S)
            kS = mx.sum(mx.expand_dims(k_i, -1) * S, axis=-2)  # [B,H,V]
            upd = mx.expand_dims(mx.expand_dims(b_i, -1) * k_i, -1) * mx.expand_dims(v_i - kS, -2)
            S = S + upd
            outs.append(mx.sum(mx.expand_dims(q_i, -1) * S, axis=-2))   # [B,H,V]

            # Forcing the chain here lets MLX release every graph node behind it. Purely
            # a scheduling change -- the arithmetic and its results are untouched.
            if (i + 1) % KDAAttention.EVAL_EVERY == 0:
                mx.eval(S, *outs[-KDAAttention.EVAL_EVERY:])
        return mx.stack(outs, axis=1), S

    # ---------------------------------------------------------------------- forward
    def __call__(self, x: mx.array, state: mx.array | None = None,
                 conv: tuple | None = None):
        c = self.cfg
        B, T, _ = x.shape
        H, D, K = c.kda_num_heads, c.kda_head_dim, c.kda_conv_size

        q, cq = short_conv(x @ self.q_w.T, self.qc, K, None if conv is None else conv[0])
        k, ck = short_conv(x @ self.k_w.T, self.kc, K, None if conv is None else conv[1])
        v, cv = short_conv(x @ self.v_w.T, self.vc, K, None if conv is None else conv[2])

        q = l2norm(q.reshape(B, T, H, D))
        k = l2norm(k.reshape(B, T, H, D))
        v = v.reshape(B, T, H, D)

        g_raw = ((x @ self.fa.T) @ self.fb.T).reshape(B, T, H, D) + self.dt_bias
        g = c.kda_gate_lower_bound * mx.sigmoid(self.A_exp * g_raw)
        beta = mx.sigmoid(x @ self.b_w.T)

        o, new_state = self._recurrent(q, k, v, g, beta, D ** -0.5, state)

        og = (x @ self.g_w.T).reshape(B, T, H, D)
        o = gated_rmsnorm(o, self.o_norm_w, og, c.rms_norm_eps)
        return o.reshape(B, T, H * D) @ self.o_w.T, new_state, (cq, ck, cv)
