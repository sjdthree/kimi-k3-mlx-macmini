"""P0-4: run one REAL Kimi-K3 KDA layer end-to-end, fp32, on CPU.

Semantics established in P0-2, all pinned to upstream source:
  A_log      : checkpoint ships [128]; entries 96..127 are exactly 0 in all 69 KDA
               layers -> it is a zero-padded per-value-head [96] vector. Slice [:96].
  dt_bias    : [12288] -> view(96, 128) == (head, k_channel)
  gate       : safe_gate + lower_bound=-5.0 =>
                 g = -5.0 * sigmoid(exp(A_log) [96,1] * (g_raw + dt_bias[96,128]))
               (fla/ops/kda/gate.py: naive_kda_lowerbound_gate)
  q,k        : L2-normalised, x / sqrt(sum(x^2) + 1e-6)   (fla/modules/l2norm.py:105)
  beta       : sigmoid(b_proj(x))    (use_beta_sigmoid_in_kernel, allow_neg_eigval=False)
  short conv : causal depthwise k=4, then silu, no bias
  o_norm     : gated RMSNorm, y = x*rstd*w*sigmoid(g)  (fla/modules/fused_norm_gate.py:104)
  recurrence : fla/ops/kda/naive.py::naive_recurrent_kda, scale = K**-0.5

Provenance: the formulas above re-implement operations from flash-linear-attention (MIT)
and Moonshot's modeling_kimi_linear.py (Kimi K3 License), at the cited source anchors.
The recurrence loads the unmodified FLA kernel vendored in k3ref/_vendor/. See NOTICE and
docs/licensing.md.
"""
import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from k3io import load

# The upstream reference, vendored unmodified. This spike originally loaded it from a
# scratch checkout of flash-linear-attention; it now points at the in-repo copy.
FLA_NAIVE = Path(__file__).resolve().parents[2] / "k3ref" / "_vendor" / "kda_naive.py"

# import upstream naive.py directly (bypasses fla/__init__, which needs triton)
_spec = importlib.util.spec_from_file_location("kda_naive", str(FLA_NAIVE))
kda_naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kda_naive)

H, D, HIDDEN = 96, 128, 7168        # num_v_heads, head_dim, hidden_size
CONV_K, LOWER_BOUND, EPS = 4, -5.0, 1e-5


def t(name, L):
    a = load(f"language_model.model.layers.{L}.self_attn.{name}")
    return torch.from_numpy(np.ascontiguousarray(a)).float()


def l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def short_conv(x, w, state=None):
    """x [B,T,C]; w [C,1,K]. Causal depthwise conv + silu. Returns (y, new_state[B,C,K-1])."""
    B, T, C = x.shape
    xt = x.transpose(1, 2)                                    # [B,C,T]
    pad = state if state is not None else xt.new_zeros(B, C, CONV_K - 1)
    xp = torch.cat([pad, xt], dim=-1)
    y = F.conv1d(xp, w, groups=C)                             # [B,C,T]
    return F.silu(y.transpose(1, 2)), xp[..., -(CONV_K - 1):]


def kda_gate(g_raw, A_log, dt_bias):
    """g_raw [B,T,H,D] -> log-decay. naive_kda_lowerbound_gate with lower_bound=-5.0."""
    g = g_raw + dt_bias.view(H, D)
    return LOWER_BOUND * torch.sigmoid(A_log.view(H, 1).exp() * g)


def gated_rmsnorm(x, w, g, eps=EPS):
    """y = x * rstd * w * sigmoid(g), rstd = 1/sqrt(mean(x^2)+eps). Last dim = D."""
    rstd = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x * rstd) * w * torch.sigmoid(g)


class KDALayer:
    def __init__(self, L):
        self.L = L
        self.q_w, self.k_w, self.v_w = t("q_proj.weight", L), t("k_proj.weight", L), t("v_proj.weight", L)
        self.g_w, self.o_w = t("g_proj.weight", L), t("o_proj.weight", L)
        self.qc, self.kc, self.vc = t("q_conv1d.weight", L), t("k_conv1d.weight", L), t("v_conv1d.weight", L)
        self.fa, self.fb = t("f_a_proj.weight", L), t("f_b_proj.weight", L)
        self.b_w = t("b_proj.weight", L)
        self.o_norm_w = t("o_norm.weight", L)
        A = t("A_log", L)
        assert A.shape == (128,) and A[96:].abs().max() == 0, "A_log padding assumption broken"
        self.A_log = A[:H]                                    # <-- the P0-2 finding
        self.dt_bias = t("dt_bias", L)

    def forward(self, x, state=None, conv=None):
        B, T, _ = x.shape
        q, cq = short_conv(x @ self.q_w.T, self.qc, None if conv is None else conv[0])
        k, ck = short_conv(x @ self.k_w.T, self.kc, None if conv is None else conv[1])
        v, cv = short_conv(x @ self.v_w.T, self.vc, None if conv is None else conv[2])

        q = l2norm(q.view(B, T, H, D))
        k = l2norm(k.view(B, T, H, D))
        v = v.view(B, T, H, D)

        g_raw = (x @ self.fa.T) @ self.fb.T
        g = kda_gate(g_raw.view(B, T, H, D), self.A_log, self.dt_bias)
        beta = torch.sigmoid(x @ self.b_w.T)                  # [B,T,H]

        o, new_state = kda_naive.naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta,
            scale=D ** -0.5, initial_state=state, output_final_state=True)

        og = (x @ self.g_w.T).view(B, T, H, D)
        o = gated_rmsnorm(o, self.o_norm_w, og)
        return (o.reshape(B, T, H * D) @ self.o_w.T), new_state, (cq, ck, cv)


def ck(x):
    a = np.ascontiguousarray(x.detach().numpy().astype(np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


if __name__ == "__main__":
    torch.manual_seed(0)
    L, T = 1, 8
    print(f"loading real weights for KDA layer {L} ...", flush=True)
    layer = KDALayer(L)
    print(f"  A_log[:96] range [{layer.A_log.min():+.4f}, {layer.A_log.max():+.4f}]  "
          f"exp() in [{layer.A_log.exp().min():.3f}, {layer.A_log.exp().max():.3f}]")

    # hidden state at a realistic RMS scale for a post-layernorm input
    x = torch.randn(1, T, HIDDEN) * 0.02

    o1, s1, c1 = layer.forward(x)
    o2, s2, c2 = layer.forward(x)

    print(f"\n  output {tuple(o1.shape)}   state {tuple(s1.shape)}")
    print(f"  finite: out={bool(torch.isfinite(o1).all())} state={bool(torch.isfinite(s1).all())}")
    print(f"  out   |mean| {o1.abs().mean():.6e}  std {o1.std():.6e}  absmax {o1.abs().max():.6e}")
    print(f"  state |mean| {s1.abs().mean():.6e}  std {s1.std():.6e}  absmax {s1.abs().max():.6e}")
    print(f"  determinism: out {ck(o1)} == {ck(o2)} -> {ck(o1)==ck(o2)} | "
          f"state {ck(s1)} == {ck(s2)} -> {ck(s1)==ck(s2)}")

    # prefill(T) must equal T sequential single-token steps (the decode path we will ship)
    st, cv = None, None
    outs = []
    for i in range(T):
        oi, st, cv = layer.forward(x[:, i:i + 1], state=st, conv=cv)
        outs.append(oi)
    o_step = torch.cat(outs, dim=1)
    do = (o_step - o1).abs().max().item()
    ds = (st - s1).abs().max().item()
    print(f"\n  prefill vs {T}x single-token decode:")
    print(f"    max|d out|   {do:.3e}   (rel {do/o1.abs().max().item():.2e})")
    print(f"    max|d state| {ds:.3e}   (rel {ds/s1.abs().max().item():.2e})")

    ok = (torch.isfinite(o1).all() and torch.isfinite(s1).all()
          and ck(o1) == ck(o2) and ck(s1) == ck(s2) and do < 1e-4 and ds < 1e-4)
    print(f"\nP0-4 {'PASS' if ok else 'FAIL'}")
