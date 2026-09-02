"""M4 + M5 gates: MLX attention must match the k3ref torch/CPU oracle on REAL weights.

Spec section 16 tolerances for fp32 validation mode: atol/rtol <= 1e-5.
For KDA, spec section 9 requires comparing STATE after every token, not just logits.

    python tests/test_attention_mlx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import require_checkpoint  # noqa: E402
from air_llm.models.kimi_k3.attention_kda import KDAAttention as KDAmlx  # noqa: E402
from air_llm.models.kimi_k3.attention_mla import MLAAttention as MLAmlx  # noqa: E402
from k3ref.layers import K3Config, KDAAttention as KDAref, MLAAttention as MLAref  # noqa: E402
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0
ATOL = 1e-5


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def cmp(a_mx, b_torch, label, atol=ATOL):
    a = np.array(a_mx, copy=False).astype(np.float64)
    b = b_torch.detach().numpy().astype(np.float64)
    ad = np.abs(a - b).max()
    scale = max(np.abs(b).max(), 1e-30)
    check(label, ad <= atol, f"max|d| {ad:.3e}  rel {ad/scale:.2e}")
    return ad


def load_mx(ws: WeightStore, layer: int, keys) -> dict:
    out = {}
    for k in keys:
        t = ws.get(f"language_model.model.layers.{layer}.self_attn.{k}")
        out[k] = mx.array(np.ascontiguousarray(t.numpy()))
    return out


# ------------------------------------------------------------------------ M4: MLA

def test_mla():
    cfg = K3Config.load()
    ws = WeightStore()
    L = cfg.mla_layers[0]
    print(f"\nM4 -- MLA, real layer {L}")
    t0 = time.time()
    ref = MLAref(cfg, L, ws)
    mlx = MLAmlx(cfg, load_mx(ws, L, MLAmlx.KEYS))
    print(f"  (loaded both in {time.time()-t0:.1f}s)")

    torch.manual_seed(0)
    for T in (1, 2, 8):
        x = torch.randn(1, T, cfg.hidden_size) * 0.02
        r, _ = ref(x)
        m, _ = mlx(mx.array(x.numpy()))
        cmp(m, r, f"seqlen {T} prefill")

    # cached decode: 4-token prefill then 4 single-token steps
    x = torch.randn(1, 8, cfg.hidden_size) * 0.02
    r_full, _ = ref(x)
    m_pre, mc = mlx(mx.array(x[:, :4].numpy()))
    outs = [m_pre]
    for i in range(4, 8):
        o, mc = mlx(mx.array(x[:, i:i + 1].numpy()), cache=mc)
        outs.append(o)
    m_cat = mx.concatenate(outs, axis=1)
    cmp(m_cat, r_full, "prefill(4) + 4 cached decode steps == prefill(8)", atol=1e-4)
    check("KV cache grew to the full sequence", mc.length == 8, f"len {mc.length}")


# ------------------------------------------------------------------------ M5: KDA

def test_kda():
    cfg = K3Config.load()
    ws = WeightStore()
    L = cfg.kda_layers[1]          # layer 1: KDA + MoE
    print(f"\nM5 -- KDA, real layer {L}")
    t0 = time.time()
    ref = KDAref(cfg, L, ws)
    mlx = KDAmlx(cfg, load_mx(ws, L, KDAmlx.KEYS))
    print(f"  (loaded both in {time.time()-t0:.1f}s)")

    torch.manual_seed(0)
    for T in (1, 2, 8):
        x = torch.randn(1, T, cfg.hidden_size) * 0.02
        r_o, r_s, _ = ref(x)
        m_o, m_s, _ = mlx(mx.array(x.numpy()))
        cmp(m_o, r_o, f"seqlen {T} output")
        cmp(m_s, r_s, f"seqlen {T} final state")

    # spec section 9: compare STATE AFTER EVERY TOKEN, not just the final one
    T = 8
    x = torch.randn(1, T, cfg.hidden_size) * 0.02
    r_state = r_conv = None
    m_state = m_conv = None
    worst_o = worst_s = 0.0
    for i in range(T):
        xi = x[:, i:i + 1]
        r_o, r_state, r_conv = ref(xi, state=r_state, conv=r_conv)
        m_o, m_state, m_conv = mlx(mx.array(xi.numpy()), state=m_state, conv=m_conv)
        worst_o = max(worst_o, np.abs(np.array(m_o, copy=False) - r_o.numpy()).max())
        worst_s = max(worst_s, np.abs(np.array(m_state, copy=False) - r_state.numpy()).max())
    check("per-token recurrent state matches at EVERY token (spec section 9)",
          worst_s <= ATOL and worst_o <= ATOL,
          f"worst |d state| {worst_s:.3e}  worst |d out| {worst_o:.3e}")

    # Conv state must match too, else decode drifts after 3 tokens.
    # NOTE: layouts differ by backend convention -- MLX is channels-last [B, k-1, C],
    # torch is channels-first [B, C, k-1]. Same values, transposed. The state
    # serialiser (spec section 12) must normalise this or saved state will not port.
    check("conv-state layouts differ as expected (MLX channels-last vs torch)",
          m_conv[0].shape == (1, cfg.kda_conv_size - 1, r_conv[0].shape[1])
          and r_conv[0].shape == (1, m_conv[0].shape[2], cfg.kda_conv_size - 1),
          f"mlx {tuple(m_conv[0].shape)} vs torch {tuple(r_conv[0].shape)}")
    dconv = max(np.abs(np.array(m, copy=False).transpose(0, 2, 1) - r.numpy()).max()
                for m, r in zip(m_conv, r_conv))
    check("short-conv state matches after 8 decode steps (layout-normalised)",
          dconv <= ATOL, f"max|d| {dconv:.3e}")


def test_alog_guard():
    """The A_log padding assumption must be enforced, not assumed."""
    cfg = K3Config.load()
    ws = WeightStore()
    w = load_mx(ws, cfg.kda_layers[1], KDAmlx.KEYS)
    bad = dict(w)
    a = np.array(w["A_log"], copy=True)
    a[100] = 0.5                                   # poison the padding region
    bad["A_log"] = mx.array(a)
    try:
        KDAmlx(cfg, bad)
        check("KDA rejects a checkpoint whose A_log padding is non-zero", False)
    except AssertionError:
        check("KDA rejects a checkpoint whose A_log padding is non-zero", True)


if __name__ == "__main__":
    require_checkpoint()
    print(f"M4/M5 gates: MLX attention vs k3ref   (device {mx.default_device()})")
    print("=" * 68)
    test_mla()
    test_kda()
    test_alog_guard()
    print("=" * 68)
    print(f"M4/M5: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    sys.exit(1 if FAILS else 0)
