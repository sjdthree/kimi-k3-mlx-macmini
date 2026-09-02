"""M2 gate: run real Kimi-K3 layers through the k3ref oracle.

Covers layer 0 (dense MLP + KDA), 1-2 (KDA + latent MoE), 3 (MLA + latent MoE), which is
the smallest chain that exercises: the attention-residual block open at layer 0, KDA and
MLA, the latent MoE wrapper, serial MXFP4 expert streaming, and shared experts.

    python tests/test_k3ref_layers.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import require_checkpoint  # noqa: E402
from k3ref.layers import (K3Config, DecoderLayer, apply_attn_res, situ,  # noqa: E402
                          rmsnorm, l2norm)
from k3ref.weights import WeightStore, decode_mxfp4  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def ck(x):
    return hashlib.sha256(x.detach().float().numpy().tobytes()).hexdigest()[:16]


# ------------------------------------------------------------------ unit-ish checks

def test_situ():
    """Independent transcription of SituAndMul.forward (modeling_kimi_linear.py:75-82)."""
    torch.manual_seed(0)
    x = torch.randn(4, 64)
    beta, lb = 4.0, 25.0
    d = x.shape[-1] // 2
    g, u = x[..., :d].to(torch.float32), x[..., d:].to(torch.float32)
    want = (beta * torch.tanh(g / beta) * torch.sigmoid(g)) * (lb * torch.tanh(u / lb))
    check("situ matches the shipped SituAndMul formula",
          torch.equal(situ(x, beta, lb), want))


def test_attn_res_is_a_softmax_mixture():
    torch.manual_seed(0)
    N, B, H = 5, 3, 64
    ps, br = torch.randn(N, H), torch.randn(N, B, H)
    w, p = torch.randn(H), torch.randn(1, H)
    out = apply_attn_res(ps, br, p, w, 1e-5)
    # output must lie in the convex hull of the B+1 mixed vectors -> bounded by their range
    v = torch.cat((br, ps.unsqueeze(1)), dim=1)
    inside = ((out <= v.max(1).values + 1e-5) & (out >= v.min(1).values - 1e-5)).all()
    check("attn-residual output is a convex mixture of block checkpoints", bool(inside))
    # with an empty block_residual it must be the identity
    empty = torch.zeros(N, 0, H)
    check("attn-residual with 0 checkpoints is the identity",
          torch.allclose(apply_attn_res(ps, empty, p, w, 1e-5), ps, atol=1e-6))


def test_mxfp4_roundtrip_is_bf16_exact():
    import numpy as np
    ws = WeightStore()
    d = ws.expert(1, 0, "w1")
    rt = d.to(torch.bfloat16).float()
    check("decoded MXFP4 is exactly representable in bf16 (P0-3)", torch.equal(rt, d),
          f"absmax {d.abs().max():.4f}")


# ---------------------------------------------------------------------- layer tests

def test_layers():
    cfg = K3Config.load()
    ws = WeightStore()
    print(f"\n  config: {cfg.num_hidden_layers} layers, {len(cfg.mla_layers)} MLA / "
          f"{len(cfg.kda_layers)} KDA, {cfg.num_experts} experts top-{cfg.num_experts_per_token}")
    check("layer 0 is KDA, layer 3 is MLA (1-indexed config converted)",
          cfg.is_kda(0) and not cfg.is_kda(3), f"MLA head: {cfg.mla_layers[:4]}")

    torch.manual_seed(0)
    T, H = 4, cfg.hidden_size
    x = torch.randn(1, T, H) * 0.02
    block_residual = torch.zeros(T, 0, H)

    state = conv = cache = None
    for L in (0, 1, 2, 3):
        t0 = time.time()
        layer = DecoderLayer(cfg, L, ws)
        x_in = x.clone()
        x, block_residual, state, conv, cache = layer(
            x, block_residual, state=None, conv=None, cache=None)
        dt = time.time() - t0
        kind = "KDA" if cfg.is_kda(L) else "MLA"
        mlp = "denseMLP" if L < cfg.first_k_dense_replace else "latentMoE"
        ok = bool(torch.isfinite(x).all())
        print(f"  [{'PASS' if ok else 'FAIL'}] layer {L:>2} {kind}+{mlp:<9} "
              f"out{tuple(x.shape)} blockres{tuple(block_residual.shape)} "
              f"|x| {x.abs().mean():.3e}  {dt:5.1f}s  sha {ck(x)}")
        if not ok:
            global FAILS
            FAILS += 1

    check("block_residual grew exactly once (only layer 0 % 12 == 0)",
          block_residual.shape[1] == 1, f"got {block_residual.shape[1]}")

    # router exactness on a real MoE layer
    moe = DecoderLayer(cfg, 1, ws).mlp
    r = moe.route(x)
    check("router selects exactly top-16 distinct experts per token",
          r.expert_indices.shape == (T, 16)
          and all(len(set(row.tolist())) == 16 for row in r.expert_indices))
    check("router weights renormalise to 1",
          torch.allclose(r.expert_weights.sum(-1), torch.ones(T), atol=1e-6),
          f"sums {r.expert_weights.sum(-1).tolist()}")
    check("router top-16 == top-16 of corrected scores (bias applied to choice only)",
          torch.equal(r.expert_indices.sort(-1).values,
                      r.corrected_scores.topk(16, -1).indices.sort(-1).values))
    raw_topk = r.raw_scores.gather(1, r.expert_indices)
    check("router weights come from RAW sigmoid scores, not the bias-corrected ones",
          torch.allclose(raw_topk / raw_topk.sum(-1, keepdim=True), r.expert_weights, atol=1e-7))
    print(f"       layer-1 token-0 experts: {sorted(r.expert_indices[0].tolist())}")


def test_mla_cache():
    """Prefill(T) must equal T sequential single-token decodes -- the streaming path."""
    cfg = K3Config.load()
    ws = WeightStore()
    from k3ref.layers import MLAAttention
    torch.manual_seed(0)
    mla = MLAAttention(cfg, 3, ws)
    T = 4
    x = torch.randn(1, T, cfg.hidden_size) * 0.02
    full, _ = mla(x)
    cache, outs = None, []
    for i in range(T):
        o, cache = mla(x[:, i:i + 1], cache=cache)
        outs.append(o)
    step = torch.cat(outs, dim=1)
    d = (step - full).abs().max().item()
    check("MLA prefill == incremental decode (KV cache correct)", d < 1e-4, f"max|d| {d:.3e}")


def test_kda_state():
    cfg = K3Config.load()
    ws = WeightStore()
    from k3ref.layers import KDAAttention
    torch.manual_seed(0)
    kda = KDAAttention(cfg, 1, ws)
    T = 6
    x = torch.randn(1, T, cfg.hidden_size) * 0.02
    full, s_full, _ = kda(x)
    st = cv = None
    outs = []
    for i in range(T):
        o, st, cv = kda(x[:, i:i + 1], state=st, conv=cv)
        outs.append(o)
    step = torch.cat(outs, dim=1)
    do = (step - full).abs().max().item()
    ds = (st - s_full).abs().max().item()
    check("KDA prefill == incremental decode (recurrent + conv state correct)",
          do < 1e-4 and ds < 1e-4, f"max|d out| {do:.2e}  max|d state| {ds:.2e}")


if __name__ == "__main__":
    require_checkpoint()
    print("k3ref M2 gate\n" + "=" * 60)
    test_situ()
    test_attn_res_is_a_softmax_mixture()
    test_mxfp4_roundtrip_is_bf16_exact()
    test_kda_state()
    test_mla_cache()
    test_layers()
    print("=" * 60)
    print(f"M2: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    sys.exit(1 if FAILS else 0)
