"""M6 gate: MLX router + latent MoE vs the k3ref oracle, on real layer-1 weights.

Spec section 16 requires EXACT top-16 expert selection (a tolerance is not acceptable --
one different expert means a different computation), and router weights within 1e-5.

    python tests/test_moe_mlx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.moe import LatentMoE as MoEmlx  # noqa: E402
from k3ref.layers import K3Config, LatentMoE as MoEref  # noqa: E402
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0
LAYER = 1


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def build(cfg, ws):
    p = f"language_model.model.layers.{LAYER}.block_sparse_moe."
    keys = ("gate.weight", "gate.e_score_correction_bias",
            "routed_expert_down_proj.weight", "routed_expert_up_proj.weight",
            "routed_expert_norm.weight", "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight", "shared_experts.down_proj.weight")
    w = {k: mx.array(np.ascontiguousarray(ws.get(p + k).numpy())) for k in keys}

    def source(layer, expert, which):
        base = (f"language_model.model.layers.{layer}.block_sparse_moe.experts."
                f"{expert}.{which}")
        pk, _ = ws.raw(base + ".weight_packed")
        sc, _ = ws.raw(base + ".weight_scale")
        return mx.array(np.ascontiguousarray(pk)), mx.array(np.ascontiguousarray(sc))

    return MoEmlx(cfg, LAYER, w, source)


def main():
    cfg = K3Config.load()
    ws = WeightStore()
    print(f"M6 gate: router + latent MoE, real layer {LAYER}   (device {mx.default_device()})")
    print("=" * 70)
    t0 = time.time()
    ref = MoEref(cfg, LAYER, ws)
    mlx = build(cfg, ws)
    print(f"  (loaded both in {time.time()-t0:.1f}s)")

    torch.manual_seed(0)
    T = 4
    x = torch.randn(1, T, cfg.hidden_size) * 0.02
    xm = mx.array(x.numpy())

    # ---------------- router
    r_ref = ref.route(x)
    r_mlx = mlx.route(xm)

    i_ref = np.sort(r_ref.expert_indices.numpy(), axis=-1)
    i_mlx = np.sort(np.array(r_mlx.expert_indices, copy=False), axis=-1)
    check("EXACT top-16 expert selection matches k3ref", np.array_equal(i_ref, i_mlx),
          f"{'identical' if np.array_equal(i_ref, i_mlx) else 'DIFFERS'} across {T} tokens")

    d = np.abs(np.array(r_mlx.raw_scores, copy=False) - r_ref.raw_scores.numpy()).max()
    check("router raw sigmoid scores match", d <= 1e-5, f"max|d| {d:.3e}")

    # weights must be compared in a canonical order (topk is unsorted in both)
    def canon(idx, wts):
        idx = np.asarray(idx)
        wts = np.asarray(wts)
        o = np.argsort(idx, axis=-1)
        return np.take_along_axis(wts, o, axis=-1)

    w_ref = canon(r_ref.expert_indices.numpy(), r_ref.expert_weights.numpy())
    w_mlx = canon(np.array(r_mlx.expert_indices, copy=False),
                  np.array(r_mlx.expert_weights, copy=False))
    d = np.abs(w_ref - w_mlx).max()
    check("router weights match (atol 1e-5, spec section 16)", d <= 1e-5, f"max|d| {d:.3e}")
    check("router weights sum to 1 per token",
          np.allclose(w_mlx.sum(-1), 1.0, atol=1e-6))

    # ---------------- shared experts alone (bf16 path, unquantised)
    d = np.abs(np.array(mlx.shared_forward(xm), copy=False)
               - ref.shared_forward(x).numpy()).max()
    check("shared experts match", d <= 1e-4, f"max|d| {d:.3e}")

    # ---------------- one routed expert in isolation
    h_ref = x.reshape(-1, cfg.hidden_size) @ ref.down.T
    e = int(r_ref.expert_indices[0, 0])
    o_ref = ref.expert_forward(h_ref, e)
    o_mlx = mlx.expert_forward(mx.array(h_ref.numpy()), e)
    d = np.abs(np.array(o_mlx, copy=False) - o_ref.numpy()).max()
    rel = d / max(np.abs(o_ref.numpy()).max(), 1e-30)
    check(f"single routed expert {e} matches (bf16 weights, fp32 accum)", rel <= 5e-3,
          f"max|d| {d:.3e}  rel {rel:.2e}")

    # ---------------- full MoE block
    t0 = time.time()
    y_ref = ref(x)
    t_ref = time.time() - t0
    t0 = time.time()
    y_mlx = mlx(xm)
    mx.eval(y_mlx)
    t_mlx = time.time() - t0
    a, b = np.array(y_mlx, copy=False), y_ref.numpy()
    d = np.abs(a - b).max()
    rel = d / max(np.abs(b).max(), 1e-30)
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))
    check("full latent-MoE block matches (cosine >= 0.999, spec section 16)",
          cos >= 0.999, f"cos {cos:.8f}  max|d| {d:.3e}  rel {rel:.2e}")
    print(f"       timing: k3ref(torch/cpu) {t_ref:.1f}s   mlx(metal) {t_mlx:.1f}s   "
          f"speedup {t_ref/max(t_mlx,1e-9):.1f}x")

    print("=" * 70)
    print(f"M6: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
