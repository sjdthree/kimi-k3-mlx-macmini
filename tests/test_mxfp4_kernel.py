"""M10 gate: fused Metal MXFP4 matvec.

Correctness is measured against the NumPy oracle (exact unpack, fp32 matmul), NOT against
the unfused MLX path -- the fused kernel keeps weights exact and accumulates in fp32, so
it should be at least as accurate as the unfused bf16-weight route.

    python tests/test_mxfp4_kernel.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import CKPT, checkpoint_available  # noqa: E402
from air_llm.kernels.metal.mxfp4_matvec import mxfp4_matvec_fused  # noqa: E402
from air_llm.models.kimi_k3.mxfp4 import (  # noqa: E402
    decode_mxfp4_mlx, decode_mxfp4_reference, mxfp4_matvec)
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def get(ws, L, e, w):
    base = f"language_model.model.layers.{L}.block_sparse_moe.experts.{e}.{w}"
    p, _ = ws.raw(base + ".weight_packed")
    s, _ = ws.raw(base + ".weight_scale")
    return np.ascontiguousarray(p), np.ascontiguousarray(s)


def test_synthetic():
    """Random packed data exercises code/scale combinations the checkpoint may not use."""
    rng = np.random.default_rng(0)
    N, K = 128, 256
    packed = rng.integers(0, 256, size=(N, K // 2), dtype=np.uint8)
    scales = rng.integers(100, 140, size=(N, K // 32), dtype=np.uint8)
    W = decode_mxfp4_reference(packed, scales)
    for M in (1, 2, 4):
        x = rng.standard_normal((M, K)).astype(np.float32)
        want = x.astype(np.float64) @ W.T.astype(np.float64)
        got = np.array(mxfp4_matvec_fused(mx.array(x), mx.array(packed), mx.array(scales)))
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
        check(f"synthetic M={M} vs fp64 oracle", rel < 1e-6, f"rel {rel:.2e}")


def test_real():
    ws = WeightStore()
    rng = np.random.default_rng(1)
    for L, e, w in [(1, 0, "w1"), (1, 0, "w2"), (1, 417, "w3"), (1, 895, "w1")]:
        p, s = get(ws, L, e, w)
        W = decode_mxfp4_reference(p, s)
        x = (rng.standard_normal((1, W.shape[1])) * 0.02).astype(np.float32)
        want = x.astype(np.float64) @ W.T.astype(np.float64)
        got = np.array(mxfp4_matvec_fused(mx.array(x), mx.array(p), mx.array(s)))
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
        check(f"real L{L} e{e} {w} {W.shape} vs fp64 oracle", rel < 1e-5, f"rel {rel:.2e}")


def test_beats_unfused_accuracy():
    """Fused (exact weights, fp32 accum) must be no worse than unfused (bf16 weights)."""
    ws = WeightStore()
    p, s = get(ws, 1, 0, "w1")
    W = decode_mxfp4_reference(p, s)
    rng = np.random.default_rng(2)
    x = (rng.standard_normal((1, W.shape[1])) * 0.02).astype(np.float32)
    want = x.astype(np.float64) @ W.T.astype(np.float64)

    fused = np.array(mxfp4_matvec_fused(mx.array(x), mx.array(p), mx.array(s)))
    unfused = np.array(mxfp4_matvec(mx.array(x), mx.array(p), mx.array(s)))
    rf = np.abs(fused - want).max() / np.abs(want).max()
    ru = np.abs(unfused - want).max() / np.abs(want).max()
    check("fused is at least as accurate as the unfused bf16 path",
          rf <= ru, f"fused rel {rf:.2e}  unfused rel {ru:.2e}")


def test_speed():
    ws = WeightStore()
    p, s = get(ws, 1, 0, "w1")
    P, S = mx.array(p), mx.array(s)
    x = mx.array((np.random.default_rng(3).standard_normal((1, p.shape[1] * 2))
                  * 0.02).astype(np.float32))
    packed_bytes = p.nbytes + s.nbytes

    def timeit(fn, n=30):
        mx.eval(fn())
        t0 = time.time()
        for _ in range(n):
            mx.eval(fn())
        return (time.time() - t0) / n

    t_dec = timeit(lambda: decode_mxfp4_mlx(P, S, out_dtype=mx.bfloat16))
    t_unf = timeit(lambda: mxfp4_matvec(x, P, S))
    t_fus = timeit(lambda: mxfp4_matvec_fused(x, P, S))

    print(f"\n  per projection ({p.shape[0]}x{p.shape[1]*2}, {packed_bytes/2**20:.1f} MiB packed):")
    print(f"      decode only        {t_dec*1e3:7.2f} ms")
    print(f"      unfused decode+mm  {t_unf*1e3:7.2f} ms")
    print(f"      FUSED matvec       {t_fus*1e3:7.2f} ms   "
          f"({t_unf/t_fus:.1f}x faster, {packed_bytes/t_fus/1e9:.1f} GB/s effective)")

    per_tok_unf = 16 * 92 * 3 * t_unf
    per_tok_fus = 16 * 92 * 3 * t_fus
    print(f"\n  16 experts x 92 layers x 3 projections per token:")
    print(f"      unfused {per_tok_unf:6.1f} s/token")
    print(f"      FUSED   {per_tok_fus:6.1f} s/token   "
          f"(saves {per_tok_unf-per_tok_fus:.1f} s/token)")
    check("fused is faster than the unfused path", t_fus < t_unf,
          f"{t_unf/t_fus:.1f}x")


if __name__ == "__main__":
    print(f"M10 gate: fused Metal MXFP4 matvec   (device {mx.default_device()})")
    print("=" * 70)
    test_synthetic()
    if checkpoint_available():
        test_real()
        test_beats_unfused_accuracy()
        test_speed()
    else:
        print(f"  [SKIP] real-weight gates: no checkpoint at {CKPT}")
        print("         set K3_CKPT=/path/to/Kimi-K3 to run them")
    print("=" * 70)
    print(f"M10: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    sys.exit(1 if FAILS else 0)
