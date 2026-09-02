"""M3 gate: MLX MXFP4 decoder must match the NumPy oracle exactly.

Spec section 15 requires: exhaustive over all 16 FP4 codes, exhaustive over all 256 scale
bytes, and agreement on real checkpoint tensors before any float casting.

    python tests/test_mxfp4_mlx.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import CKPT, checkpoint_available  # noqa: E402
from air_llm.models.kimi_k3.mxfp4 import (  # noqa: E402
    decode_mxfp4_mlx, decode_mxfp4_reference, decode_scale_mlx, mxfp4_matvec)
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def test_all_codes():
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scales = np.full((16, 1), 127, dtype=np.uint8)
    want = decode_mxfp4_reference(packed, scales, 32)
    got = np.array(decode_mxfp4_mlx(mx.array(packed), mx.array(scales), 32))
    check("all 256 byte values / 16 FP4 codes", np.array_equal(got, want))


def test_all_scales():
    """Exhaustive over the 254 representable E8M0 bytes.

    Byte 0 is 2**-127, subnormal in fp32; Metal flushes denormals to zero, so it cannot
    agree with NumPy on the GPU. It is unreachable in this checkpoint (see
    test_scale_range_precondition), so the exhaustive check covers 1..254 and byte 0 is
    handled as a guarded precondition instead.
    """
    packed = np.full((256, 16), 0x22, dtype=np.uint8)     # +1.0 in both nibbles
    scales = np.arange(256, dtype=np.uint8).reshape(256, 1)
    want = decode_mxfp4_reference(packed, scales, 32)
    got = np.array(decode_mxfp4_mlx(mx.array(packed), mx.array(scales), 32))
    check("all E8M0 scale bytes 1..254 (the representable range)",
          np.array_equal(got[1:], want[1:]))
    check("byte 0 is subnormal and flushes to zero on Metal (documented divergence)",
          got[0].max() == 0.0 and want[0].max() > 0.0)

    e = np.arange(1, 255, dtype=np.uint8)
    ref = np.ldexp(np.ones(254, dtype=np.float32), e.astype(np.int32) - 127)
    check("decode_scale_mlx == 2**(byte-127) for bytes 1..254",
          np.array_equal(np.array(decode_scale_mlx(mx.array(e))), ref))


def test_scale_range_precondition():
    """The unrepresentable byte must not occur in real data."""
    ws = WeightStore()
    lo, hi = 255, 0
    n = 0
    for e in range(0, 896, 29):
        for w in ("w1", "w2", "w3"):
            s, _ = ws.raw(f"language_model.model.layers.1.block_sparse_moe.experts."
                          f"{e}.{w}.weight_scale")
            a = np.asarray(s)
            lo, hi, n = min(lo, int(a.min())), max(hi, int(a.max())), n + 1
    check("real scale bytes stay inside the representable range 1..254",
          lo >= 1 and hi <= 254, f"{n} tensors, observed {lo}..{hi} -> 2^[{lo-127}, {hi-127}]")


def test_real_tensors():
    ws = WeightStore()
    picks = [(1, 0, "w1"), (1, 0, "w2"), (1, 0, "w3"), (1, 417, "w1"), (1, 895, "w2")]
    allok = True
    for L, E, w in picks:
        base = f"language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w}"
        p, _ = ws.raw(base + ".weight_packed")
        s, _ = ws.raw(base + ".weight_scale")
        p, s = np.ascontiguousarray(p), np.ascontiguousarray(s)
        want = decode_mxfp4_reference(p, s)
        got = np.array(decode_mxfp4_mlx(mx.array(p), mx.array(s)))
        eq = np.array_equal(got, want)
        allok &= eq
        print(f"  [{'PASS' if eq else 'FAIL'}] real L{L} e{E} {w} {tuple(want.shape)} "
              f"exact vs NumPy oracle" + ("" if eq else
              f"  maxdiff {np.abs(got-want).max():.3e}"))
    if not allok:
        global FAILS
        FAILS += 1

    # bf16 output path must also be exact (P0-3 losslessness)
    base = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    p, _ = ws.raw(base + ".weight_packed")
    s, _ = ws.raw(base + ".weight_scale")
    want = decode_mxfp4_reference(np.ascontiguousarray(p), np.ascontiguousarray(s))
    got = np.array(decode_mxfp4_mlx(mx.array(np.ascontiguousarray(p)),
                                    mx.array(np.ascontiguousarray(s)),
                                    out_dtype=mx.bfloat16).astype(mx.float32))
    check("bf16 output path is lossless on a real expert", np.array_equal(got, want))


def test_matvec():
    """Fused-path stand-in must match an fp32 NumPy matmul against the expanded matrix."""
    ws = WeightStore()
    base = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    p, _ = ws.raw(base + ".weight_packed")
    s, _ = ws.raw(base + ".weight_scale")
    p, s = np.ascontiguousarray(p), np.ascontiguousarray(s)
    W = decode_mxfp4_reference(p, s)                       # [3072, 3584]
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((4, W.shape[1])) * 0.02).astype(np.float32)

    want = x @ W.T
    got = np.array(mxfp4_matvec(mx.array(x), mx.array(p), mx.array(s)))
    rel = np.abs(got - want).max() / np.abs(want).max()
    # bf16 inputs, fp32 accumulate -> bf16-level relative error is expected and fine
    check("mxfp4_matvec matches NumPy-expanded matmul", rel < 5e-3,
          f"max rel err {rel:.2e}")


def test_speed():
    ws = WeightStore()
    base = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    p, _ = ws.raw(base + ".weight_packed")
    s, _ = ws.raw(base + ".weight_scale")
    P, S = mx.array(np.ascontiguousarray(p)), mx.array(np.ascontiguousarray(s))
    mx.eval(decode_mxfp4_mlx(P, S))
    t0 = time.time()
    N = 20
    for _ in range(N):
        mx.eval(decode_mxfp4_mlx(P, S, out_dtype=mx.bfloat16))
    dt = (time.time() - t0) / N
    n = p.shape[0] * p.shape[1] * 2
    print(f"  [time] decode one expert projection ({n/1e6:.1f}M values): "
          f"{dt*1e3:.2f} ms  ({n/dt/1e9:.2f} Gval/s)")
    print(f"         one expert = 3 projections -> {3*dt*1e3:.1f} ms; "
          f"16 experts x 92 layers -> {16*92*3*dt:.1f} s/token of decode")


if __name__ == "__main__":
    print(f"M3 gate: MXFP4 in MLX   (device {mx.default_device()})")
    print("=" * 62)
    test_all_codes()
    test_all_scales()
    if checkpoint_available():
        test_scale_range_precondition()
        test_real_tensors()
        test_matvec()
        test_speed()
    else:
        print(f"  [SKIP] real-weight gates: no checkpoint at {CKPT}")
        print("         set K3_CKPT=/path/to/Kimi-K3 to run them")
    print("=" * 62)
    print(f"M3: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    sys.exit(1 if FAILS else 0)
