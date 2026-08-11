"""KDA prefill must use memory bounded by the state, not by the prompt length.

WHAT THIS CAUGHT
----------------
The 2026-08-10 long-prompt run died at layer 0 with

    MemoryError: peak unified memory 48.9 GiB exceeded the ceiling 48.0 GiB

Layer 0 is a KDA layer. `_recurrent` walks the prompt one token at a time, and MLX is
lazy, so the entire T-step chain stayed live as an unevaluated graph holding one
[B,H,D,D] state per step -- 6 MiB at the real 96 x 128 x 128 fp32. That is 1.7 GiB for
the 291-token prompt every previous run used, and 11.5 GiB for a 1,955-token one.

Nothing was wrong with the arithmetic, which is why every parity test passed: the bug is
invisible until the prompt is long enough. The fix forces evaluation every EVAL_EVERY
steps so MLX can release the graph behind it.

Gates
-----
    K1  results are bit-identical with and without the periodic eval
    K2  peak memory grows with STATE size, not with T
    K3  incremental decode still matches a single long prefill

Runs on random weights at real KDA head dimensions; no checkpoint access.

    conda activate airllm && python tests/test_kda_prefill_memory.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.attention_kda import KDAAttention  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return ok


@dataclass
class KDACfg:
    """Real head geometry, fewer heads so the test runs in seconds. The memory claim is
    about the T dimension, which is kept realistic."""
    hidden_size: int = 512
    kda_num_heads: int = 8
    kda_head_dim: int = 128        # real
    kda_conv_size: int = 4         # real
    kda_gate_lower_bound: float = -5.0
    rms_norm_eps: float = 1e-5


def inputs(cfg, T, seed=0):
    rng = np.random.default_rng(seed)
    H, D = cfg.kda_num_heads, cfg.kda_head_dim
    f = lambda *s: mx.array((rng.standard_normal(s) * 0.3).astype(np.float32))
    return (f(1, T, H, D), f(1, T, H, D), f(1, T, H, D),
            mx.array((-np.abs(rng.standard_normal((1, T, H, D))) * 0.1).astype(np.float32)),
            mx.array((rng.random((1, T, H)) * 0.5 + 0.2).astype(np.float32)))


def run(cfg, T, every, seed=0):
    """Run the recurrence with EVAL_EVERY temporarily set to `every`."""
    q, k, v, g, beta = inputs(cfg, T, seed)
    old = KDAAttention.EVAL_EVERY
    KDAAttention.EVAL_EVERY = every
    try:
        mx.eval(q, k, v, g, beta)
        base = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else 0
        if hasattr(mx, "reset_peak_memory"):
            mx.reset_peak_memory()
        o, S = KDAAttention._recurrent(q, k, v, g, beta, cfg.kda_head_dim ** -0.5, None)
        mx.eval(o, S)
        peak = mx.get_peak_memory() if hasattr(mx, "get_peak_memory") else 0
        return np.array(o, copy=False), np.array(S, copy=False), peak
    finally:
        KDAAttention.EVAL_EVERY = old


def main():
    cfg = KDACfg()
    H, D = cfg.kda_num_heads, cfg.kda_head_dim
    state_mib = H * D * D * 4 / 2 ** 20
    print(f"KDA prefill memory  --  {H} heads x {D} x {D} fp32 = {state_mib:.2f} MiB/state")

    # ---- K1: the periodic eval must not change a single number
    print("\nK1 -- periodic eval does not change results")
    T = 256
    o_lazy, S_lazy, _ = run(cfg, T, every=10 ** 9)     # effectively never
    o_eval, S_eval, _ = run(cfg, T, every=64)
    check("outputs bit-identical",
          np.array_equal(o_lazy, o_eval), f"max|d| {np.abs(o_lazy - o_eval).max():.3e}")
    check("final state bit-identical",
          np.array_equal(S_lazy, S_eval), f"max|d| {np.abs(S_lazy - S_eval).max():.3e}")

    # ---- K2: peak must not scale with T
    print("\nK2 -- peak memory bounded by state, not prompt length")
    peaks = {}
    for T in (256, 512, 1024):
        _, _, p = run(cfg, T, every=64)
        peaks[T] = p
        print(f"       T={T:>5}  peak {p / 2 ** 20:8.1f} MiB   "
              f"(unbounded would need ~{T * state_mib:.0f} MiB of live state)")
    growth = peaks[1024] / max(peaks[256], 1)
    check("4x the prompt does not cost ~4x the memory", growth < 2.0,
          f"{growth:.2f}x for 4x T")

    # ---- K3: chunked evaluation must not break the carried state
    print("\nK3 -- prefill then decode == one long prefill")
    T, split = 192, 128
    q, k, v, g, beta = inputs(cfg, T, seed=3)
    scale = cfg.kda_head_dim ** -0.5
    o_full, S_full = KDAAttention._recurrent(q, k, v, g, beta, scale, None)
    o_a, S_a = KDAAttention._recurrent(q[:, :split], k[:, :split], v[:, :split],
                                       g[:, :split], beta[:, :split], scale, None)
    for i in range(split, T):
        sl = lambda x: x[:, i:i + 1]
        o_b, S_a = KDAAttention._recurrent(sl(q), sl(k), sl(v), sl(g), sl(beta), scale, S_a)
    d = float(np.abs(np.array(S_a, copy=False) - np.array(S_full, copy=False)).max())
    check("carried state matches after split prefill+decode", d <= 1e-4, f"max|d| {d:.3e}")

    print(f"\n{'KDA PREFILL: PASS' if FAILS == 0 else f'KDA PREFILL: {FAILS} FAILED'}")
    return FAILS


if __name__ == "__main__":
    sys.exit(main())
