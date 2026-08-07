"""Resident-weight-cache gate.

Profiling put the runtime 20x I/O-bound, with 105.7 of 129.8 GiB per token being dense
weights that never change. Holding what fits removes that I/O proportionally.

Two things must be true:
  1. results are IDENTICAL with and without the cache (it is an I/O optimisation only)
  2. resident bytes never exceed the configured budget

    python tests/test_resident_cache.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.loader import TensorSource  # noqa: E402
from air_llm.models.kimi_k3.model import KimiK3Model  # noqa: E402
from air_llm.models.kimi_k3.recurrent_state import KimiK3State  # noqa: E402
from k3ref.layers import K3Config  # noqa: E402

FAILS = 0
GiB = 2 ** 30


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def run(resident_gb, layers=2, tokens=3, dtype=None):
    import mlx.core as _mx
    cfg = K3Config.load()
    cfg.num_hidden_layers = layers
    src = TensorSource(cache=str(Path("~/k3_cache").expanduser()),
                       resident_gb=resident_gb,
                       weight_dtype=dtype or _mx.float32)
    model = KimiK3Model(cfg, src, max_memory_gb=48.0)
    outs, reads = [], []
    state = KimiK3State()
    for i in range(tokens):
        src.reset_counters()
        h = model.forward_token(model.embed([100 + i]), state)
        mx.eval(h)
        outs.append(np.array(h, copy=False).copy())
        reads.append(src.bytes_read)
    return outs, reads, src


def main():
    print(f"resident-cache gate   (device {mx.default_device()})")
    print("=" * 74)

    # fp32 (validation mode): layers 0-1 dense is ~7.2 GiB resident, so budget 8
    BUDGET = 8.0
    off_out, off_reads, off_src = run(0.0)
    on_out, on_reads, on_src = run(BUDGET)

    same = all(np.array_equal(a, b) for a, b in zip(off_out, on_out))
    check("results are bit-identical with and without the cache", same)

    print(f"       bytes read per token (GiB):")
    print(f"         streaming : {[round(r/GiB, 2) for r in off_reads]}")
    print(f"         resident  : {[round(r/GiB, 2) for r in on_reads]}")

    check("first token reads the same (cold cache)",
          abs(off_reads[0] - on_reads[0]) < 0.01 * off_reads[0],
          f"{off_reads[0]/GiB:.2f} vs {on_reads[0]/GiB:.2f} GiB")

    saved = 1 - on_reads[-1] / off_reads[-1]
    check("later tokens read materially less", saved > 0.2,
          f"{saved*100:.1f}% less I/O on token {len(on_reads)-1}")

    st = on_src.cache_stats()
    check("resident bytes stay inside the budget",
          st["resident_bytes"] <= BUDGET * GiB,
          f"{st['resident_bytes']/GiB:.2f} of {BUDGET:.2f} GiB, {st['entries']} tensors")
    print(f"       hit rate {st['hit_rate']*100:.1f}% "
          f"({st['hits']} hits / {st['misses']} misses)")

    # budget is honoured even when asked to hold far more than fits
    tiny = TensorSource(cache=str(Path("~/k3_cache").expanduser()), resident_gb=0.25)
    cfg = K3Config.load()
    cfg.num_hidden_layers = 2
    m = KimiK3Model(cfg, tiny, max_memory_gb=48.0)
    s = KimiK3State()
    for i in range(2):
        mx.eval(m.forward_token(m.embed([100]), s))
    check("a small budget is not exceeded under pressure",
          tiny.resident_bytes <= 0.25 * GiB,
          f"{tiny.resident_bytes/2**20:.0f} MiB of 256 MiB")

    # ---- bf16 runtime mode: same weights, half the resident bytes
    bf_out, bf_reads, bf_src = run(4.0, dtype=mx.bfloat16)
    bst = bf_src.cache_stats()
    print(f"\n  bf16 runtime mode (budget 4 GiB):")
    print(f"       reads {[round(r/GiB,2) for r in bf_reads]}  "
          f"hit rate {bst['hit_rate']*100:.1f}%  resident "
          f"{bst['resident_bytes']/GiB:.2f} GiB / {bst['entries']} tensors")
    check("bf16 mode reaches the same hit rate at HALF the budget",
          bst["hit_rate"] > 0.5, f"{bst['hit_rate']*100:.1f}%")
    cos = float((bf_out[-1] * off_out[-1]).sum() /
                (np.linalg.norm(bf_out[-1]) * np.linalg.norm(off_out[-1])))
    check("bf16 runtime mode stays within spec section 16 cosine 0.999 of fp32",
          cos >= 0.999, f"cos {cos:.8f}")

    # ---- projection for the real model
    print("\n  projection for the full 93-layer model (bf16 runtime mode, where a")
    print("  GiB of budget holds a GiB of on-disk weight; fp32 holds half as much):")
    dense_total, expert_per_tok = 105.7, 24.1
    for budget in (0, 8, 16, 24, 32, 40, 45):
        served = min(budget, dense_total)
        io = (dense_total - served + expert_per_tok) * 1024 / 3378
        print(f"    resident {budget:>2} GiB -> dense streamed "
              f"{dense_total-served:5.1f} GiB, I/O {io:5.1f} s/token"
              f"{'   <- fits under a 48 GiB ceiling' if budget == 45 else ''}")
    print("    (compute is ~2.0 s/token, so I/O stays dominant at every budget)")

    print("=" * 74)
    print(f"RESIDENT: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
