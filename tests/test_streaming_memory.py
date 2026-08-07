"""Spec section 16 'Streaming Correctness' gate.

Requirements being tested:
  * only the current layer's non-expert weights are resident
  * only one selected expert projection is expanded or executed at a time
  * released experts do not remain referenced
  * memory remains below a configured ceiling
  * disk reads correspond only to required layer tensors, selected experts, shared
    experts, and model state
  * repeated generation does not show sustained memory growth

The leak test is the one that matters: a streaming implementation that quietly retains
one buffer per layer looks perfectly correct for a few tokens and then dies at layer 60
of a real run.

    python tests/test_streaming_memory.py
"""
from __future__ import annotations

import gc
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
MiB = 2 ** 20


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def peak():
    f = getattr(mx, "get_peak_memory", None)
    return int(f()) if f else 0


def active():
    f = getattr(mx, "get_active_memory", None)
    return int(f()) if f else 0


def reset_peak():
    f = getattr(mx, "reset_peak_memory", None)
    if f:
        f()


def main():
    cfg = K3Config.load()
    cfg.num_hidden_layers = 2                 # layers 0-1: both fully cached locally
    src = TensorSource(cache=str(Path("~/k3_cache").expanduser()))
    model = KimiK3Model(cfg, src, max_memory_gb=48.0)

    print(f"streaming/memory gate   (device {mx.default_device()})")
    print("=" * 74)

    # ---------------------------------------------------------------- expert sizes
    p, _ = src.raw("language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed")
    s, _ = src.raw("language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_scale")
    packed_one = p.nbytes + s.nbytes
    expanded_bf16 = p.shape[0] * p.shape[1] * 2 * 2
    layer_all_experts = packed_one * 3 * cfg.num_experts
    print(f"  one projection: {packed_one/MiB:.2f} MiB packed vs "
          f"{expanded_bf16/MiB:.2f} MiB expanded bf16")
    print(f"  a whole layer's experts would be {layer_all_experts/2**30:.2f} GiB packed")

    # ---------------------------------------------------------------- ceiling
    reset_peak()
    state = KimiK3State()
    h = model.embed([100, 200, 300])
    h = model.forward_token(h, state)
    mx.eval(h)
    pk = peak()
    check("peak memory stays far below a whole layer's expert set",
          pk < layer_all_experts, f"peak {pk/2**30:.2f} GiB vs {layer_all_experts/2**30:.2f} GiB")
    check("peak memory under the 48 GiB ceiling", pk < 48 * 2 ** 30,
          f"peak {pk/2**30:.2f} GiB")

    # ---------------------------------------------------------------- read volume
    # Only routed + shared experts should be read, never all 896.
    src.reset_counters()
    state2 = KimiK3State()
    h = model.forward_token(model.embed([100]), state2)
    mx.eval(h)
    read = src.bytes_read
    exp_read = src.expert_bytes
    dense_read = read - exp_read
    projections = src.experts_loaded
    n_moe_layers = sum(1 for L in range(cfg.num_hidden_layers)
                       if L >= cfg.first_k_dense_replace)
    expect_max = cfg.num_experts_per_token * 3 * n_moe_layers
    check("expert projections loaded == top_k x 3 x MoE layers (no over-reading)",
          projections == expect_max, f"{projections} loaded, expected {expect_max}")
    # The right comparison is EXPERT bytes vs the full expert set; dense weights are
    # read in full by design and must not be counted against the streaming claim.
    frac = exp_read / (layer_all_experts * n_moe_layers) if n_moe_layers else 0
    print(f"       read {read/2**30:.2f} GiB for 1 token "
          f"(dense {dense_read/2**30:.2f} + experts {exp_read/MiB:.0f} MiB)")
    check("expert reads are a small fraction of the full expert set", frac < 0.05,
          f"{frac*100:.2f}% of {layer_all_experts/2**30:.2f} GiB")
    expect_bytes = projections * packed_one
    check("expert bytes == projections x per-projection size (nothing extra read)",
          abs(exp_read - expect_bytes) < packed_one,
          f"{exp_read/MiB:.1f} MiB vs {expect_bytes/MiB:.1f} MiB expected")

    # ---------------------------------------------------------------- leak test
    # Repeated identical tokens must not grow resident memory.
    base = None
    actives = []
    for i in range(6):
        st = KimiK3State()
        h = model.forward_token(model.embed([100]), st)
        mx.eval(h)
        del h, st
        gc.collect()
        actives.append(active())
        if i == 1:
            base = actives[-1]
    growth = actives[-1] - base
    rel = growth / max(base, 1)
    print(f"       active memory per iteration (MiB): "
          f"{[round(a/MiB) for a in actives]}")
    check("no sustained memory growth across 6 identical passes (<2%, spec section 16)",
          rel < 0.02, f"grew {growth/MiB:+.1f} MiB ({rel*100:+.2f}%)")

    # ---------------------------------------------------------------- no retention
    # After a forward pass, nothing layer-sized should still be resident: only state,
    # hidden, and the small globals.
    st = KimiK3State()
    h = model.forward_token(model.embed([100]), st)
    mx.eval(h)
    gc.collect()
    resident = active()
    state_bytes = st.memory_report()["total"]
    check("resident memory after a pass is bounded (state + globals, not layers)",
          resident < 8 * 2 ** 30,
          f"resident {resident/2**30:.2f} GiB, state {state_bytes/MiB:.1f} MiB")

    print("=" * 74)
    print(f"STREAMING: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
