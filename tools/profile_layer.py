"""Separate I/O from compute per layer, and project a real 93-layer token time.

Everything so far has been measured with I/O and compute tangled together. To know what
to optimise next we need them apart: weight loading is bounded by the drive, compute is
bounded by kernels, and only one of those we can fix in software.

    python tools/profile_layer.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.layer import LayerState, empty_block_residual
from air_llm.models.kimi_k3.loader import build_layer, TensorSource
from k3ref.layers import K3Config

CACHE = str(Path("~/k3_cache").expanduser())


def timed(fn, n=3):
    fn()
    t0 = time.time()
    for _ in range(n):
        r = fn()
    mx.eval(r) if isinstance(r, mx.array) else None
    return (time.time() - t0) / n, r


def main():
    cfg = K3Config.load()
    src = TensorSource(cache=CACHE)
    H = cfg.hidden_size
    T = 1                                        # decode step

    print(f"per-layer profile, T={T} (decode)   device {mx.default_device()}")
    print("=" * 78)
    print(f"{'layer':>6} {'kind':>4} {'build(I/O)':>11} {'attn':>9} {'moe':>9} "
          f"{'total':>9} {'expert MiB':>11}")

    rows = []
    for L in (0, 1, 3):                          # dense-MLP KDA, MoE KDA, MoE MLA
        x = mx.random.normal((1, T, H)) * 0.02
        br = empty_block_residual(T, H)

        src.reset_counters()
        t0 = time.time()
        layer = build_layer(cfg, L, src)
        mx.eval(*[v for v in [getattr(layer.attn, 'o_w', None)] if v is not None])
        t_build = time.time() - t0
        dense_mib = src.bytes_read / 2 ** 20

        # attention alone
        src.reset_counters()
        xin = mx.array(np.array(x, copy=False))
        from air_llm.models.kimi_k3.ops import rmsnorm
        xn = rmsnorm(xin, layer.in_norm, cfg.rms_norm_eps)
        mx.eval(xn)
        if layer.is_kda:
            t_attn, _ = timed(lambda: layer.attn(xn)[0])
        else:
            t_attn, _ = timed(lambda: layer.attn(xn)[0])

        # mlp/moe alone (includes streaming its experts)
        src.reset_counters()
        t0 = time.time()
        y = layer.mlp(xn, return_router=True)[0]
        mx.eval(y)
        t_moe = time.time() - t0
        exp_mib = src.expert_bytes / 2 ** 20

        # whole layer
        src.reset_counters()
        t0 = time.time()
        out, br2, _ = layer(x, br, state=LayerState())
        mx.eval(out, br2)
        t_total = time.time() - t0

        kind = "KDA" if cfg.is_kda(L) else "MLA"
        print(f"{L:>6} {kind:>4} {t_build*1e3:9.1f}ms {t_attn*1e3:7.1f}ms "
              f"{t_moe*1e3:7.1f}ms {t_total*1e3:7.1f}ms {exp_mib:9.1f}")
        rows.append((L, kind, t_build, t_attn, t_moe, t_total, dense_mib, exp_mib))
        del layer

    # ---------------------------------------------------------------- projection
    kda_moe = next(r for r in rows if r[0] == 1)
    mla_moe = next(r for r in rows if r[0] == 3)
    n_kda, n_mla = len(cfg.kda_layers), len(cfg.mla_layers)

    print("\n" + "=" * 78)
    print("projected full 93-layer token (weights served from the USB4 NVMe @3.4 GB/s)")

    def project(row, n, label):
        L, kind, t_build, t_attn, t_moe, t_total, dense_mib, exp_mib = row
        # build() here reads from the LOCAL CACHE on internal NVMe; rescale the I/O
        # portion to the external drive's measured 3378 MB/s random read.
        io_s = (dense_mib + exp_mib) / 3378.0
        compute_s = max(t_total - t_build, 0.0)
        print(f"  {label:<22} x{n:<3} io {io_s*n:6.1f}s  compute {compute_s*n:6.1f}s  "
              f"({(dense_mib+exp_mib):.0f} MiB/layer)")
        return io_s * n, compute_s * n

    io1, c1 = project(kda_moe, n_kda, "KDA + latent MoE")
    io2, c2 = project(mla_moe, n_mla, "MLA + latent MoE")
    head_io = 2.19 * 1024 / 3378.0
    print(f"  {'lm_head stream':<22} x1   io {head_io:6.1f}s  compute    ~0s")
    print(f"\n  TOTAL  io {io1+io2+head_io:5.1f}s + compute {c1+c2:5.1f}s "
          f"= {io1+io2+head_io+c1+c2:5.1f}s/token")
    print(f"  (measured end-to-end will differ: this ignores overlap and Python overhead)")


if __name__ == "__main__":
    main()
