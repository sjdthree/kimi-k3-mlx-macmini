"""Measure real sec/token by running actual layers off the checkpoint drive.

The copy is partial, but shards land in layer order, so layers 0..N-1 are already
readable. Running them end-to-end gives a measured per-layer cost -- disk read, host
conversion, MLX upload, and compute together -- which is what a projection from raw
os.read throughput cannot capture.

    python tools/estimate_token_time.py --ckpt "$K3_CKPT"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.layer import LayerState, empty_block_residual
from air_llm.models.kimi_k3.loader import TensorSource, build_layer
from k3ref.layers import K3Config


def available_layers(ckpt: Path, manifest: Path, upto: int) -> int:
    """How many leading layers have their shard present on the target drive."""
    import gzip
    need = {}
    with gzip.open(manifest, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("__meta__") or r.get("layer") is None:
                continue
            need.setdefault(r["layer"], set()).add(r["file"])
    n = 0
    for L in range(upto):
        if L in need and all((ckpt / f).exists() for f in need[L]):
            n += 1
        else:
            break
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path(os.environ.get("K3_CKPT", "Kimi-K3")))
    ap.add_argument("--manifest", type=Path,
                    default=Path(os.environ.get("K3_MANIFEST", "manifest.jsonl.gz")))
    ap.add_argument("--mode", default="runtime", choices=["validation", "runtime"])
    ap.add_argument("--resident-gb", type=float, default=0.0)
    ap.add_argument("--cache", type=Path, default=None,
                    help="dense blob on faster storage (two-tier)")
    ap.add_argument("--max-layers", type=int, default=24)
    a = ap.parse_args()

    cfg = K3Config.load()
    n_avail = available_layers(a.ckpt, a.manifest, cfg.num_hidden_layers)
    n = min(n_avail, a.max_layers)
    if n == 0:
        raise SystemExit(f"no complete layers on {a.ckpt} yet")

    dtype = mx.bfloat16 if a.mode == "runtime" else mx.float32
    src = TensorSource(manifest=a.manifest, ckpt=a.ckpt, cache=a.cache,
                       resident_gb=a.resident_gb, weight_dtype=dtype)

    print(f"measuring {n} layers off {a.ckpt}  (mode {a.mode}, "
          f"resident {a.resident_gb} GiB, {n_avail} layers available)")
    print("=" * 76)

    H = cfg.hidden_size
    x = mx.random.normal((1, 1, H)) * 0.02
    br = empty_block_residual(1, H)
    mx.eval(x)

    kda_t, mla_t, kda_b, mla_b = [], [], [], []
    for L in range(n):
        src.reset_counters()
        t0 = time.time()
        layer = build_layer(cfg, L, src)
        out, br2, _ = layer(x, br, state=LayerState())
        mx.eval(out, br2)
        dt = time.time() - t0
        nb = src.bytes_read
        (kda_t if cfg.is_kda(L) else mla_t).append(dt)
        (kda_b if cfg.is_kda(L) else mla_b).append(nb)
        print(f"  L{L:>2} {'KDA' if cfg.is_kda(L) else 'MLA'}  {dt*1e3:7.1f} ms  "
              f"{nb/2**20:7.1f} MiB  {nb/dt/1e6:7.0f} MB/s effective")
        del layer

    def avg(v):
        return sum(v) / len(v) if v else 0.0

    n_kda, n_mla = len(cfg.kda_layers), len(cfg.mla_layers)
    tk, tm = avg(kda_t), avg(mla_t)
    bk, bm = avg(kda_b), avg(mla_b)

    print("\n" + "=" * 76)
    print(f"  measured  KDA {tk*1e3:6.1f} ms/layer ({bk/2**20:6.1f} MiB)  "
          f"n={len(kda_t)}")
    print(f"  measured  MLA {tm*1e3:6.1f} ms/layer ({bm/2**20:6.1f} MiB)  "
          f"n={len(mla_t)}")

    layers_s = tk * n_kda + tm * n_mla
    layers_b = bk * n_kda + bm * n_mla
    head_b = 2.19 * 2 ** 30
    # lm_head is a pure streaming read; use the effective rate we just measured
    eff = layers_b / layers_s if layers_s else 1
    head_s = head_b / eff
    total = layers_s + head_s

    print(f"\n  extrapolated to {cfg.num_hidden_layers} layers "
          f"({n_kda} KDA + {n_mla} MLA):")
    print(f"      layers    {layers_s:6.1f} s   ({layers_b/2**30:5.1f} GiB)")
    print(f"      lm_head   {head_s:6.1f} s   (2.19 GiB)")
    print(f"      TOTAL     {total:6.1f} s/token   =  {1/total:.3f} tok/s")
    print(f"      effective throughput {eff/1e6:.0f} MB/s")
    if a.cache:
        st = src.cache_stats()
        print(f"  two-tier: dense from {a.cache}, experts from {a.ckpt}")


if __name__ == "__main__":
    main()
