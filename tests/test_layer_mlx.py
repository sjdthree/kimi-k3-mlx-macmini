"""M7 gate: full MLX decoder layers vs k3ref, on real weights.

Exercises the attention-residual machinery: layer 0 opens a block (0 % 12 == 0), layers
1..N run inside it. The full 12-layer block period is covered by --layers 12 once the
checkpoint is local (it needs experts for every layer).

    python tests/test_layer_mlx.py [--layers N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.layer import empty_block_residual  # noqa: E402
from air_llm.models.kimi_k3.loader import TensorSource, build_layer  # noqa: E402
from k3ref.layers import K3Config, DecoderLayer as RefLayer  # noqa: E402
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--cache", default=str(Path("~/k3_cache").expanduser()))
    ap.add_argument("--ckpt", default=None, help="override checkpoint dir (e.g. the USB4 copy)")
    a = ap.parse_args()

    cfg = K3Config.load()
    ws = WeightStore()
    src = TensorSource(ckpt=a.ckpt, cache=a.cache)
    print(f"M7 gate: full decoder layers 0..{a.layers-1}   (device {mx.default_device()})")
    print(f"  attn_res_block_size={cfg.attn_res_block_size}; layers opening a block: "
          f"{[i for i in range(a.layers) if i % cfg.attn_res_block_size == 0]}")
    print("=" * 74)

    torch.manual_seed(0)
    T = 2
    x = torch.randn(1, T, cfg.hidden_size) * 0.02

    x_ref, br_ref = x.clone(), torch.zeros(T, 0, cfg.hidden_size)
    x_mlx = mx.array(x.numpy())
    br_mlx = empty_block_residual(T, cfg.hidden_size)

    worst = 0.0
    for L in range(a.layers):
        t0 = time.time()
        ref = RefLayer(cfg, L, ws)
        x_ref, br_ref, _, _, _ = ref(x_ref, br_ref)
        t_ref = time.time() - t0

        t0 = time.time()
        mlx_layer = build_layer(cfg, L, src)
        x_mlx, br_mlx, _ = mlx_layer(x_mlx, br_mlx)
        mx.eval(x_mlx, br_mlx)
        t_mlx = time.time() - t0

        a_, b_ = np.array(x_mlx, copy=False), x_ref.detach().numpy()
        d = np.abs(a_ - b_).max()
        rel = d / max(np.abs(b_).max(), 1e-30)
        cos = float((a_ * b_).sum() / (np.linalg.norm(a_) * np.linalg.norm(b_)))
        worst = max(worst, 1.0 - cos)
        kind = "KDA" if cfg.is_kda(L) else "MLA"
        opens = " *opens block*" if L % cfg.attn_res_block_size == 0 else ""
        ok = cos >= 0.999
        if not ok:
            global FAILS
            FAILS += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] layer {L:>2} {kind}  cos {cos:.8f}  "
              f"rel {rel:.2e}  br{tuple(br_mlx.shape)}  ref {t_ref:5.1f}s mlx {t_mlx:5.1f}s{opens}")

        # block_residual must stay identical too -- it feeds every later layer
        dbr = np.abs(np.array(br_mlx, copy=False) - br_ref.detach().numpy()).max() \
            if br_ref.shape[1] else 0.0
        if dbr > 1e-4:
            check(f"layer {L} block_residual matches", False, f"max|d| {dbr:.3e}")

    check(f"all {a.layers} layers within cosine 0.999 (spec section 16)", worst <= 1e-3,
          f"worst 1-cos {worst:.2e}")
    check("block_residual has exactly the expected number of checkpoints",
          br_mlx.shape[1] == sum(1 for i in range(a.layers) if i % cfg.attn_res_block_size == 0),
          f"{br_mlx.shape[1]} checkpoints")

    print("=" * 74)
    print(f"M7: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
