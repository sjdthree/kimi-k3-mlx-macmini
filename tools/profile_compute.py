"""Pure COMPUTE cost per layer, with every weight preloaded into unified memory.

The previous pass conflated I/O and compute: layer 3's experts are not in the local cache,
so its "MoE" time was 35 s of NAS reads. Here nothing touches disk inside the timed
region, so what is left is kernel time -- the only part software can still improve.

    python tools/profile_compute.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.layer import LayerState, empty_block_residual
from air_llm.models.kimi_k3.loader import build_layer, TensorSource
from k3ref.layers import K3Config

CACHE = str(Path("~/k3_cache").expanduser())


def bench(fn, n=10):
    r = fn()
    mx.eval(r)
    t0 = time.time()
    for _ in range(n):
        r = fn()
        mx.eval(r)
    return (time.time() - t0) / n


class Preloaded:
    """Memoising expert source. Warm it once, then `misses` must stay 0 while timing,
    otherwise a disk read has crept into the measured region."""

    def __init__(self, src):
        self.src = src
        self.d = {}
        self.misses = 0

    def __call__(self, layer, expert, which):
        key = (layer, expert, which)
        if key not in self.d:
            self.misses += 1
            self.d[key] = self.src.expert_source(layer, expert, which)
        return self.d[key]


def main():
    cfg = K3Config.load()
    src = TensorSource(cache=CACHE)
    H = cfg.hidden_size
    print(f"compute-only profile, T=1 decode   device {mx.default_device()}")
    print("=" * 76)

    for L in (1, 3):
        kind = "KDA" if cfg.is_kda(L) else "MLA"
        layer = build_layer(cfg, L, src)
        x = mx.random.normal((1, 1, H)) * 0.02
        mx.eval(x)

        from air_llm.models.kimi_k3.ops import rmsnorm
        xn = rmsnorm(x, layer.in_norm, cfg.rms_norm_eps)
        mx.eval(xn)

        t_attn = bench(lambda: layer.attn(xn)[0])

        # warm every expert BOTH paths route to, so timing never touches disk
        pre = Preloaded(src)
        layer.mlp.src = pre
        r = layer.mlp.route(xn)
        mx.eval(r.expert_indices)
        experts = sorted({int(v) for v in r.expert_indices.reshape(-1).tolist()})
        br_warm = empty_block_residual(1, H)
        mx.eval(layer.mlp(xn))                       # warm the xn routing
        mx.eval(layer(x, br_warm, state=LayerState())[0])   # warm the layer routing
        pre.misses = 0

        t_route = bench(lambda: layer.mlp.route(xn).expert_weights)
        t_moe = bench(lambda: layer.mlp(xn), n=5)
        t_shared = bench(lambda: layer.mlp.shared_forward(xn))

        h_lat = xn.reshape(-1, H) @ layer.mlp.down.T
        mx.eval(h_lat)
        t_one_expert = bench(lambda: layer.mlp.expert_forward(h_lat, experts[0]), n=20)

        br = empty_block_residual(1, H)
        t_layer = bench(lambda: layer(x, br, state=LayerState())[0], n=5)
        assert pre.misses == 0, f"{pre.misses} disk reads leaked into the timed region"

        print(f"\n  layer {L} ({kind}, {len(experts)} experts routed)")
        print(f"      attention          {t_attn*1e3:8.2f} ms")
        print(f"      router             {t_route*1e3:8.2f} ms")
        print(f"      one expert (3 proj){t_one_expert*1e3:8.2f} ms")
        print(f"      all {len(experts):>2} experts     {t_one_expert*len(experts)*1e3:8.2f} ms (implied)")
        print(f"      shared experts     {t_shared*1e3:8.2f} ms")
        print(f"      MoE block total    {t_moe*1e3:8.2f} ms")
        print(f"      FULL LAYER         {t_layer*1e3:8.2f} ms")
        del layer, pre

        if kind == "KDA":
            kda_layer = t_layer
        else:
            mla_layer = t_layer

    n_kda, n_mla = len(cfg.kda_layers), len(cfg.mla_layers)
    total = kda_layer * n_kda + mla_layer * n_mla
    print("\n" + "=" * 76)
    print(f"projected COMPUTE for 93 layers: "
          f"{kda_layer*1e3:.1f}ms x{n_kda} + {mla_layer*1e3:.1f}ms x{n_mla} "
          f"= {total:.1f} s/token")
    print(f"projected I/O (129.8 GiB @ 3378 MB/s measured)     = "
          f"{129.8*1024/3378:.1f} s/token")
    print(f"  => I/O-bound by {(129.8*1024/3378)/max(total,1e-9):.1f}x; "
          f"further kernel work has limited headroom")


if __name__ == "__main__":
    main()
