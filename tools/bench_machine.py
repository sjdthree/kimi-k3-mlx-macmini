"""Synthetic component microbenchmarks at K3-like shapes.

Does not measure full-model GPU time or predict end-to-end throughput.
Do not run concurrently with inference: it consumes memory and GPU resources.
"""
from __future__ import annotations

import platform
import time

import mlx.core as mx

G = 2 ** 30

# --- real Kimi-K3 dimensions (config.json) --------------------------------------
H = 7168                 # hidden_size
N_LAYERS = 93
N_MLA = 24               # layers using MLA attention; the rest are KDA
TOPK = 16                # num_experts_per_token
MOE_INT = 3072           # moe_intermediate_size
EXP_H = 3584             # routed_expert_hidden_size
N_SHARED = 2
KV_LORA, Q_LORA = 512, 1536
NH, QHD, VHD, NOPE = 96, 192, 128, 128
KDA_H, KDA_D = 96, 128

def timeit(fn, warmup=2, iters=5):
    for _ in range(warmup):
        mx.eval(fn())
    mx.synchronize() if hasattr(mx, "synchronize") else None
    t0 = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t0) / iters


def bench_membw():
    """Streaming bandwidth on buffers too large to sit in cache."""
    n = 512 * 1024 * 1024 // 2                      # 512 MiB of bf16
    a = mx.random.normal((n,)).astype(mx.bfloat16)
    b = mx.random.normal((n,)).astype(mx.bfloat16)
    mx.eval(a, b)
    t = timeit(lambda: a + b)                        # read 2, write 1
    return 3 * n * 2 / G / t


def bench_compute():
    """One token through one layer's real shapes, then scaled to 93 layers.

    Weights are random and bf16, matching runtime mode. This measures kernel time on the
    shapes that actually run -- not a generic FLOPs number.
    """
    def w(*s):
        return mx.random.normal(s).astype(mx.bfloat16)

    x = w(1, 1, H)

    # ---- MLA attention (24 layers), latent-cache form at a 2k context
    T = 2048
    q_a, q_b = w(Q_LORA, H), w(NH * QHD, Q_LORA)
    kv_a, kv_b = w(KV_LORA + 64, H), w(NH * (NOPE + VHD), KV_LORA)
    o_w, g_w = w(H, NH * VHD), w(NH * VHD, H)   # gate projects to nh*vhd, not hidden
    lat, rot = w(1, T, KV_LORA), w(1, T, 64)
    wk = kv_b.reshape(NH, NOPE + VHD, KV_LORA)[:, :NOPE, :][None]
    wv = kv_b.reshape(NH, NOPE + VHD, KV_LORA)[:, NOPE:, :][None]

    def mla():
        q = (x @ q_a.T) @ q_b.T
        q = q.reshape(1, 1, NH, QHD).transpose(0, 2, 1, 3)
        ckv = x @ kv_a.T
        qa = q[..., :NOPE] @ wk
        s = qa @ mx.expand_dims(lat.transpose(0, 2, 1), 1)
        s = s + q[..., NOPE:] @ mx.expand_dims(rot.transpose(0, 2, 1), 1)
        p = mx.softmax(s.astype(mx.float32), axis=-1).astype(mx.bfloat16)
        ctx = p @ mx.expand_dims(lat, 1)
        o = (ctx @ wv.transpose(0, 1, 3, 2)).transpose(0, 2, 1, 3).reshape(1, 1, NH * VHD)
        return (o * mx.sigmoid(x @ g_w.T)) @ o_w.T

    # ---- KDA attention (69 layers): projections + one recurrence step
    kq, kk, kv_, kg = (w(KDA_H * KDA_D, H) for _ in range(4))
    ko = w(H, KDA_H * KDA_D)
    S = mx.random.normal((1, KDA_H, KDA_D, KDA_D))

    def kda():
        qq = (x @ kq.T).reshape(1, 1, KDA_H, KDA_D).astype(mx.float32)
        kk_ = (x @ kk.T).reshape(1, 1, KDA_H, KDA_D).astype(mx.float32)
        vv = (x @ kv_.T).reshape(1, 1, KDA_H, KDA_D).astype(mx.float32)
        St = S * 0.99
        kS = mx.sum(mx.expand_dims(kk_[:, 0], -1) * St, axis=-2)
        St = St + mx.expand_dims(kk_[:, 0], -1) * mx.expand_dims(vv[:, 0] - kS, -2)
        o = mx.sum(mx.expand_dims(qq[:, 0], -1) * St, axis=-2)
        return o.reshape(1, 1, KDA_H * KDA_D).astype(mx.bfloat16) @ ko.T

    # ---- MoE: top-16 routed experts + 2 shared, gate/up/down each
    eg, eu = w(TOPK, MOE_INT, EXP_H), w(TOPK, MOE_INT, EXP_H)
    ed = w(TOPK, EXP_H, MOE_INT)
    din, dout = w(EXP_H, H), w(H, EXP_H)
    sg, su, sd = w(N_SHARED * MOE_INT, H), w(N_SHARED * MOE_INT, H), w(H, N_SHARED * MOE_INT)

    def moe():
        h = x @ din.T                                     # into expert latent space
        hx = mx.broadcast_to(h, (TOPK, 1, EXP_H))
        a = mx.matmul(hx, eg.transpose(0, 2, 1))
        b = mx.matmul(hx, eu.transpose(0, 2, 1))
        y = mx.matmul(a * mx.sigmoid(b), ed.transpose(0, 2, 1))
        routed = mx.sum(y, axis=0) @ dout.T
        s = (x @ sg.T) * mx.sigmoid(x @ su.T)
        return routed + s @ sd.T

    t_mla, t_kda, t_moe = timeit(mla), timeit(kda), timeit(moe)
    total = N_MLA * t_mla + (N_LAYERS - N_MLA) * t_kda + (N_LAYERS - 1) * t_moe
    return t_mla, t_kda, t_moe, total


def main():
    print(f"machine : {platform.machine()}  {platform.platform()}")
    try:
        import subprocess
        m = subprocess.run(["sysctl", "-n", "hw.model", "hw.memsize"],
                           capture_output=True, text=True).stdout.split()
        print(f"model   : {m[0]}   RAM {int(m[1])/G:.0f} GiB")
        ram_gib = int(m[1]) / G
    except Exception:
        ram_gib = 0

    print("\n--- measuring ---")
    bw = bench_membw()
    print(f"  unified memory bandwidth : {bw:8.1f} GiB/s")
    t_mla, t_kda, t_moe, compute = bench_compute()
    print(f"  MLA  layer               : {t_mla*1000:8.2f} ms  x{N_MLA}")
    print(f"  KDA  layer               : {t_kda*1000:8.2f} ms  x{N_LAYERS-N_MLA}")
    print(f"  MoE  block               : {t_moe*1000:8.2f} ms  x{N_LAYERS-1}")
    print(f"  scaled synthetic time    : {compute:8.2f} s")

    print("Synthetic scaled component sum only; excludes real routing, I/O and execution overhead.")


if __name__ == "__main__":
    main()
