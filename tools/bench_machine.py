"""Estimate Kimi-K3 s/token on a machine WITHOUT copying the 1.42 TiB checkpoint.

Per-token cost decomposes into three terms, all measured on the Mac mini (M4, 64 GiB):

    disk I/O          95.57 GiB/token  @ 3.70 GiB/s  =  25.8 s
    compute+overhead                                 =   6.6 s
                                                        ------
                                                        32.4 s   (measured 32.44)

and the 95.57 GiB splits into 71.52 GiB of DENSE weights -- the same bytes every token,
so perfectly cacheable -- and 24.06 GiB of EXPERT projections drawn from a ~1.3 TiB pool.

A machine with >=128 GiB holds the dense set in RAM permanently, so its disk term collapses
to experts only. What that machine then costs per token depends on two things this script
measures directly, with random weights at the real model dimensions:

    1. unified memory bandwidth   -- how fast the resident dense weights stream
    2. compute throughput         -- the actual matmul shapes, one layer, x93

Nothing here touches the checkpoint. Run it on both machines and compare.

    python tools/bench_machine.py
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

# --- measured on the reference machine (Mac mini M4, 64 GiB) --------------------
REF = dict(name="Mac mini M4 Pro / 64 GiB", s_per_token=32.44, disk_bw=3.70,
           dense_gib=71.52, expert_gib=24.06, compute_s=6.61,
           kernel_s=0.93, membw=188.5)   # measured by running this script there


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
    print(f"  kernel time per token    : {compute:8.2f} s")

    print(f"\n--- reference: {REF['name']} ---")
    print(f"  measured                 : {REF['s_per_token']:8.2f} s/token")
    print(f"  of which compute+overhead: {REF['compute_s']:8.2f} s")

    # Compute+overhead is scaled by this machine's measured kernel time relative to the
    # reference's. That assumes the non-kernel part of the 6.61 s (Python, memory copies,
    # MXFP4 decode) scales with kernel speed. It is an assumption, not a measurement --
    # stated plainly so it can be argued with rather than trusted.
    scale = compute / REF["kernel_s"]
    non_kernel = REF["compute_s"] - REF["kernel_s"]      # 5.68 s on the reference
    holds_dense = ram_gib >= 110

    # Two bounds, because the split matters more than either number.
    #   optimistic  : all 6.61 s scales with kernel speed
    #   conservative: only the 0.93 s of real kernel time scales; the 5.68 s of Python,
    #                 memory copies and MXFP4 decode is a fixed cost a faster GPU cannot touch
    oh_opt = REF["compute_s"] * scale
    oh_con = compute + non_kernel

    print(f"\n--- projection for THIS machine ---")
    print(f"  memory bandwidth vs ref  : {bw/REF['membw']:8.2f}x")
    print(f"  kernel time vs ref       : {scale:8.2f}x   "
          f"({REF['kernel_s']:.2f} s -> {compute:.2f} s)")
    print(f"  dense weights            : "
          f"{'RAM-RESIDENT (fits)' if holds_dense else 'streamed from disk'}"
          f"   [{ram_gib:.0f} GiB present, need ~110]")
    print(f"  compute+overhead         : {oh_opt:5.2f} s optimistic / {oh_con:5.2f} s conservative")
    print()
    for disk_bw in (1.0, 3.0, 6.0):
        dense_t = REF["dense_gib"] / bw if holds_dense else REF["dense_gib"] / disk_bw
        io = dense_t + REF["expert_gib"] / disk_bw
        lo, hi = io + oh_opt, io + oh_con
        print(f"  expert disk {disk_bw:>4.1f} GiB/s -> {lo:5.1f}-{hi:5.1f} s/token   "
              f"{REF['s_per_token']/hi:4.1f}-{REF['s_per_token']/lo:4.1f}x vs Mac mini   "
              f"job {lo*2197/3600:4.1f}-{hi*2197/3600:4.1f} h")
    print("\nPASTE EVERYTHING ABOVE BACK.")


if __name__ == "__main__":
    main()
