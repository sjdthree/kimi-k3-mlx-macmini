# Where the time goes, and what hardware would change

Measured with `tools/bench_machine.py` — standalone, MLX only, no checkpoint. It times
unified memory bandwidth and kernel time on the real Kimi-K3 shapes (24 MLA layers,
69 KDA layers, 92 MoE blocks at top-16 of 896 experts) with random weights.

| | Mac mini M4 Pro / 64 GiB | M5 MBP / 128 GiB |
|---|---:|---:|
| unified memory bandwidth | 188.5 GiB/s | **477.1 GiB/s** (2.53×) |
| MLA layer | 2.58 ms | 1.99 ms |
| KDA layer | 3.65 ms | 1.60 ms |
| MoE block | 6.71 ms | 2.97 ms |
| **kernel time / token** | **0.93 s** | **0.43 s** (2.2× faster) |
| dense weights (71.5 GiB) | streamed from disk | **RAM-resident** |

## Only 0.93 s of the 6.61 s "compute" is actually the GPU

Per-token cost on the Mac mini decomposes as:

    disk I/O            25.83 s   (95.57 GiB @ 3.70 GiB/s)
    kernel time          0.93 s   <- measured
    other overhead       5.68 s   <- Python, memory copies, MXFP4 decode
                        -------
                        32.44 s   (measured 32.44)

**A faster GPU can only touch the 0.93 s.** The M5's 2.2× faster kernels recover about
0.5 s per token — roughly 0.3 h on a 2,197-token job. Every projection below therefore
gives two bounds: optimistic (all 6.61 s scales with the chip) and conservative (only the
0.93 s does). The truth is nearer the conservative one.

That 5.68 s is the largest single software target left in the system — six times the
kernel time, and it costs nothing in hardware to attack.

## Projected job time (295-token prompt, 2,197 out)

| machine / expert source | s/token | job time | vs mini |
|---|---:|---:|---:|
| Mac mini, master (before this work) | 83.6 | 51.0 h | 0.4× |
| Mac mini, branch (measured) | ~35 | **~21.6 h** | 1.0× |
| M5 MBP, experts on USB4 external | 11.2–14.3 | **6.9–8.7 h** | 2.3–2.9× |
| M5 MBP, experts on internal NVMe (~5 GiB/s) | 8.0–11.1 | 4.9–6.8 h | 2.9–4.0× |
| M5 MBP, experts on fast NVMe (~6.5 GiB/s) | 6.9–10.0 | 4.2–6.1 h | 3.3–4.7× |
| Moonshot hosted API (44 t/s) | — | **52 s** | ~1,400× |

## The bottleneck moves to the drive

On the M5 at 3.0 GiB/s expert bandwidth:

    dense from RAM        0.15 s   <- was 19.33 s streaming on the mini
    experts from disk     8.02 s   <- now dominant
    compute + overhead    3.07-6.11 s

128 GiB of RAM does what the model predicted: it removes the dense term almost entirely.
What remains is 24.06 GiB/token of expert projections drawn from a ~1.3 TiB pool, which no
practical amount of RAM can hold. **On the M5 the drive matters more than the chip.**

## Rule of thumb

1. **Buy RAM until the 71.5 GiB dense set is resident** — about 128 GiB total. This is by
   far the best value: it alone takes ~20 h to ~7-9 h.
2. **Then buy the fastest NVMe you can** for the expert shards. That is the remaining
   I/O term and it does not cache.
3. **Buy a faster GPU last.** It addresses 0.93 s of a 32 s token.
4. **Before any of that, profile the 5.68 s of non-kernel overhead.** It is bigger than
   the kernel time by 6× and it is free to fix.

Even the best case here — 4.2 h — is ~290× slower than the hosted API. This hardware path
makes local viable for overnight work; it does not make it competitive on time.
