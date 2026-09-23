> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# P0 Feasibility Spike — Results (2026-08-06)

Gates defined in `K3_MLX_PLAN.md` §7. Scripts in `p0/`. Env: conda `airllm`, arm64 Python 3.12.13.

| Task | Gate | Result |
|---|---|---|
| P0-1 SMB throughput | ≥100 MB/s random read | **FAIL — 36–48 MB/s, no concurrency scaling** |
| P0-2 KDA `A_log` semantics | unambiguous, source-backed | **PASS** |
| P0-3 MXFP4 decode | bit-exact vs `compressed-tensors` | **PASS** |
| P0-4 Real KDA layer on CPU | runs, finite, deterministic | **PASS** |
| P0-5 Stage dense locally | ≤130 GB, byte-verified | **not started** — gated on the P0-1 decision |

Overall: **3 of 4 runnable gates green.** The two hardest questions in the project (KDA
semantics, MXFP4 layout) are both resolved. The one failure is an I/O problem, not a
correctness problem.

---

## P0-2 — KDA `A_log` resolved (`p0/p0_2_alog.py`)

The `K3_MLX_PLAN.md` §1 finding was that the checkpoint ships `A_log F32 [128]` while the
shipped `modeling_kimi_linear.py` declares `[num_heads] = [96]`. Confirmed the same
declaration in the **live** HF copy, so it is not a stale local file.

Two competing readings existed. Web sources asserted both, contradicting each other:
1. decay is **per-K-channel [128]**, broadcast across heads;
2. it is a **per-head [96]** vector **zero-padded to 128**.

Tested directly against the weights. Across **all 69 KDA layers**, without exception:

```
A_log[96:128] == 0.0 exactly   (0 nonzero of 32)
A_log[:96]    all nonzero      (96 nonzero of 96)
```

**Reading 2 is correct.** Slice `A_log[:96]`; fla's published contract holds unmodified.
Reading 1 would have been catastrophic and silent — `exp(0)=1` on channels 96–127 plus a
head/channel axis swap.

> Do not trust the search-result summaries on this. One of them stated reading 1 confidently.
> The weights are the only authority.

Full semantics, each pinned to upstream source:

```
A_log      A_log[:96], per value head            (zero-padded to 128 on disk)
dt_bias    [12288] -> view(96, 128) = (head, k_channel)
gate       safe_gate + lower_bound=-5.0:
             g = -5.0 * sigmoid(exp(A_log)[96,1] * (g_raw + dt_bias[96,128]))
           fla/ops/kda/gate.py::naive_kda_lowerbound_gate
q, k       L2-normalised: x / sqrt(sum(x^2) + 1e-6)   fla/modules/l2norm.py:105
beta       sigmoid(b_proj(x))    allow_neg_eigval=False => scale 1.0
short conv causal depthwise k=4, then silu, no bias
o_norm     gated RMSNorm: y = x*rstd*w*sigmoid(g), eps=1e-5
           fla/modules/fused_norm_gate.py:104
recurrence fla/ops/kda/naive.py::naive_recurrent_kda, scale = 128**-0.5
state      [B, HV, K, V]; `transpose_state_layout` is DEPRECATED, renamed
           `state_v_first` -> stored V-first. Affects serialisation, not math.
```

## P0-3 — MXFP4 decode is bit-exact (`p0/p0_3_mxfp4.py`)

NumPy reference decoder matches `compressed-tensors` 0.17.1 **exactly**:

- all 256 byte values / 16 FP4 codes — PASS
- all 256 E8M0 scale bytes — PASS
- 6 real expert tensors across layers 1/40/92 — **bit-exact**
- validated against the library's **public** `MXFP4PackedCompressor.decompress`, after
  confirming `MXFP4PackedCompressor` extends `NVFP4PackedCompressor` and overrides only
  scale encoding (so `nvfp4/helpers.py` is the right packing oracle)

```
nibble order  low nibble = even element (2i), high nibble = odd (2i+1)
code          bit3 = sign (1 => negative), bits 0-2 = index into
              [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
scale         E8M0, value = 2**(scale_byte - 127)
grouping      group_size 32 along the INPUT axis (last axis of the packed tensor)
```

**Bonus finding:** every decoded value is exactly representable in bf16 (verified by
fp32→bf16→fp32 round-trip on a full real tensor). Expert weights can be held in bf16 with
**zero** error — halves the expanded-expert working set at no numerical cost. The library
itself decodes to bf16.

Observed on real experts: scale bytes span `2^-15 … 2^-5`, ~11.5% exact zeros,
|w| ≤ 0.125, mean|w| ≈ 0.019.

## P0-4 — Real KDA layer runs on CPU (`p0/p0_4_kda_layer.py`)

Layer 1, real weights, fp32, T=8. Loads and runs in ~22 s (dominated by NAS I/O).

```
output (1, 8, 7168)   state (1, 96, 128, 128)
finite: yes      deterministic across runs: yes (sha256 match on output and state)
prefill(8) vs 8x single-token decode:  max|d out| 4.7e-09,  max|d state| 2.0e-08
```

The prefill-vs-step agreement matters more than it looks: the streaming runtime will
generate one token at a time, so it exercises the recurrent path plus conv-state
threading. Those now provably agree with the batch path.

This is the seed of the `k3ref` torch-CPU oracle (plan §3). `KDALayer` in that script is
directly reusable for M2.

## P0-1 — SMB throughput FAILS the gate (`p0/p0_1_smb_bench.py`, `p0_1b_parallel.py`)

```
sequential 8 MiB blocks           102.6 MB/s
random 5.25 MiB reads (serial)     36.2 MB/s   p50 latency 141 ms, p90 190 ms
random, 4 threads                  47.6 MB/s
random, 8 threads                  44.1 MB/s
random, 16 threads                 46.0 MB/s
random, 32 threads                 43.2 MB/s
```

**Concurrency does not help at all** — flat ~45 MB/s from 4 to 32 workers. So this is not
request latency that parallelism can hide; it is a hard ceiling on the path.

Consequences at ~45 MB/s:
- routed experts alone: 24.1 GiB/token → **~550–600 s/token (~10 min)**
- if dense weights also stayed on the NAS: **+3380 s/token**. Local dense staging is
  mandatory, not an optimisation.

The Mac's `en0` negotiates **10Gbase-T**, so the 102 MB/s sequential figure (≈ 1 GbE) points
at the NAS link, an intermediate switch, or SMB configuration — not the Mac. There may be
real headroom here, but it is a hardware/network question, not a code one.

**This gate is red and needs a decision before M8+ becomes practical.** It does not block
M1–M7: component and single-layer work needs only a handful of layers, which cache locally
in seconds.

---

## Amendments to `K3_MLX_PLAN.md`

- §1 row 6 / §4: dense-local staging is now **required**, not merely advantageous.
- §7 P0-2: resolved — record `A_log[:96]` in the implementation notes as a checkpoint-specific
  layout quirk (spec §18 deliverable).
- §4: expanded experts should be **bf16, not fp32** — proven lossless in P0-3.
- New risk: published third-party descriptions of K3's KDA are contradictory and at least one
  is wrong. Derive from weights + upstream source only.
