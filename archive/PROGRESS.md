> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Kimi-K3 on Apple Silicon — Milestone Status

> **Point-in-time record, 2026-08-06.** The engineering log through milestone 11, left
> unedited. Performance figures and the `--resident-gb 32` used below predate the KV-cache
> rewrite; see [README.md](../README.md) and [docs/kv-cache/](../docs/kv-cache/README.md) for
> where the project actually landed.

Plan: `K3_MLX_PLAN.md` · Spike results: `P0_RESULTS.md` · Date: 2026-08-06

## Status: 11 of 11 milestones — **full-depth parity achieved**

```
$ python -m air_llm.run --model /Volumes/USB4_2TBB/Kimi-K3 \
    --cache ~/k3_dense --mode runtime --resident-gb 32 \
    --prompt "The capital of France is" --max-new-tokens 4

  93 layers (24 MLA, 69 KDA), 896 experts top-16
  [ 0] id=17374   72.9s  read 196.46 GiB  peak 35.49 GiB  experts 17061
  [ 1] id=20829   34.8s  read  95.57 GiB  peak 36.21 GiB  experts  4416
  [ 2] id=10      34.0s  read  95.57 GiB  peak 36.17 GiB  experts  4416
  [ 3] id=427     33.8s  read  95.57 GiB  peak 36.18 GiB  experts  4416

  full: 'The capital of France is Paris.",\n+           '
```

The real 1.42 TiB checkpoint, all 93 layers, 2.8T parameters, on a Mac mini —
**peak 36 GiB**, and every decode token loads exactly `16 x 3 x 92 = 4416` expert
projections, no more.

| # | Milestone | Gate |
|---|---|---|
| P0 | Feasibility spike | 4/4 runnable (P0-1 red → resolved by hardware) |
| M1 | Manifest builder + verifier | **PASS** 15/15, 497,220 tensors in 8 s |
| M2 | `k3ref` torch/CPU fp32 oracle | **PASS** 17/17 |
| M3 | MXFP4 decoder in MLX | **PASS** 12/12, bit-exact |
| M4 | MLA attention (24 layers) | **PASS**, max\|d\| 3.1e-06 |
| M5 | KDA attention (69 layers) | **PASS**, per-token state 8.4e-09 |
| M6 | Router + latent MoE + expert streaming | **PASS**, exact top-16, cos 0.99999994 |
| M7 | Full decoder layer + block residuals | **PASS** layers 0–3, cos 1.00000000 |
| M7b | 13-layer block period | **PASS**, worst 1-cos 5.96e-08 |
| M8 | Full 93-layer forward + generation | **PASS** — "The capital of France is **Paris**" |
| M9 | Full-depth parity vs `k3ref` | **PASS** — 0 router divergence in 92 MoE layers |
| M11 | Reliability, real checkpoint | running (5x16x3, ~2.6 h) |
| M10 | Fused Metal MXFP4 matvec | **PASS** (promoted ahead of M9) |
| — | State save/restore + conv-layout fix | **PASS** 8/8 |
| — | Streaming/memory correctness (§16) | **PASS** 8/8 |
| — | Resident weight cache | **PASS** 7/7, 92.8% less I/O |
| — | Numerical modes (§13) | **PASS** 5/5, identical greedy tokens |
| — | Synthetic integration model (§15) | **PASS** 9/9, all 6 paths agree |
| M11 | Reliability harness | **BUILT**, 7/7 on the synthetic model (15/15 runs) |
| — | Tokenizer shim | **PASS** 11/11, ids identical to the shipped tokenizer |

Full suite, all green:

```
test_k3ref_layers    M2: PASS      test_moe_mlx          M6: PASS
test_mxfp4_mlx       M3: PASS      test_layer_mlx        M7: PASS
test_attention_mlx   M4/M5: PASS   test_model_mlx        M8-partial: PASS
test_mxfp4_kernel    M10: PASS     test_state_roundtrip  STATE: PASS
                                   test_streaming_memory STREAMING: PASS
```

## Numerical parity vs `k3ref` (real weights, fp32)

| Component | Metric |
|---|---|
| MXFP4 decode | **bit-exact**, bf16-lossless |
| MLA prefill / cached decode | 3.1e-06 / 2.0e-06 |
| KDA output / final state | 5.1e-09 / 1.4e-09 |
| KDA per-token state (§9) | 8.4e-09 at **every** token |
| Router top-16 selection | **exact** |
| Router weights | 3.0e-08 |
| Single routed expert (fused) | rel 4.3e-07 |
| Full latent-MoE block | cos 0.99999994 |
| Decoder layers 0–3 | cos 1.00000000 |
| Full model fwd + output-res + norm | cos 1.00000000 |
| Streamed lm_head argmax | **identical token** to full-matrix argmax |

Spec §16 asks for ≤1e-5 fp32 and cosine ≥0.999. Everything is 3–6 orders inside.

## M10: fused Metal kernel (promoted, and worth it)

| | unfused | fused |
|---|---|---|
| per projection | 3.33 ms | **0.35 ms** (9.5×) |
| per token (16×92×3) | 14.7 s | **1.5 s** |
| accuracy vs fp64 | 2.95e-03 | **1.93e-06** |

It is simultaneously ~9.5× faster and ~1500× more accurate: the fused path never
materialises the matrix, keeps weights exact, and accumulates in fp32, where the unfused
path rounds weights to bf16. `fused=False` remains only for kernel isolation.

## Streaming correctness (§16)

- expert projections loaded per token = **exactly** `top_k × 3 × MoE layers`, no over-read
- expert bytes = **exactly** projections × 5.58 MiB — 1.79% of the layer's full expert set
- peak 4.44 GiB vs 14.64 GiB for one layer's experts; ceiling enforced per layer
- active memory **flat at 27 MiB** across 6 identical passes (0.00% growth)
- resident after a pass: 0.04 GiB — state + globals only, no layer retention

## Storage (resolved)

NAS was the blocker: 102 MB/s sequential, **45 MB/s random with no concurrency scaling**.
The Zike USB4 enclosure first benchmarked at 1.0 GB/s because it sat on a **front Mac mini
port (10 Gbps) behind a hub**, enumerating as USB mass storage. On a **rear Thunderbolt
port** it enumerates as `IONVMeController`:

| | NAS | front | rear |
|---|---|---|---|
| sequential read | 102 MB/s | 1061 MB/s | **3875 MB/s** |
| random 5.25 MiB | 45 MB/s | 975 MB/s | **3378 MB/s** (p50 1.69 ms) |

No random-access penalty, which is exactly what expert streaming needs. **Single-tier**:
whole checkpoint on the external drive. Copy in progress, ~0.36/1.42 TiB, ETA ~3.2 h.

Projected floor: **~37 s I/O + ~1.5 s decode ≈ 40 s/token.**

## Bugs found and fixed during M8

- **`block_residual` was wrongly persisted across tokens.** It is allocated fresh inside
  every `forward()` (`modeling_kimi_linear.py:1188`), accumulates across *layers*, and is
  discarded. Carrying it breaks the moment prefill (N=prompt) gives way to decode (N=1).
  Spec §12's `attention_residuals` field does not match the architecture.
- **Byte accounting under-reported** — cache-path reads skipped the counter.
- **lm_head double-counted** — `row_block` already tallies what `logits()` also returned.

## Profiling changed the optimisation picture

With every weight preloaded so nothing touches disk inside the timed region:

```
compute   ~2.0 s/token   (22.2 ms x 69 KDA + 18.5 ms x 24 MLA)
I/O      ~39.3 s/token   (129.8 GiB @ 3378 MB/s)
=> I/O-bound by 20x
```

**A fused KDA kernel is not worth building.** Even making all compute free saves 2 s of a
41 s token; M10 already captured the available win. The lever is I/O, and its shape is
the opening: of 129.8 GiB per token, **105.7 GiB is dense weights byte-identical on every
token**; only 24.1 GiB is expert data.

### Resident weight cache (`--resident-gb`)

Bounded LRU over dense tensors only — experts are 98% unused per token and would evict
weights certain to be reused. Measured on 2 layers: **92.8% less I/O after warm-up**
(3.62 -> 0.26 GiB/token), 65.4% hit rate, results **bit-identical**.

| resident | dense streamed | I/O |
|---:|---:|---:|
| 0 GiB | 105.7 GiB | 39.3 s/token |
| 16 GiB | 89.7 GiB | 34.5 s/token |
| 32 GiB | 73.7 GiB | 29.6 s/token |
| 45 GiB | 60.7 GiB | **25.7 s/token** (fits a 48 GiB ceiling) |

### Numerical modes (spec section 13)

We were widening bf16 weights to fp32 on load, doubling memory for zero information —
the checkpoint is bf16. Runtime mode now reinterprets the bits directly:

- **validation**: fp32 weights, unfused path — exact parity with `k3ref`
- **runtime**: bf16 weights (`max|d| = 0.0` vs the fp32 conversion) + fused kernel,
  **half** the resident bytes, so the same hit rate at half the budget

Spec section 13 requires both modes to produce equivalent greedy tokens. Verified across
3 prompts: **identical token ids**, not merely close logits. Measured 2x faster per token
end-to-end (3.4 -> 1.6 s at 2 layers).

## Synthetic integration model (spec section 15)

A 2.7 MB miniature K3 — 3 layers (2 KDA + 1 MLA), 8 experts top-2, 1 shared, vocab 512,
`attn_res_block_size=2` — preserving every real operation: latent MoE, sigmoid router with
correction bias, `situ`, block attention residuals, NoPE MLA, `A_log` zero-padding, real
MXFP4 packing. `tools/make_synthetic_k3.py` also required writing an MXFP4 **encoder**
(verified idempotent against the M3-proven decoder).

All six required execution paths produce **identical** greedy output
`[340, 93, 254, 324, 399, 413]`:

```
1. eager full-memory        4. standard MLX kernels
2. layer streaming          5. fused Metal kernels
3. expert streaming         6. saved-state continuation
```

plus bf16 runtime mode. The same manifest builder verifies it (15/15) after the `A_log`
check was generalised from the literal `[128]` to `head_dim`, which is the real invariant.

**It immediately earned its cost.** With an MLA layer at index 2, it reached a path the
real-checkpoint tests could not: those run layers 0–1, both KDA, so `state.mla_caches`
had never round-tripped through `memory_report`/`save`. It was storing an `MLACache`
object where the serialiser expects a `(key, value)` tuple — a crash on the first MLA
layer of any real run. Now normalised at the state boundary.

## M11 reliability harness

`tools/reliability.py` runs 5 prompts x N tokens x 3 launches. "Independent process
launches" is load-bearing — an in-process loop shares MLX's allocator and compiled-kernel
cache, which is exactly what a leak test must not do — so each run is a real subprocess
writing its own section-14 JSONL.

Checks: 15/15 complete · deterministic tokens · **matching logit checksums** (identical
ids with different checksums would mean logits drifted while argmax happened to survive)
· <2% peak growth between equivalent token positions · <2% intra-run creep · no NaN/inf ·
identical bytes read per token across launches.

`--check-finite` was added to the model for this: it asserts finiteness of hidden states,
recurrent states, conv states, MLA caches, routing weights, and logits, costing one sync
per layer, so it is opt-in.

Validated end-to-end on the synthetic model: **15/15 runs, 7/7 criteria**. Ready to point
at the real checkpoint the moment M9 passes.

## Tokenizer shim

`air_llm/models/kimi_k3/tokenizer.py` rebuilds the `tiktoken.Encoding` directly from the
checkpoint's `tiktoken.model`, the `pat_str` transcribed from `tokenization_kimi.py`, and
`added_tokens_decoder` — no `trust_remote_code`, no transformers-version coupling.

**Correction to an earlier note in this document:** the shipped tokenizer was said to fail
under transformers 5.12. It does not — that claim came from `FINDINGS.md`, where it
applies to the *modeling* code. `verify_against_remote_code()` loads the shipped tokenizer
successfully and the shim matches its ids exactly on 8 probes. The shim is worth keeping
for independence, but it was not repairing a breakage.

The trap it guards: the 256-slot reserved block starts at `len(mergeable_ranks)`. Get that
offset wrong and ordinary text still tokenizes perfectly while every special token
silently shifts. Asserted directly.

`air_llm/run.py` now accepts `--prompt`/`--prompt-file` text and prints the decoded
continuation.

## M9: full 93-layer parity vs the oracle

```
token 0: k3ref 17374   mlx 17374
  [PASS] top-1 matches the oracle
  [PASS] router selection exact at all 92 MoE layers -- 0 layers differ
  [PASS] worst per-layer cosine 0.99999988 (at layer 13)
```

Both implementations carried independent state through all 93 layers, so error
accumulates rather than being re-synchronised at each boundary. 1-cos = 1.2e-07 after
full depth, against spec section 16's bar of >= 0.999.

**Router exactness is the load-bearing result.** Cosine can look perfect while one
expert differs, and one different expert means the two implementations are computing
different functions from that layer on. Zero divergence across 92 layers x 16 experts
says the numerics hold where it counts.

A test bug surfaced first: the reference router was computed as
`route(post_norm(layer_output))`, but the router consumes a mid-layer tensor (after the
attention residual, before the MoE). That reported all 92 layers differing -- the tell
being uniform failure from layer 1, whereas real drift appears at some depth and
spreads. `k3ref` already exposed the correct value via its `diag` hook.

## Measured performance (real model, decode steady state)

| configuration | s/token | read/token |
|---|---:|---:|
| NAS (SMB) | ~600 (experts alone) | — |
| all external USB4 | 54 | 127.6 GiB |
| two-tier (dense internal) | 44 | 127.6 GiB |
| **two-tier + 32 GiB pinned** | **34** | **95.6 GiB** |

**1.6x** over all-external. Prefill costs more (72.9 s for 5 tokens) because each prompt
token routes independently — 17,061 projections vs 4,416 for a decode step.

### The resident cache was broken and the full model exposed it

A 24 GiB LRU cache produced **127.58 GiB of reads per token — identical to no cache**.
LRU is pathological against this access pattern: a token walks layers 0..92 in order and
the next token repeats the walk, so by the time layer 0 comes round again layer 92 has
evicted it. Hit rate ~0 at any cache size below the working set.

Replaced with **fill-and-pin**: fill once, never evict. Hit rate becomes
`cache_size / working_set`, the best available without knowing the cycle length. Result:
127.58 -> 95.57 GiB and 54 -> 34 s/token, with identical output tokens.

The earlier "92.8% less I/O" measurement was taken on a 2-layer model where the whole
working set fit in cache, so LRU never had to evict. It did not generalise, and only the
full model could reveal that.

### mmap does not help

`np.memmap` + `mx.array` measured **389 MB/s vs 3551 MB/s** for `read()` on cold data —
9x slower. Page faults arrive one page at a time with no readahead, so the drive never
streams. An 8 MiB `read()` buffer is already correct. (This invalidates an earlier
projection of 2.2x that assumed mmap would raise loader efficiency.)

### Expert reads: one read instead of two

`weight_scale` starts exactly where `weight_packed` ends — verified 240/240 projections,
zero gap — so both come from one contiguous read. Internal expert throughput
3769 -> 6276 MB/s.

## Remaining

| # | Deliverable | Blocked on |
|---|---|---|
| M9 | 16 greedy tokens vs `k3ref`; CUDA confirmation optional | — |
| M11 | run the built harness against the real checkpoint | M9 |

At 34 s/token, M11 (5 prompts x 16 tokens x 3 launches = 240 tokens) is ~2.3 hours.

Open question for M9: the **tokenizer**. K3's `tokenization_kimi.py` targets transformers
4.56 and this env has 5.12. `air_llm/run.py` accepts `--prompt-tokens` to work around it;
a real text prompt needs either a pinned env or a direct TikToken shim.
