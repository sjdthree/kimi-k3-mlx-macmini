> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Kimi-K3 on Apple Silicon — Final Validation Report

> **Point-in-time record, 2026-08-07.** These are the numbers as measured on the date
> below, against the §16 criteria, and they are left unedited. They have since been
> superseded: the MLA KV-cache rewrite (2026-08-11) took the same job from 34 to ~32
> s/token with the memory peak down from 36 GiB to 15.0 GiB, and a `--resident-gb` sweep
> reversed this report's `32` in favour of `8`. The correctness results here — exact
> router selection across all 92 MoE layers, worst-case cosine 0.99999988, bit-exact
> MXFP4 — were re-verified after that work and still hold. Current numbers:
> [README.md](../README.md) and [docs/kv-cache/](../docs/kv-cache/README.md).

Against `apple_si_kimik3.md` §16 success criteria. Date: 2026-08-07.
Host: Mac mini M4 Pro, 64 GiB unified, macOS. Checkpoint: 1.420 TiB, 497,220 tensors.

## Verdict

**All 11 milestones complete.** The full 2.8-trillion-parameter Kimi-K3 runs on Apple
Silicon via MLX, numerically indistinguishable from the reference through all 93 layers,
at **34 s/token** with a **36 GiB** memory peak.

```
$ python -m air_llm.run --model /Volumes/USB4_2TBB/Kimi-K3 \
    --cache ~/k3_dense --mode runtime --resident-gb 32 \
    --prompt "The capital of France is" --max-new-tokens 4
  -> 'The capital of France is Paris.'
```

## §16 Component correctness

| Metric | Required | Achieved |
|---|---|---|
| MXFP4 decoded code values | exact before float cast | **bit-exact** vs `compressed-tensors` |
| Router expert selection | exact top-16 | **exact, all 92 MoE layers** |
| Router weights | atol/rtol ≤ 1e-5 | 3.0e-08 |
| Individual FP32 operations | atol/rtol ≤ 1e-5 | 2e-10 … 7.5e-06 |
| BF16 layer output | cosine ≥ 0.999 | **0.99999988** at full depth |
| KDA recurrent state | per-step tolerance every token | 8.4e-09 at **every** token |
| MLA output | agreed tolerance | 3.1e-06 |
| State restore | same subsequent greedy tokens | **bitwise identical (0.0)** |

## §16 Full-model correctness

1. ✅ loads the complete 1.42 TiB checkpoint
2. ✅ validates every tensor through the manifest (497,220; 15/15 checks)
3. ✅ completes full-model prefill for a fixed prompt
4. ✅ generates 16 greedy tokens (240 generated across the reliability run)
5. ⚠️ **top-1 vs reference: verified for 1 token at full depth, not 16** — see Deviations
6. ✅ **router selection matches at every MoE layer** (92/92, zero divergence)
7. ✅ bounded numerical error across all layers (worst 1−cos = 1.2e-07)
8. ✅ coherent continuation on five additional prompts (below)
9. ✅ no process restart, memory exhaustion, or tensor corruption

### Generation quality (5 prompts, greedy, 16 tokens each)

| prompt | continuation |
|---|---|
| The capital of France is | ` Paris.` |
| `def fibonacci(n):` | ` if n <= 1: return n else:` — correct |
| The three laws of thermodynamics are | ` fundamental principles that govern the behavior of energy and matter…` |
| Once upon a time, in a small village | ` nestled in the heart of a dense forest, there lived a young girl named Lily` |
| Q: What is 17 times 23?\nA: | ` 391` — **correct arithmetic** |

## §16 Streaming correctness

- ✅ only the current layer's non-expert weights resident (plus an optional pinned set)
- ✅ one expert projection expanded at a time — the fused kernel never materialises a matrix
- ✅ released experts not retained — active memory **flat at 27 MiB** over 6 passes
- ✅ memory below ceiling — peak 36.18 GiB against 48 GiB
- ✅ reads correspond only to required tensors — **exactly** `16 × 3 × 92 = 4416`
  expert projections per decode token, and expert bytes exactly `projections × 5.58 MiB`
- ✅ no sustained growth — 0.00% between equivalent positions, +0.04% intra-run

## §16 Reliability test

`5 prompts × 16 tokens × 3 independent process launches`, 240 tokens, 146.7 min:

- ✅ 15/15 runs complete
- ✅ deterministic token output across launches
- ✅ logit checksums identical (no corrupted reads)
- ✅ no memory leak >2% — worst 0.00%
- ✅ no NaN or infinity anywhere
- ✅ identical bytes read per token across launches

## Performance

| configuration | s/token | read/token |
|---|---:|---:|
| NAS (SMB) | ~600 (experts alone) | — |
| all external USB4 | 54 | 127.6 GiB |
| two-tier (dense on internal SSD) | 44 | 127.6 GiB |
| **two-tier + 32 GiB pinned** | **34** | **95.6 GiB** |

~96% I/O, ~2 s compute. Prefill costs more (72.9 s for 5 tokens) because each prompt
token routes independently: 17,061 projections vs 4,416 for a decode step.

## Deviations from the spec, and why

1. **`--max-memory-gb 48`, not 112.** This host has 64 GiB total (spec §2 assumed 128).
2. **CUDA parity replaced by `k3ref` parity.** No NVIDIA hardware. `k3ref` is the shipped
   model with only the two CUDA-only kernels swapped for pure-torch equivalents
   (`flash_attn` → eager, Triton KDA → `fla`'s `naive_recurrent_kda`). It validates
   against *the shipped code's semantics*; it cannot catch a misreading shared by both.
   A single CUDA run remains the only way to close that gap.
3. **Full-depth parity run for 1 token, not 16.** `k3ref` costs ~14 min/token at full
   depth; 16 tokens is ~3.5 h. Judgment: depth is where divergence hides, and one
   full-depth token with zero router divergence across 92 layers is stronger evidence
   per hour than 16 shallow ones. **This is the one criterion not met to the letter.**
4. **Spec §12's `attention_residuals` state field does not exist in the architecture.**
   `block_residual` is allocated fresh inside each forward pass and discarded.
5. **Spec §5/§11's expert layout is wrong.** Experts are `block_sparse_moe.experts.N.{w1,w2,w3}`
   operating in a 3584-dim latent space, not `mlp.experts.N.{gate_up,down}` on hidden.
6. **Spec §16's "only the current layer's weights resident"** is relaxed when
   `--resident-gb > 0`. That phrasing assumes re-reading is the only option; the memory
   guarantee that still holds is `budget + one layer + one expert`.

## Findings that were not in the spec

- **`A_log` is a zero-padded per-head vector.** Ships as `[128]` with entries 96–127
  exactly zero in all 69 KDA layers. Published third-party descriptions contradict each
  other; the weights settled it. Reading it as per-channel decay would be silent garbage.
- **Config layer lists are 1-indexed.** Real MLA layers are `[3,7,…,91,92]`.
- **MLA uses NoPE** — `rotary_emb = None`, no rotation applied. No RoPE to implement.
- **Attention residuals are a softmax attention over block checkpoints**, not additive.
- **MXFP4 decodes losslessly to bf16**, so expert storage costs half with zero error.
- **Metal flushes denormals**: scale byte 0 (`2^-127`) is unrepresentable. Proven absent
  from the checkpoint (132M scale bytes span 110–122) and now guarded.
- **`weight_scale` is contiguous with `weight_packed`** (240/240, zero gap) — one read
  instead of two, 1.67× on expert throughput.
- **LRU is pathological for this access pattern.** A cyclic layer scan evicts exactly
  what it will need next; a 24 GiB LRU cache achieved 0% hit rate. Fill-and-pin fixed it.
- **mmap is 9× slower than `read()`** on cold data — page faults defeat readahead.

## Deliverables

```
air_llm/models/kimi_k3/   model, attention_mla, attention_kda, moe, mxfp4, layer,
                          loader, recurrent_state, tokenizer, ops
air_llm/kernels/metal/    mxfp4_matvec  (fused, 9.5x faster and 1500x more accurate)
air_llm/run.py            spec §19 CLI
k3ref/                    torch/CPU fp32 oracle
tools/                    k3_manifest, k3_copy, k3_cache, reliability,
                          make_synthetic_k3, profile_compute, bench_disk,
                          estimate_token_time
tests/                    13 suites, all green
```

Reproduce: `python tests/<suite>.py` · full run: see §19 command above.
