# Evidence ledger and measurement corrections

Historical results below predate release-preparation changes. Raw JSONL, run metadata,
output text, and parity logs are preserved. New tests do not retroactively validate old
runs, and old runs do not validate modified execution paths. Recompute aggregates with:

```sh
python tools/summarize_evidence.py
```

## Model, checkpoint, and hardware

- **2.8T parameters:** upstream model designation. The 92 × 896 × 3 expert matrices
  with dimensions 3,072 × 3,584 account for 2,722,740,830,208 logical weights before
  dense/shared/attention weights. A complete independent trainable-parameter census
  is not checked in. The manifest tool now prints logical tensor elements excluding
  quantization scales; that includes buffers and must not be mislabeled parameters.
- **~1.42 TiB:** recorded checkpoint size 1,560,860,324,864 bytes ≈ 1.419594 TiB.
  No complete shard-hash inventory or immutable original snapshot identity is retained.
  Recheck the downloaded snapshot rather than presenting this as a signed inventory.
- **93 layers:** 24 MLA, 69 KDA; 92 routed MoE layers, 896 experts, top-16 per token.
  All language-model layers execute; vision tensors in the snapshot are unused.
  Upstream MXFP4 is the stored representation, not an unquantized fp16 checkpoint.
- **64 GiB M4 Pro Mac mini:** `docs/kv-cache/baseline-system.txt` records Mac16,11 and
  68,719,476,736 bytes. “64 GB Mac mini” is the product shorthand.

## Resident-budget sweep

Source: `docs/kv-cache/sweep/r{8,16,24,32,40}-tokens.jsonl`. Each has 40 records.
The headline uses all 39 decode records (index ≥ 1), excluding prefill. A separate
steady sample excludes the first decode as well (index ≥ 2, 38 records).

| Resident GiB | All decode mean s/token | Steady mean s/token | Steady logical read GiB/token | Maximum logged MLX peak GiB |
|---:|---:|---:|---:|---:|
| 8 | 32.4275 | 32.3524 | 119.5796 | 14.9563 |
| 16 | 32.8099 | 32.7821 | 111.5776 | 22.9565 |
| 24 | 33.2027 | 33.1627 | 103.5770 | 30.9566 |
| 32 | 42.8991 | 42.3838 | 95.5742 | 38.9569 |
| 40 | 69.1949 | 69.0192 | 87.5728 | 46.1817 |

**95.57 GiB/read-token belongs to the 32 GiB configuration, not the winning 8 GiB
configuration.** These counters measure bytes requested by the loader, including
file-cache hits; they are not physical SSD bytes. Dividing them by total step time
produces effective logical-read throughput, not independent disk bandwidth. The old
`95.57 / 3.70 + 6.61` decomposition and hardware extrapolations are withdrawn.

Increasing residency hurt this particular sweep. OS page-cache competition is a
plausible explanation, not a controlled causal result or universal rule. Dense-cache
hit/miss counts do not measure routed-expert OS-cache behavior. More hardware, repeated
runs, and physical I/O/page-fault tracing are needed to separate mechanisms.

## End-to-end comparison and long context

`runs/polymath/progress.json`: 183,718.6 seconds / 2,197 output tokens.
`runs/final-validation/progress.json`: 49,164.4 seconds / 1,383 output tokens.
The like-for-like **whole-run seconds/output-token** ratio is about **2.35×**, including
prefill. The historical **2.38×** mixed a rounded baseline whole-run mean with the
final run’s 35.1758 s steady decode mean. It is not the preferred release metric.

These runs used different resident budgets (32 versus 8 GiB), different generated
lengths and context trajectories, and the final run was disturbed by a machine
microbenchmark. They support an observed configuration improvement, not an isolated
causal speedup for latent KV storage. “Same answer” is a qualitative reading of two
outputs, not a task-quality evaluation or exact transcript match.

Final validation contains 1,383 token-log records, 295 prompt tokens as rendered by
the server, and a final consumed context of 1,677 tokens. TTFT was about 544 seconds;
1,381 records after index 1 average 35.1758 s. The historical peak counter is cumulative:
a flat high-water mark does not prove flat live allocations or absence of a leak.

`runs/longctx-2k/tokens.jsonl` begins with a 1,959-token prefill taking 1,541 seconds.
It produced 150 tokens; 148 steady decode records average about 34.965 s. That log
has no peak-memory field. It does not establish million-token context support.

## Memory and MLA algebra

The ~15 GiB number is `mx.get_peak_memory()` (with a fallback in older code), an MLX
allocator metric. It excludes Python/NumPy and kernel/OS page-cache memory and cannot
be compared directly with the older 51 GB process-footprint number. Existing logs
cannot establish ~15 GiB total application memory or “no swap pressure.”

Expanded fp32 MLA payload is `24 × 96 × (192 + 128) × 4 = 2,949,120 B/token`.
Latent payload is `24 × (512 + 64) × 4 = 55,296 B/token`: **53.333×** smaller.
This excludes chunk capacity, metadata, and fixed KDA state (464,633,856 bytes in the
full configuration). Linear reassociation is mathematically equivalent but floating
point execution is not bitwise identical. Lower-precision/quantized KV options are
separate accuracy tradeoffs, not included in this fp32 ratio.

## Numerical and reliability evidence

- `docs/kv-cache/full-parity-branch.log` and `suite-logs/test_full_parity.log` record
  full-depth fp32 comparison to the Moonshot-derived CPU/PyTorch oracle: one generated
  token following five prompt tokens, worst-layer cosine **0.99999988** (layer 22),
  92/92 router selections matched, greedy token 17374 matched. The oracle substitutes
  CPU operations for CUDA/Triton kernels; this is not a CUDA-reference run. Validation
  mode previously still used the fused path despite its help text. The corrected
  unfused validation path needs a fresh checkpoint run.
- **9.5×** is a historical one-projection Metal microbenchmark (3.33 / 0.35 ms).
  **“1500× more accurate”** was a ratio of normalized maximum errors against an fp64
  projection oracle (2.95e-3 / 1.93e-6 ≈ 1,528). It is not model accuracy, general
  quality, or an end-to-end ratio. The comparison path rounds activations to bf16;
  exactly representable checkpoint weights alone do not make all arithmetic lossless.
- Historical reliability logs cover five prompts × three launches × 16 tokens, with
  matching token sequences. The old `logit_checksum` hashes only the selected id and
  rounded maximum logit, not the vocabulary. It does not prove uncorrupted reads.
  Finite diagnostics were cleared too late. New code checks full logit chunks,
  fails on nonfinite values, and hashes fp32 vocabulary logits; old fields retain
  their original meaning. Short runs do not prove leak freedom or production reliability.
- MXFP4 observed scales were 110..122 in the sampled checkpoint data, not every
  possible E8M0 code. The loader now rejects scale ranges that can underflow, overflow,
  or encode NaN on Metal. No arbitrary-scale-format conformance is claimed.
- HTTP tests exercise a mock engine; synthetic engine tests cover actual model-control
  paths. Neither establishes complete OpenAI API conformance. See `docs/API.md`.

## Measurement protocol for new results

Record git commit plus dirty status, checkpoint revision/hashes, prompt bytes/hash,
rendered token count, all flags, dependency versions, hardware, filesystem, disk port,
thermal/load conditions, swap before/after, and wall-clock boundaries. Use a fresh run
directory and exactly one model process. Report prefill/TTFT, all decode samples, warmup
policy, mean/median/tails, sample count, logical bytes, MLX active/peak memory, process
footprint and system pressure as separate metrics. Repeat configurations and alternate
order. Keep background benchmarks off. Do not turn a fitted residual into measured GPU time.
