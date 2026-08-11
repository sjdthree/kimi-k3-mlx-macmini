# Kimi-K3 on Apple Silicon (MLX)

Runs the full **2.8-trillion-parameter Kimi-K3** — the complete 1.42 TiB checkpoint, all
93 layers — on a Mac mini, at **~32 s/token** with a **15 GiB** memory peak.

```
$ python -m air_llm.run --model /path/to/Kimi-K3 \
    --cache ~/k3_dense --mode runtime --resident-gb 8 \
    --prompt "The capital of France is" --max-new-tokens 4

  93 layers (24 MLA, 69 KDA), 896 experts top-16
  [ 0] id=17374   72.9s  read 196.46 GiB  peak 35.49 GiB  experts 17061
  [ 1] id=20829   34.8s  read  95.57 GiB  peak 36.21 GiB  experts  4416
  ...
  full: 'The capital of France is Paris.'
```

*(transcript predates the KV-cache work below; peak is now ~15 GiB and `--resident-gb 8`
beats 32 — see [Latest](#latest-2026-08-11--kv-cache-rewrite-and-what-it-uncovered))*

## Latest (2026-08-11) — KV-cache rewrite, and what it uncovered

Validated end to end by re-running the original 2,500-token job and comparing against it:

| | before | after |
|---|---:|---:|
| s/token | 83.6 | **35.2** (2.38×; ~32.4 clean) |
| s/token growth per 1k context | +11.2 (+52.4 early) | **+1.8** — flat |
| memory peak | 51 GB footprint | **15.0 GiB, never moved** |
| swap | 24.95 of 25.6 GiB | unchanged |
| MLA KV cache | 2,949,120 B/token | **55,296 B/token** (53.3×, lossless) |

Same answer: the model ended its own audit in 1,383 tokens instead of 2,197, finding every
planted defect and covering several *more* thoroughly than the baseline.

**What changed.** MLA cached the key/value expanded per head; `kv_b` is a fixed linear map,
so caching the 512-wide latent and absorbing `kv_b` into the query and output projections
is exactly equivalent for 53.3× less memory — verified to 3.0e-07 against an independent
fp64 oracle, with `test_full_parity` returning the identical token and worst-case cosine
(0.99999988) before and after. Appends are chunked too; master reallocated and copied the
whole cache every decode step.

**Three findings that outlast the change**, all in
[docs/kv-cache/](docs/kv-cache/README.md):

- **Per-token cost is `95.6 GiB ÷ read bandwidth`.** The cache mattered, but the ceiling
  is I/O, not attention.
- **More pinned memory is monotonically *worse*.** Measured across a 5× sweep: 40 GiB →
  69.0 s/token, 8 GiB → 32.4. Fill-and-pin starves the OS page cache, which is what
  actually serves the 105.7 GiB working set. This reverses a standing project assumption.
- **Only 0.93 s of the 6.61 s "compute" is GPU kernel time.** The other 5.68 s is Python,
  memory copies and MXFP4 decode — the largest untouched target in the system, and free to
  fix. See [MACHINE-COMPARISON.md](docs/kv-cache/MACHINE-COMPARISON.md) for what hardware
  would and would not buy.

Long prompts also work now: a 1,955-token prompt used to fail in prefill at layer 0 on
master *and* branch, and never had.

K3 is CUDA-locked in its own remote modeling code — flash-attn is force-enabled and the
KDA linear-attention layers call Triton kernels, neither of which has a Metal backend.
This is a from-scratch MLX implementation of the architecture, validated against a
torch/CPU reference built from the shipped code.

## Docs

| | |
|---|---|
| **[SESSION.md](SESSION.md)** | current state, what's running, how to resume |
| **[QUICKSTART.md](QUICKSTART.md)** | checkout → generated token in ~5 min |
| **[docs/API.md](docs/API.md)** | OpenAI-compatible `/v1` server: compatibility matrix, examples, ops |
| **[INSTALL.md](INSTALL.md)** | hardware, storage, dependencies, troubleshooting |
| **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)** | results against every spec criterion |
| [PROGRESS.md](PROGRESS.md) | milestone-by-milestone engineering record |
| [K3_MLX_PLAN.md](K3_MLX_PLAN.md) | the implementation plan, and where the spec was wrong |

## Status

All 11 milestones complete. See **[VALIDATION_REPORT.md](VALIDATION_REPORT.md)**.

| | |
|---|---|
| Router selection vs reference | **exact, all 92 MoE layers** |
| Worst per-layer cosine (full depth) | **0.99999988** |
| MXFP4 decode | **bit-exact** vs `compressed-tensors` |
| Reliability (5 prompts × 16 tokens × 3 launches) | **15/15**, no leak, no NaN |

Generation is coherent and correct — `17 × 23` → `391`, and a working `fibonacci`
implementation.

## Layout

| path | what |
|---|---|
| `air_llm/models/kimi_k3/` | the MLX model — MLA, KDA, latent MoE, MXFP4, state, tokenizer |
| `air_llm/kernels/metal/` | fused MXFP4 matvec (9.5× faster, 1500× more accurate than unfused) |
| `air_llm/run.py` | CLI |
| `air_llm/server/` | OpenAI-compatible `/v1` API (FastAPI) + `/native/v1` |
| `k3ref/` | torch/CPU fp32 oracle — the parity reference |
| `tools/` | manifest builder, weight staging, reliability harness, profilers, long-job runner |
| `prompts/` | architecture-targeted prompts (KDA, AttnRes, NoPE, LatentMoE) |
| `tests/` | 13 suites |
| `p0/` | feasibility-spike scripts (see [P0_RESULTS.md](P0_RESULTS.md)) |

## Setup

Apple Silicon Mac, arm64 Python 3.11/3.12, and 1.42 TiB of **fast local** storage.
The model itself needs only `mlx`, `numpy`, and `tiktoken` — torch is required solely
by the correctness oracle.

```bash
pip install mlx numpy tiktoken

# index the checkpoint (~8 s, 497,220 tensors)
python tools/k3_manifest.py build --ckpt /path/to/Kimi-K3 --out manifest.jsonl.gz
python tools/k3_manifest.py verify --ckpt /path/to/Kimi-K3

# run
python -m air_llm.run --model /path/to/Kimi-K3 \
    --mode runtime --prompt "..." --max-new-tokens 16
```

Full walkthrough in **[QUICKSTART.md](QUICKSTART.md)**; hardware, storage layout and
troubleshooting in **[INSTALL.md](INSTALL.md)**.

`--mode validation` uses fp32 weights and the unfused expert path for exact parity work;
`runtime` uses bf16 (the on-disk dtype, lossless) and the fused kernel. Both produce
identical greedy tokens.

## OpenAI-compatible API

```bash
export AIR_LLM_API_KEYS=sk-your-key
python -m air_llm.serve --model /path/to/Kimi-K3 --cache ~/k3_dense \
    --mode runtime --resident-gb 32 --port 8000
```

Any OpenAI SDK works with only `base_url` + `api_key` — verified by running the official
SDK against it (29 conformance checks). Chat completions, SSE streaming, logprobs,
sampling, cancellation, auth, quotas, Prometheus metrics.

**Tools, JSON-schema output, vision and embeddings are deliberately reported as
unsupported** rather than approximated: there is no constrained decoding in this backend
and no embedding model deployed, so those requests return a structured
`unsupported_capability` error. See [docs/API.md](docs/API.md) for the full matrix.

Note the backend is disk-bound at ~34 s/token, so clients need timeouts ≥ 900 s and
should prefer `stream=true`.

## Performance notes

~96% of a token is I/O; compute is ~2 s. Storage dominates:

| | s/token |
|---|---:|
| NAS over SMB | ~600 (experts alone) |
| external USB4 NVMe | 54 |
| + dense on internal SSD | 44 |
| + 32 GiB pinned in RAM | **34** |

Two counter-intuitive results worth knowing before optimizing:

- **LRU caching is pathological here.** A token walks layers 0→92 and the next token
  repeats the walk, so LRU evicts precisely what it needs next — a 24 GiB LRU cache
  measured a **0% hit rate**. `--resident-gb` uses fill-and-pin instead.
- **mmap is 9× slower than `read()`** on cold data (389 vs 3551 MB/s); page faults
  defeat readahead.

## Checkpoint findings

Several details are absent from or contradicted by the available documentation, and
getting them wrong produces silently wrong output rather than errors:

- `A_log` ships as `[128]` but is a **zero-padded per-head `[96]` vector** (entries
  96–127 are exactly zero in all 69 KDA layers). Published descriptions disagree; the
  weights settle it.
- `config.json`'s layer lists are **1-indexed** — real MLA layers are `[3,7,…,91,92]`.
- MLA uses **NoPE**: `rotary_emb = None` and no rotation is applied.
- Experts operate in a **3584-dim latent space** via `routed_expert_{down,up}_proj`,
  not on the 7168-dim hidden state.
- Attention residuals are a **softmax attention over block checkpoints**, not additive.
- The router's `e_score_correction_bias` affects **selection only**; weights come from
  the raw sigmoid scores.

## Credits

`third_party/kda_naive.py` is `fla/ops/kda/naive.py` from
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT),
vendored unmodified as the KDA reference.

AirLLM (MIT): https://github.com/lyogavin/airllm — the per-expert streaming approach this
work builds on. Gavin Li's write-up of running K3 on a single 4 GB GPU under CUDA is the
original motivation for the port:
["Unbelievable! Run Kimi K3 — 2.8 Trillion Parameters — on a Single 4GB GPU"](https://medium.com/@lyo.gavin).

Kimi-K3 weights and modeling code: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3).
