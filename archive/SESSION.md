> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Session state — 2026-08-07, updated 2026-08-09

> **Archived working notes.** This is the author's live scratchpad as it stood on
> 2026-08-09, kept because it records what was running, what was uncertain, and what the
> open questions were at the time. PIDs, in-flight jobs and "right now" are all historical.
> For the project as it stands, start at [README.md](../README.md) and
> [VALIDATION_REPORT.md](VALIDATION_REPORT.md).

Picking this up after a break? Read this first, then `PROGRESS.md` for the engineering
record and `VALIDATION_REPORT.md` for results.

**Resuming on 2026-08-09 or later? Two things are in flight at once:** the long run below
(on `master`), and an unmerged branch waiting on it. Jump to
[Branch in flight](#branch-in-flight-mla-latent-cache).

## What exists

The full 2.8-trillion-parameter Kimi-K3 runs on this Mac mini via MLX: the complete
1.42 TiB checkpoint, all 93 layers, **34 s/token**, 36 GiB peak. Numerically
indistinguishable from a torch/CPU reference through the entire stack — **zero router
divergence across all 92 MoE layers**, worst per-layer cosine 0.99999988.

All 11 spec milestones complete. 14 test suites green. An OpenAI-compatible `/v1` API
serves it, verified with the official OpenAI SDK (29/29 conformance checks).

## What is running right now

A 24-hour generation was launched at **09:48 on 2026-08-07**.

| | |
|---|---|
| job | `tools/run_long_job.py`, PID 99281 (plus `caffeinate` PID 99282 holding sleep assertions) |
| prompt | `prompts/04_polymath_multi_expert.md` — CUDA / cryptography / formal-verification audit |
| target | 2,500 tokens ≈ 23.6 h decode + ~10 min prefill |
| output | `runs/polymath/` (gitignored while in flight) |
| server | `python -m air_llm.serve`, PID 98807, detached under launchd, port 8080, 32 GiB pinned, warm |

### Checking on it

```bash
cat runs/polymath/progress.json      # percent, s/token, ETA, predicted finish
tail -f runs/polymath/output.txt     # the audit as it is written
tail -f runs/polymath/job.log        # milestones and errors
curl -s -H "Authorization: Bearer sk-k3" localhost:8080/native/v1/engine | python -m json.tool
```

### If it died

Every token was fsynced, so whatever is in `output.txt` is intact. Resume costs one
prefill (~10-20 min) rather than the whole run:

```bash
conda activate airllm && cd ~/kimi-k3-mlx
# N = tokens_generated_total from progress.json
python tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
    --out runs/polymath --max-tokens 2500 --resume --resume-token-count N
```

If the **server** died, restart it first — the job cannot do that itself, and after 3
retries at 60 s it gives up. This is the single most likely way to lose the run:

```bash
export AIR_LLM_API_KEYS=sk-k3
nohup python -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
    --manifest manifest_local.jsonl.gz --cache ~/k3_dense \
    --mode runtime --resident-gb 32 --port 8080 --max-output-tokens 3000 \
    > /tmp/serve.log 2>&1 &
```

If the run finished, keep the result deliberately: `git add -f runs/polymath/output.txt`.

### It finished 2026-08-09T12:50:44

`finish_reason=stop` at **2,197 tokens** — the model ended its answer rather than reaching
the 2,500 cap. 2 d 3 h 02 m, no resumes, 83.6 s/token mean. Artifacts are committed
(`runs/` is gitignored; they were added with `-f`). Prompt is **295 tokens**.

It slowed markedly over the run: **57.9 s/token early, 80.0 late**, against the 34
s/token this file records as achieved. See `docs/kv-cache/README.md` for the full curve
and why the shape argues against the KV cache being the cause.

## A/B complete — the branch works, twice measured

Full write-up in `docs/kv-cache/README.md`. Both runs' artifacts are committed.

| run | context | s/token | vs master |
|---|---:|---:|---|
| `polymath-latent-300` | 296–594 | **48.10** | master 60.09 → **−20%** |
| `longctx-2k` | 1,960–2,108 | **34.97** | master 90.6 → **−61%** |

Context scaling is flat: slope +6.6 s/tok per 1k at short context, **−4.33** at 2k, against
master's +49.4. The branch is *faster* at context 2,108 than at 595.

**The finding that matters more than the speedup.** Comparing the two branch runs
token-for-token, `bytes_read` (102,622,070,784), `expert_bytes` and `experts_loaded` are
byte-for-byte identical per token, while effective throughput went 2.08 → 2.73 GiB/s. Same
bytes, same experts, 31% faster. The only remaining variable is where the bytes came from —
pinned resident cache versus disk. The 1,959-token prefill warmed the cache far more than
the 7-token smoke request that preceded the short run.

So: **once the KV cache is fixed, per-token cost is governed by resident-cache hit rate,
not context length.** The old 90 s/token had three contributors — the KV cache, ~40 h of
accumulated swap, and an under-warmed resident cache — and attributing it wholly to any one
of them was wrong.

## What is running right now (2026-08-10 16:54)

**The final validation run.** Prompt 04, 2,500-token target — the exact baseline job —
on the branch at `--resident-gb 8`. `runs/final-validation`, server PID 49436, job PID
49445 under `caffeinate`. ETA ~13:00 on 2026-08-11.

| | baseline | this run |
|---|---:|---:|
| resident | 32 GiB | **8 GiB** |
| s/token | 83.6 | **~32.4 expected** |
| wall clock | 51.0 h | **~20 h expected** |
| peak / swap | 51 GB footprint, 25 GiB swap | **~15 GiB peak** |

Three questions it answers: does it reach the same substantive result, is it much faster,
and is the memory stress gone. On the first — do **not** expect identical text. The server
samples, and fp32 reassociation can move a logit past a tie. Compare on substance: does it
still find the `(t + m*N) >> 64` undefined behaviour, the single-limb Montgomery failure
for 256/384-bit curves, and the `__umul64hi` correction?

```bash
cat runs/final-validation/progress.json
diff <(fold -w100 runs/polymath/output.txt) <(fold -w100 runs/final-validation/output.txt)
```

## Branch in flight: `mla-latent-cache`

Four commits, **not merged, not yet validated against real weights.** Everything about
the comparison — baseline, procedure, predictions — is in `docs/kv-cache/README.md`.
Read that before doing anything else with this branch.

**What it changes.** `master` cached the MLA key/value expanded per head. The tensors
those are built from are 53.3x smaller, and `kv_b` absorbs into the query and output
projections, so caching the latent instead is exact rather than approximate:
**2,949,120 -> 55,296 B/token** across the 24 MLA layers, which at the run's 2,371-token
context is **6.51 GiB -> 0.12 GiB**. Appends are also chunked now; master reallocated and
copied the whole cache every decode step (measured: 255 reallocations per 256 appends,
now 1).

Verified lossless to **3.0e-07** against an independent fp64 oracle and against a frozen
copy of the master path, holding across every decode step. The synthetic model returns
the same six tokens as master across all six execution paths.

**Real-weight validation: DONE 2026-08-09, all 9 checkpoint suites pass.** The decisive
one is `test_full_parity` (M9, 93 layers, 3,095 s):

    token 0 top-1 vs k3ref oracle    17374 == 17374
    router selection                 exact at all 92 MoE layers, 0 differ
    worst per-layer cosine           0.99999988 at layer 22

That cosine is identical to the figure this file already records for master, so the
branch reproduces master's numerics layer-for-layer on the real checkpoint.

**Do not over-trust the causal story.** The slowdown is front-loaded and saturating, not
linear: +53.4 s/tok per 1k context over the first third, then **-7.9** over the last
third, plateauing around token 800. A per-token O(T) copy would keep costing more
forever; it does not. And MoE expert-routing diversity produces the same curve shape with
no involvement from the KV cache. Both mechanisms fit the data — the A/B is what
discriminates them. The memory result (6.51 GiB -> 0.12 GiB) is arithmetic plus a
losslessness proof and stands either way.

Quantised latent storage (`mla_cache_bits`, `air_llm/models/kimi_k3/mla_quant.py`) is
implemented and **off by default**; it saves ~90 MiB at this context and only earns its
place past ~32k. Its quality gates run on random weights, which makes the latent Gaussian
by CLT — so `test_mla_quant.py` gate Q6, on whether the TurboQuant/VeloxQuant rotation
trick transfers here, is **inconclusive rather than negative** until re-run on real
latents.

State format went to **v2**; a v1 file cannot be upgraded (the expansion through `kv_b`
is not invertible) and `load()` refuses it with that reason.

## Hardware / data layout

| | |
|---|---|
| checkpoint | `/Volumes/USB4_2TBB/Kimi-K3` — 1.42 TiB, USB4 NVMe on a **rear** Thunderbolt port |
| dense stage | `~/k3_dense` — 105.7 GiB on the internal SSD (two-tier: dense internal, experts external) |
| manifest | `manifest_local.jsonl.gz` (gitignored; rebuild in ~8 s) |
| env | conda `airllm`, arm64 Python 3.12 — `conda init zsh` was run, backup at `~/.zshrc.bak-preconda` |

The NAS copy at `/Volumes/LLMModels/Kimi-K3` is the original but is **unusable for
inference** (45 MB/s on the random reads this workload needs).

## Open threads

0. **Expert residency is now the bottleneck, not the KV cache.** The A/B settled it:
   identical bytes read per token, 31% throughput difference, explained only by resident
   cache hit rate. `--resident-gb` uses fill-and-pin, so it keeps whatever it saw FIRST,
   which is not necessarily the most-routed 30% of a 105.7 GiB working set. Pinning by
   observed routing frequency is the obvious unexplored win, and needs per-expert hit
   counters that do not exist yet.
0b. **Long prompts have a memory wall in prefill, cause still unknown.** A 1,955-token
   prompt needs a 58 GiB ceiling; 48 GiB fails at layer 0. Bounding the KDA recurrence
   (`EVAL_EVERY`) recovered only 0.5 GiB of the ~16 GiB in play, so the dominant consumer
   is still unidentified. `peak_bytes` now lands in the token log, which should name it on
   the next long run.
1. **Spec section 16 criterion 5** asks for top-1 parity over 16 tokens; verified for
   1 token at full depth (`k3ref` costs ~14 min/token, so 16 is ~3.5 h). The one
   criterion not met to the letter.
2. **CUDA confirmation.** `k3ref` validates against the shipped code's semantics and
   cannot catch a misreading shared by both. One rented NVIDIA run would close it.
3. **Open WebUI** untested end-to-end; needs its client timeout raised past the usual
   30–120 s default before it can work at 34 s/token.
4. **Coding agents are impractical** — a 1k–10k-token task is 9–94 hours.
5. ~~**More pinned memory** is the clearest remaining speedup: 32 GiB covers ~30% of the
   105.7 GiB dense working set. ~40 GiB should reach ~31 s/token.~~ **WRONG — measured
   2026-08-10.** More pinning is monotonically *slower* across a 5× sweep: 40 GiB gives
   69.0 s/token, 8 GiB gives 32.4. Fill-and-pin takes unified memory permanently from the
   OS page cache, which is what actually serves the 105.7 GiB working set, and the app
   cache holds only what it loaded first rather than what is being used. See
   `docs/kv-cache/sweep/RESULTS.md`. `tests/test_resident_cache.py` predicts the opposite
   because its model has no term for physical memory limits or page-cache displacement.
   **The remaining lever is reading fewer bytes, not caching them better** — per-token
   cost is 95.6 GiB ÷ achievable bandwidth, and the floor is ~32 s/token.
6. **AMD Strix Halo box** — its value is the 96 GB unified VRAM (potential ~3x), not the
   disk (~10%). Gated on freeing ~800 GB and a hard ROCm-PyTorch-on-Windows problem.
   First experiment: prove ~89 GiB can actually be held in the GPU carve-out with only
   31.65 GB CPU-visible.

## Things that cost time, so they do not have to again

* **A USB4 enclosure in a front Mac mini port silently runs 3.6x slower** (10 Gbps USB
  fallback). Check `ioreg -rc IONVMeController` shows it; if it appears under
  `IOUSBHostDevice` instead, move it to a rear port.
* **LRU caching is pathological here** — a cyclic layer scan evicts exactly what it needs
  next. A 24 GiB LRU cache measured a 0% hit rate. `--resident-gb` uses fill-and-pin.
* **mmap is 9x slower than `read()`** on cold data; page faults defeat readahead.
* **`request.is_disconnected()` never fires inside a StreamingResponse** — Starlette owns
  the receive channel. Cancellation is detected structurally instead.
* **Prefill is expensive**: every prompt token routes independently, so a 291-token
  prompt takes ~10 min before the first output token.
* **Fit segmented slopes, not one line.** The s/token curve bends hard around token 800.
  A single OLS fit over the whole run reported +16.1 s/tok per 1k context, which is true
  of no part of it — the first third is +53.4 and the last is -7.9. `kv_cache_report.py`
  reports thirds for this reason.
* **A test that compares execution paths against each other can pass while all of them
  are wrong together.** `test_synthetic_integration.py` checks six paths for agreement,
  so it stays green if a change moves all six. Confirm token IDs against `master`
  (`git stash`, run, `git stash pop`) before believing it.
