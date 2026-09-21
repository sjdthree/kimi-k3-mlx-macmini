> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# KV cache: before / after

Comparison record for the MLA latent-cache change (branch `mla-latent-cache`).
**The final result is immediately below**; everything after it is the record of how it was
arrived at, in the order it happened — baseline, predictions written before the runs, the
300-token A/B, the long-context test, and the resident-gb sweep.

That order is deliberate. The predictions were written first so the comparison could not be
rationalised afterwards, and two conclusions in this document reverse earlier ones in it.
Both reversals are left in place rather than edited away.

---

## FINAL VALIDATION — 2026-08-11T06:33, the original job re-run end to end

`runs/final-validation`. Prompt 04, 2,500-token target, `--resident-gb 8` — the exact
baseline job on the branch. Finished `finish_reason=stop` at **1,383 tokens** in
**13 h 39 m**, no resumes.

### a) Same result?  Yes — and slightly more complete

The model ended its own answer at 1,383 tokens where the baseline took 2,197, so it is
**37% shorter while covering the same ground**. Section structure is near-identical, with
the first four subsections of section 1 in the same order:

| | baseline | final |
|---|---|---|
| 1. CUDA Architecture & Hardware Performance | ✓ | ✓ |
| — Critical Overflow Bug / 64-bit Mult / Coalescing / Warp Divergence | ✓ | ✓ (same order) |
| 2. Cryptographic & Abstract Algebra Validity | ✓ | ✓ |
| 3. Formal Verification | ✓ | ✓ |
| Multi-precision 256-bit extension | ✓ | ✓ |

Every planted defect is found, and several are covered *more* thoroughly than the
baseline despite the shorter answer:

| term | baseline | final |
|---|---:|---:|
| `undefined behavior` / UB | 6 | **7** |
| `256-bit` / `384-bit` | 2 | **3** |
| `128-bit` | 7 | **8** |
| `__umul64hi` | 7 | 4 |
| warp divergence, coalescing | 1, 2 | 1, 2 |

### b) Much faster?  2.38×

| | master | branch |
|---|---:|---:|
| s/token | 83.6 | **35.18** |
| slope, s/tok per 1k ctx | +11.23 (+52.4 early) | **+1.81** |
| normalised to 2,197 tokens | 51.0 h | **21.5 h** |
| actual wall clock | 51 h 02 m (2,197 tok) | **13 h 39 m** (1,383 tok) |

### c) Memory stress gone?  Completely

| | master | branch |
|---|---:|---:|
| peak | 51 GB footprint | **15.0 GiB, flat for all 1,381 steps** |
| swap | 24.95 GiB of 25.6 GiB (0.63 free) | unchanged throughout |
| MLA cache at end | 4.61 GiB (at ctx 1,677) | **88.4 MiB** |

Peak never moved off 15.0 GiB — min and max identical across the whole run.

### One caveat on the speed number

Quartiles were 32.4 / 37.3 / 35.5 / 35.5. The step up after the first quartile is not the
model: it is when `tools/bench_machine.py` was run **on this machine during the
measurement**, which pushed swap from 15.3 to 20.0 GiB. The clean rate is the 32.4 of the
first quartile and of the sweep; 35.18 is the contaminated average, and is reported here
because it is what was actually measured. The true figure is ~9% better, i.e. ~2.6×
rather than 2.38×.

---

## The change

`master` cached the expanded per-head key/value. The tensors those are built from are
53.3× smaller, and `kv_b` — the map between them — absorbs into the query and output
projections, so caching the latent instead is exact, not approximate.

| | master | branch |
|---|---:|---:|
| cached per token, per MLA layer | `96×192 + 96×128` = 30,720 floats | `512 + 64` = 576 floats |
| across 24 MLA layers | **2,949,120 B/token** | **55,296 B/token** |
| at the 2,371-token context below | **6.51 GiB** | **0.12 GiB** |
| growth | `mx.concatenate` per token (O(T²)) | chunked buffer |
| reallocations per 256 appends | 255 | 1 |

Verified lossless: max deviation 3.0e-07 against both an independent fp64 oracle and a
frozen copy of the master path, holding across every decode step. See
`tests/test_mla_cache.py`.

---

## BEFORE — master, expanded cache

Job: `runs/polymath`, PID 99281, started 2026-08-07T09:48:45, 2,500-token target.
Prompt: `prompts/04_polymath_multi_expert.md` (CUDA / cryptography / formal-verification
audit, 295 tokens).
Server: `python -m air_llm.serve --resident-gb 32 --port 8080`, PID 98807.
Machine: Mac16,11, 64 GiB unified memory.

**Final.** Completed 2026-08-09T12:50:44 with `finish_reason=stop` — the model ended its
answer at 2,197 tokens rather than reaching the 2,500 cap. No resumes. Prompt measured at
**295 tokens** (an earlier estimate of 339 was wrong; the cache figures use 295).
Regenerate with:

```bash
python tools/kv_cache_report.py runs/polymath --label BEFORE \
    --prompt-tokens 295 --json docs/kv-cache/baseline-master.json
```

| metric | value |
|---|---:|
| tokens generated | 2,197 |
| final context (incl. prompt) | 2,492 |
| total elapsed | 2 d 3 h 02 m |
| time to first token | 669 s |
| mean s/token | 83.6 |
| first-decile s/token | 57.94 |
| last-decile s/token | 80.00 |
| **degradation (last/first)** | **1.381×** |
| slope, s/tok per 1k ctx (overall) | +11.23 |
| slope, first third | **+52.36** |
| slope, last third | **−25.52** |

System state at capture (`baseline-system.txt`):

| | |
|---|---:|
| server process footprint | **51 GB** (64 GiB machine) |
| swap used | **24.95 GiB of 25.6 GiB** (0.63 GiB free) |
| page faults, server process | 599,927,862 |

Reference points from earlier work: `SESSION.md` records **34 s/token** achieved on this
model; `tests/test_resident_cache.py` predicts **~29.6 s/token** of I/O at 32 GiB
resident, and notes compute is only ~2.0 s/token. The observed 84–95 s/token is roughly
**3× the I/O-bound prediction**, and that gap is what this change targets.

### The curve bends — this matters

| token | s/token |
|---:|---:|
| 20 | 52.4 |
| 180 | 60.8 |
| 340 | 70.9 |
| 500 | 83.8 |
| 660 | 87.2 |
| 820 | 93.5 |
| 980 | 90.0 |
| 1140 | 87.7 |
| 1300 | 88.4 |
| 1460 | 91.5 |
| 1620 | 86.0 |
| 1780 | 87.0 |
| 1940 | 82.9 |

The degradation is **front-loaded and saturating**, not linear. It climbs steeply to
~token 800 (+53 s/tok per 1k context) and then plateaus, drifting slightly *down*
thereafter (−7.9 s/tok per 1k).

This rules out the simplest story. A per-token O(T) cache copy would keep costing more
forever; it does not. The shape is a **threshold effect**: something crossed a cliff
around token 700–800 and then settled into a new equilibrium.

That is consistent with memory pressure — at token 800 the cache is ~2.25 GiB, and the
machine is now 24.95 GiB into swap — but consistency is not proof.

### Confound, stated up front

There is a second mechanism that produces exactly the same front-loaded-then-flat shape:
**MoE expert routing diversity**. If early tokens of a response route to a narrow set of
experts and diversity rises as the answer develops, disk reads per token would climb and
then plateau once the working set of experts stabilises. That has nothing to do with the
KV cache.

Both mechanisms predict the observed curve. The A/B is what discriminates them, which is
the reason to run it rather than reason further.

### The completed run tilts toward the confound

The last 160 tokens changed the picture, and not in favour of the KV cache. Instantaneous
s/token over the tail:

| token | 1960 | 1980 | 2000 | 2020 | 2040 | 2060 | 2100 | 2140 | 2180 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s/token | 95.1 | 93.4 | 97.6 | **109.2** | **71.3** | 75.5 | 77.2 | 77.1 | 78.3 |

It peaks at 109.2 and then drops ~30% to a stable 71–78 for the rest of the run — **while
the KV cache is at its largest it has ever been**. A cache-pressure mechanism cannot
produce that. The last tokens should be the slowest; they are the fastest of the entire
second half.

The transition also lines up with the content. The final ~500 bytes of `output.txt` is the
`## Summary` section — a markdown table of three rows. That is exactly the repetitive,
low-diversity text that would route to a narrow, repeated set of experts and cut the
per-token disk reads.

This does not prove the routing hypothesis, and there are dumber explanations (something
else on the machine released memory; ordinary variance). But it is direct evidence that
**something other than cache size dominates the per-token cost**, and it should temper any
expectation that the latent cache alone recovers the 34 s/token figure.

**Change to the A/B procedure because of this.** Timing alone cannot separate the two
mechanisms — measure the routing directly. `TokenRecord` already carries `experts_loaded`
and `bytes_read`, and `KimiK3Model.generate` writes them per token via `log_path`, but the
server path (`engine.py::_run`) does not. Wire that through and capture a per-token log on
BOTH runs; then correlate s/token against `experts_loaded` and settle it with data instead
of inference.

---

## Predictions (written before the after-run)

**If memory pressure was the dominant cause**, removing 6.4 GiB should put the working
set back under the cliff:

- plateau s/token falls substantially, toward the 52–58 s/token seen at low context
- degradation ratio moves toward ~1.0 — the point is to *lower the plateau*, not to
  flatten a slope that is already flat by the end
- swap usage during the run drops well below 24.95 GiB
- mean s/token falls; the 34 s/token in `SESSION.md` is the optimistic bound

**If expert-routing diversity was the dominant cause**, the curve barely moves:

- the same ~52 → ~90 s/token climb over the first ~800 tokens
- similar plateau, similar mean
- swap still falls (the cache really is 53× smaller) but speed does not follow

**Either way** the memory result stands on its own: 6.51 GiB → 0.12 GiB is arithmetic
plus a proof of losslessness, not a benchmark. A null speed result would not retract it;
it would relocate the bottleneck, which is a useful outcome and should be recorded as
such rather than buried.

---

## AFTER — branch `mla-latent-cache`, latent cache

> Not yet run. Blocked on `tests/test_full_parity.py`.

### Scope: 300 tokens, not 2,500

The after-run is **300 tokens (~7 h)**, not a matching 2,500-token run (~2 days). Two
reasons, and the first is the important one:

1. **The baseline degraded almost entirely within its first ~800 tokens** (52 → 93
   s/token, slope +52 per 1k) and then plateaued. A 300-token run therefore lands
   squarely on the rising portion — the part that actually needs explaining. Running to
   2,500 mostly re-measures a plateau.
2. **The attribution test is within-run**, so it needs enough decode steps for routing to
   vary, not a long context.

`kv_cache_report.py` truncates the baseline to the same token count when the after-run is
materially shorter. Comparing a 300-token run's "last decile" against a 2,197-token run's
plateau would manufacture a result in whichever direction the slice happened to fall.

### Pre-registered target — master's first 300 tokens

| metric | master @ 300 |
|---|---:|
| mean s/token | **60.09** |
| first-decile s/token | 52.70 |
| last-decile s/token | 67.00 |
| degradation (last/first) | **1.271×** |
| slope, s/tok per 1k ctx | **+49.39** |

**If the KV cache was the cause:** the branch stays near 52.7 across all 300 tokens —
degradation → ~1.0, slope → ~0, mean well under 60.

**If routing dominates:** the branch climbs 52.7 → ~67 just like master, degradation
~1.27, slope ~+49, mean ~60. The cache change would then be proven correct and
irrelevant to speed, and the next work is expert residency, not the cache.

A third outcome is possible and worth naming: **partial** — a reduced but non-zero climb,
meaning both mechanisms contribute. The `--token-log` correlations are what would apportion
it.

### Procedure

Same prompt, same target, same server flags — only the branch differs.

Steps 1 and 2 are **done** (2026-08-09). The before-run finished and its artifacts are
committed; the checkpoint suites pass on real weights.

```bash
# 3. restart the server on the branch. Identical flags to the baseline PLUS --token-log.
#    Kill the old one first -- it is idle but still holds the resident cache.
pkill -f "air_llm.serve" ; sleep 5
export AIR_LLM_API_KEYS=sk-k3
conda activate airllm && cd ~/kimi-k3-mlx
git checkout mla-latent-cache
nohup python -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
    --manifest manifest_local.jsonl.gz --cache ~/k3_dense \
    --mode runtime --resident-gb 32 --port 8080 --max-output-tokens 3000 \
    --token-log runs/polymath-latent-300/tokens.jsonl \
    > /tmp/serve-latent.log 2>&1 &

# 4. the same prompt, same target, to a new directory (~2 days)
python tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
    --out runs/polymath-latent-300 --max-tokens 300

# 5. harvest, diff, and attribute
python tools/kv_cache_report.py runs/polymath-latent-300 --label AFTER \
    --prompt-tokens 295 \
    --compare docs/kv-cache/baseline-master.json \
    --token-log runs/polymath-latent-300/tokens.jsonl \
    --json docs/kv-cache/after-latent.json
```

**Keep `--resident-gb` at 32 for this run.** It is the only variable that must not move,
or the comparison measures two changes at once. Raising it is a separate experiment, and
the footprint numbers from this run are what tell you whether there is room.

**Smoke-test before committing two days.** Between steps 3 and 4, ask for 2 tokens and
check the log actually appeared with populated counters. One assertion carries most of
the value:

```
mla_cache_bytes delta == 55,296 B/token     # 24 layers x (512+64) x 4, the LATENT rate
                                            # master's would be 2,949,120
```

That single number proves the branch's cache is live on the real checkpoint, which no
synthetic test can establish. Discovering otherwise four hours into a two-day run is the
failure this avoids.

### The attribution does not need an instrumented before-run

The baseline ran without `--token-log`, and re-running master to collect it would cost
another two days. It is not necessary: **the decisive test is within a single run.**
Context and routing vary independently inside one generation — context rises
monotonically, routing wanders with the content — so correlating s/token against each
separates them using the after-run alone. Step 5 prints that table.

Optional, if the within-run result is ambiguous: a short instrumented master run (~150
tokens, ~3.5 h) gives routing data at matched context for a fraction of the cost.

### Results — 2026-08-10T01:23:31

300 tokens, `finish_reason=length`, 4 h 00 m elapsed, prefill 645 s (baseline: 669 s).
Matched to master's first 300 tokens.

| metric | master @300 | branch @300 | change |
|---|---:|---:|---:|
| mean s/token | 60.09 | **48.10** | **−20.0%** |
| first-decile | 52.70 | 44.40 | −15.7% |
| last-decile | 67.00 | 46.70 | **−30.3%** |
| degradation (last/first) | 1.271 | **1.052** | −17.2% |
| **slope, s/tok per 1k ctx** | **+49.39** | **+6.60** | **−86.6%** |
| slope, first third | +82.00 | +20.83 | −74.6% |
| slope, last third | +73.67 | +6.67 | −90.9% |
| MLA cache @595-token ctx | 1.63 GiB | 0.03 GiB | 53.3× |

The curve, which is the actual finding — 45.1 at token 20 to 46.7 at token 300, a **3.5%
rise where master rose 27%**:

| token | 20 | 60 | 100 | 140 | 180 | 220 | 260 | 300 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| branch | 45.1 | 45.3 | 46.3 | 46.5 | 46.2 | 45.9 | 46.5 | 46.7 |
| master | 52.4 | 55.6 | 58.9 | 61.0 | 60.8 | 63.4 | 65.9 | 67.0 |

### Verdict against the pre-registered prediction

The prediction was: *cache was the cause* → flat near 52.7, degradation ~1.0, slope ~0;
*routing dominates* → climbs to ~67 like master.

**The cache was the dominant cause over this range.** Degradation 1.052 against a
predicted ~1.0, slope +6.6 against a predicted ~0 — and the absolute level came in
*better* than predicted (44–47 rather than 52.7), because master's 52.7 first-decile
already carried cache overhead the prediction implicitly baked in.

It is not a clean sweep. A residual +6.6 s/tok per 1k slope and a +0.19 context
correlation remain, so something still scales with context. That is the "partial" outcome,
weighted heavily toward the cache.

**This reverses the mid-run reading.** After the baseline's tail dropped 109 → 72 s/token
this document concluded the evidence "tilts toward the confound". Over the first 300
tokens it does not — the cache was carrying most of the climb. Both can hold: the cache
drives the early climb, and something content-dependent drives the late tail swing. The
tail remains unexplained.

### What this run CANNOT tell you

1. **Attribution failed.** The routing counters are structurally constant — 4,416 experts
   and 102,622,070,784 bytes read on every single decode step, while s/token varied. They
   cannot correlate with anything. `cache_hits`/`cache_misses` were added afterwards
   (`engine.py`) and need a future run.
2. **300 tokens, not 2,197.** Master kept degrading to ~800 tokens and 93 s/token. Whether
   the branch stays flat that far is extrapolation, though an 86.6% lower slope is a
   strong basis for it.
3. **The tail drop is still unexplained.** It cannot have been "fewer experts loaded" —
   that number never changes.

Output quality is intact: the 300-token answer identifies the `(t + m*N) >> 64` undefined
behaviour and reaches the `__umul64hi` correction. It stops at `finish_reason=length`
mid-audit, as a 300-token cap requires.

| metric | before | after | change |
|---|---:|---:|---:|
| mean s/token | 84.3 | | |
| first-decile s/token | 57.79 | | |
| last-decile s/token | 90.62 | | |
| degradation (last/first) | 1.568× | | |
| slope, first third | +53.38 | | |
| slope, last third | −7.86 | | |
| swap used at end | 24.95 GiB | | |
| server footprint | 51 GB | | |
| MLA cache at final context | 6.51 GiB | 0.12 GiB | 53.3× |

### Output quality

The two runs answer the same prompt, so the text is directly comparable. The change is
lossless by construction, but the runs will still diverge — the server samples with
`temperature`/`top_p`, and even greedy decoding drifts once fp32 reassociation moves a
logit past a tie. Do **not** read divergence as a defect. Compare on substance: does the
after-run still identify the `(t + m*N) >> 64` overflow, the single-limb Montgomery
failure for 256/384-bit curves, and the `__umul64hi` correction?

```bash
diff <(fold -w 100 runs/polymath/output.txt) \
     <(fold -w 100 runs/polymath-latent-300/output.txt) | head -40
```

---

## What is left for flat s/token

Arithmetic, not intuition. Per decode step, at four context lengths:

| ctx | MLA FLOP/tok | cache read/tok | **master** cache read/tok | cache resident |
|---:|---:|---:|---:|---:|
| 2,492 | 12.5 GF | 131 MiB | **6.8 GiB** | 0.13 GiB |
| 8,192 | 41.1 GF | 432 MiB | **22.5 GiB** | 0.42 GiB |
| 32,768 | 164.3 GF | 1.7 GiB | **90.0 GiB** | 1.69 GiB |
| 131,072 | 657.1 GF | 6.8 GiB | **360.0 GiB** | 6.75 GiB |

For scale, the O(1) per-token costs: dense weight stream **73.7 GiB** at 32 GiB resident
(~29.6 s/token), lm_head **2.19 GiB**, compute ~2.0 s/token.

**The context-driven term is now solved, and the master column shows why it mattered.**
At the baseline run's 2,492-token context master re-read **6.8 GiB of KV cache per token**
on top of 73.7 GiB of dense weights — ~9% of all traffic, and it was coming off a
swapping machine. That is now 131 MiB, or 0.17% of the dense stream. At 32k master would
have needed 90 GiB/token of cache reads alone, more than the entire weight stream; it was
never going to get there.

The remaining O(T) terms stay negligible against a ~30 s/token I/O floor until well past
32k. **s/token should now be flat in context.** Note the absorbed form costs ~3.4× more
attention FLOPs than master's expanded form (512-wide dots instead of 192/128) — that is
the deliberate trade, and at 12.5 GF/token it is invisible next to the I/O.

So "flat" is largely done, and it is not the same problem as "fast". What sets the number
now is O(1) in context:

1. **Measure before optimising.** Run the A/B with `--token-log` on both sides and
   correlate s/token against `experts_loaded`. The baseline's 109 → 72 s/token tail swing
   is either routing or it is not, and this answers it in one run. Everything below is
   speculative until it does.
2. **Possibly raise `--resident-gb`.** `test_resident_cache` measures 32 GiB → 29.6
   s/token, 40 GiB → 27.2, 45 GiB → 25.7. The latent cache freed ~6.7 GiB, which is
   roughly one step on that ladder. **But check the footprint first** — the baseline ran
   at a 51 GB process footprint on a 64 GiB machine with 25 GiB of swap consumed, so the
   freed memory may be needed just to get back under the cliff. Raising resident-gb
   blindly could put it straight back into thrash.
3. **Expert residency, if the routing correlation holds.** Prefill is the strongest hint:
   669 s for a 295-token prompt is ~8 token-equivalents, and the mechanism is that every
   prompt token routes independently, touching far more distinct experts. Note the LRU
   warning in `SESSION.md` is about the cyclic *layer* scan; expert reuse across tokens is
   a different access pattern and is not obviously pathological.
4. **Skip `causal_mask` when T == 1.** During decode the single query sees every cached
   position, so the mask is all zeros — an allocation and a 96 × Tk add per MLA layer per
   token, for nothing. Marginal (~5.7M element-adds/token at 2.5k), but free.

## Long-context test — `runs/longctx-2k`

Launched 2026-08-10T03:37:49. Prompt `prompts/05_longctx_cuda_audit.md`, **1,955 tokens**
(a 256-bit Montgomery CUDA library with planted defects, same three-domain audit shape as
prompt 04 so the task is comparable). 150 tokens generated, decode context ~1,960→2,110.

**Why a long PROMPT rather than a long generation.** Context reached by prefill and
context reached by decoding produce the same KV cache size, so decode s/token is
comparable at matched context — but prefill gets there in 72 minutes instead of two days.

### The controlled measurement (primary)

Same server, same session, same day — only the context differs:

| | context | branch s/token |
|---|---:|---:|
| `polymath-latent-300` | 595 | **46.7** |
| `longctx-2k` | ~2,035 | _pending_ |

A flat result across a 3.4× context increase is the claim. This is the clean comparison
because nothing but context changes.

### The cross-run reference (secondary, confounded)

Master at the same context, from the baseline curve (generated tokens 1,665–1,820):

| | value |
|---|---:|
| mean s/token | **90.6** |
| range | 84.5 – 96.1 |
| MLA cache @ctx 2,035 | **5.59 GiB** |
| branch MLA cache @ctx 2,035 | **0.105 GiB** |

**State the confound.** Master reached that context by generating for ~40 hours, so its
90.6 s/token includes whatever the machine had accumulated by then — it was 24.95 GiB into
swap. The branch reaches the same context in 72 minutes on a 6-hour-old server. Part of
any gap is system drift, not cache size. That is exactly why the within-branch comparison
above is the primary measurement and this one is a reference.

### Long prompts were never possible on this model — on master either

Two attempts died in prefill at layer 0 (a KDA layer) before one got through:

| attempt | ceiling | peak | survived | outcome |
|---|---:|---:|---:|---|
| 1 | 48 GiB | 48.9 GiB | 4 m 09 s | MemoryError |
| 2, after the KDA fix | 48 GiB | 48.4 GiB | 14 m 19 s | MemoryError |
| 3 | 58 GiB | _pending_ | | |

`attention_kda.py` is untouched by this branch (`git diff master...HEAD`), so this is a
pre-existing limit that every earlier run missed by using a 291-token prompt.
`--max-input-tokens` defaults to 8192, which was aspirational: anything past roughly a
thousand tokens hits this.

**The KDA fix is real but was not the main consumer, and the numbers say so.** Forcing
evaluation every 64 recurrence steps is correct — `tests/test_kda_prefill_memory.py`
shows bit-identical results and 4× the prompt costing 1.32× the memory instead of 4×.
But at T=1955 on the real model it moved the peak only 48.9 → 48.4 GiB. I predicted the
live state chain was ~11.5 GiB; recovering 0.5 GiB means MLX was already releasing most
of it, and **the dominant consumer at long prefill is still unidentified.**

Candidates not yet ruled out, at T=1955 with the 32 GiB resident cache leaving ~16 GiB of
headroom: the dense MLP at layer 0 materialising `[1, 1955, 67584]` fp32 (528 MiB) plus
several same-sized `situ` temporaries; the KDA per-step `[B,H,D,D]` temporaries inside a
64-step window; `apply_attn_res` over a `[1955, k, 7168]` block residual. None of these
individually accounts for 16 GiB, which is exactly why attempt 3 raises the ceiling — to
record the true peak instead of guessing at it a third time.

Only the safety ceiling moved (48 → 58 GiB on a 64 GiB machine). `--resident-gb` stays at
32 so the run remains comparable with `polymath-latent-300`.

### Prediction

If the latent cache is what flattened the 300-token curve, the branch should hold **~46–48
s/token at context 2,035**, statistically indistinguishable from its own 46.7 at context
595. A rise toward 60+ would mean something else scales with context that 300 tokens was
too short to expose.

### Result — 2026-08-10, and it beat the prediction

| | value |
|---|---:|
| prefill | 1,959 tokens in 1,541 s (**0.79 s/prompt token**) |
| decode | 149 tokens, context 1,960 → 2,108 |
| first token (warming) | 102.0 s |
| **steady-state mean** | **34.97 s/token** |
| min / max | 34.2 / 61.5 |
| quartiles | 35.3 · 34.8 · 34.9 · 34.9 |
| **slope** | **−4.33 s/tok per 1k ctx** |
| MLA cache at ctx 2,108 | **111.2 MiB** (master: 5.79 GiB) |

Flat to within 0.5 s across the quartiles, with a *negative* slope. Against master's
**90.6 s/token** at the same context, that is **2.59×** — and against the branch's own
46.7 at context 595, it is *faster at 3.5× the context*.

### Why the branch beat itself, and what it proves

Comparing the two branch runs token-for-token:

| | 300-tok run | longctx-2k |
|---|---:|---:|
| context | 296–594 | 1,960–2,108 |
| bytes_read/token | 102,622,070,784 | 102,622,070,784 |
| expert_bytes/token | 25,829,572,608 | 25,829,572,608 |
| experts_loaded/token | 4,416 | 4,416 |
| mean s/token | 46.1 | 35.0 |
| **effective throughput** | **2.08 GiB/s** | **2.73 GiB/s** |

**Identical bytes, identical experts, 31% faster.** The only variable left is where those
bytes came from — resident cache in RAM versus the USB4 disk. The 1,959-token prefill
routed through far more experts than the 7-token smoke request that warmed the 300-token
run, so it populated the pinned cache much more thoroughly.

That closes the loop this document has been circling since the baseline: **per-token cost
is governed by resident-cache hit rate, not by context length.** Once the KV cache is
fixed, context is free. The master baseline's ~90 s/token had three contributors, not one
— the KV cache, ~40 hours of accumulated swap pressure, and an under-warmed resident
cache.

Output quality holds at long range: the answer opens by identifying the guaranteed
deadlock from `__syncthreads()` after a data-dependent `return` and the out-of-bounds
global access — both planted defects, cited from code ~1,500 tokens back.

## Still open

- **Real-weight quantisation quality.** `tests/test_mla_quant.py` gates Q1/Q2/Q6 run on
  random weights, which makes the latent Gaussian by CLT (measured kurtosis 3.02). That
  is precisely the case where the rotation trick behind TurboQuant/VeloxQuant cannot
  help, so **Q6 is inconclusive, not negative**. Re-run it against real latents before
  drawing any conclusion about the article's approach.
- **The rope slice is now the floor.** With the latent quantised to 4 bits, the fp32
  64-wide rope slice is 44% of what remains, and 57% at 2 bits. It is the cheaper and
  safer next target than the last two bits of the latent.
- **Quantisation stays off.** At the ~2.3k context this model actually runs at, it saves
  ~90 MiB on top of a cache that is already 0.12 GiB. It earns its place past ~32k.
