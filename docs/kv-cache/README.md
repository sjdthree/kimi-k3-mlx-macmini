# KV cache: before / after

Comparison record for the MLA latent-cache change (branch `mla-latent-cache`). The
"before" half is captured from the in-flight `runs/polymath` job on `master`. The "after"
half is blank until that job finishes and the same prompt is re-run on the branch.

**Read the prediction section before filling in the results.** It is written first on
purpose, so the comparison cannot be rationalised after the fact.

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
audit, ~339 tokens).
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

> Not yet run. Blocked on `runs/polymath` finishing (~20:15 on 2026-08-09) so the two
> runs do not contend for the checkpoint volume or for memory.

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
    --token-log runs/polymath-latent/tokens.jsonl \
    > /tmp/serve-latent.log 2>&1 &

# 4. the same prompt, same target, to a new directory (~2 days)
python tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
    --out runs/polymath-latent --max-tokens 2500

# 5. harvest, diff, and attribute
python tools/kv_cache_report.py runs/polymath-latent --label AFTER \
    --prompt-tokens 295 \
    --compare docs/kv-cache/baseline-master.json \
    --token-log runs/polymath-latent/tokens.jsonl \
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

### Results

_(paste the output of step 5 here)_

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
     <(fold -w 100 runs/polymath-latent/output.txt) | head -40
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
