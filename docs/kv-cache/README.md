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

```bash
# 1. the before-run must be finished and its result kept
cat runs/polymath/progress.json          # expect state: done
git checkout master && git add -f runs/polymath/output.txt runs/polymath/job.log

# 2. checkpoint-dependent suites, which could not run while the volume was busy
conda activate airllm && git checkout mla-latent-cache
for t in test_attention_mlx test_layer_mlx test_model_mlx test_moe_mlx \
         test_full_parity test_state_roundtrip test_mxfp4_mlx test_mxfp4_kernel; do
    python tests/$t.py || echo "FAILED: $t"
done

# 3. restart the server on the branch, identical flags
export AIR_LLM_API_KEYS=sk-k3
nohup python -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
    --manifest manifest_local.jsonl.gz --cache ~/k3_dense \
    --mode runtime --resident-gb 32 --port 8080 --max-output-tokens 3000 \
    > /tmp/serve-latent.log 2>&1 &

# 4. the same job, to a new directory
python tools/run_long_job.py --prompt-file prompts/04_polymath_multi_expert.md \
    --out runs/polymath-latent --max-tokens 2500

# 5. harvest and diff
python tools/kv_cache_report.py runs/polymath-latent --label AFTER \
    --compare docs/kv-cache/baseline-master.json \
    --json docs/kv-cache/after-latent.json
```

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
