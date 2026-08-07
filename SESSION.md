# Session state — 2026-08-07

Picking this up after a break? Read this first, then `PROGRESS.md` for the engineering
record and `VALIDATION_REPORT.md` for results.

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

1. **Spec section 16 criterion 5** asks for top-1 parity over 16 tokens; verified for
   1 token at full depth (`k3ref` costs ~14 min/token, so 16 is ~3.5 h). The one
   criterion not met to the letter.
2. **CUDA confirmation.** `k3ref` validates against the shipped code's semantics and
   cannot catch a misreading shared by both. One rented NVIDIA run would close it.
3. **Open WebUI** untested end-to-end; needs its client timeout raised past the usual
   30–120 s default before it can work at 34 s/token.
4. **Coding agents are impractical** — a 1k–10k-token task is 9–94 hours.
5. **More pinned memory** is the clearest remaining speedup: 32 GiB covers ~30% of the
   105.7 GiB dense working set. ~40 GiB should reach ~31 s/token.
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
