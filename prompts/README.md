# Kimi-K3 architecture-targeted prompts

Four prompts, each designed to exercise a specific K3 architectural component rather
than general capability. Sourced 2026-08-07.

| # | File | Targets | Est. output | Est. wall time @34 s/tok |
|---|---|---|---|---|
| 1 | `01_delta_state_tracking.md` | Kimi Delta Attention — state overwrite/decay vs static KV | ~1,500 tok | ~14 h |
| 2 | `02_deep_layer_precision.md` | Attention Residuals — late layers pulling low-level bit detail | ~2,000 tok | ~19 h |
| 3 | `03_order_agnostic_causal.md` | NoPE — causality from semantics, not token distance | ~1,200 tok | ~11 h |
| 4 | `04_polymath_multi_expert.md` | LatentMoE 896 experts / 16 active — simultaneous multi-domain | ~2,500 tok | ~24 h |

## Budget

A 48-hour continuous run at 34 s/token yields ~5,080 output tokens
(48 x 3600 / 34). The original estimates assume 30 s/token (5,760 tokens); this
backend measures **34 s/token** in the two-tier + pinned configuration, so budget
~12% fewer tokens than the per-prompt estimates suggest.

**Prefill is charged separately and is not free.** Every prompt token routes
independently through the MoE, so a long prompt can touch most of the 896 experts in
each of the 92 layers. Measured: a 5-token prompt loads 17,061 expert projections
against 4,416 for a single decode step. A ~500-token prompt approaches the per-layer
ceiling of 896 x 3 = 2,688 projections, i.e. ~1.3 TiB of expert reads before the
first token appears. Long prompts are expensive here in a way they are not on batched
backends.

## Running

```bash
export AIR_LLM_API_KEYS=sk-key
python -m air_llm.serve --model /path/to/Kimi-K3 --manifest manifest_local.jsonl.gz \
    --cache ~/k3_dense --mode runtime --resident-gb 8 --max-output-tokens 3000

python - <<'PY'
from openai import OpenAI
from pathlib import Path
c = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-key", timeout=None)
body = Path("prompts/04_polymath_multi_expert.md").read_text()
for ch in c.chat.completions.create(model="chat", max_tokens=2500, stream=True,
                                    messages=[{"role":"user","content":body}]):
    if ch.choices and ch.choices[0].delta.content:
        print(ch.choices[0].delta.content, end="", flush=True)
PY
```

Use `stream=true` and no client timeout. Disconnecting cancels the generation within
~1 s, so an abandoned run stops burning disk rather than continuing.

## Set-and-forget: `tools/run_long_job.py`

For a real multi-hour run, don't drive it from an interactive shell. Use the job runner,
which writes every token to disk as it arrives and can resume if anything dies:

```bash
# terminal 1 -- the server
export AIR_LLM_API_KEYS=sk-k3
python -m air_llm.serve --model /Volumes/USB4_2TBB/Kimi-K3 \
    --manifest manifest_local.jsonl.gz --cache ~/k3_dense \
    --mode runtime --resident-gb 8 --port 8080 --max-output-tokens 3000

# terminal 2 -- the job
python tools/run_long_job.py \
    --prompt-file prompts/04_polymath_multi_expert.md \
    --out runs/polymath --max-tokens 2500
```

Check on it from any terminal, without attaching to the process:

```bash
cat runs/polymath/progress.json     # %, s/token, ETA, finish timestamp
tail -f runs/polymath/output.txt    # the answer as it is written
tail -f runs/polymath/job.log       # milestones and errors
```

What it guarantees over a plain SDK loop:

* **Per-token durability.** Each token is appended, flushed, and `fsync`ed. A crash at
  hour 23 keeps 23 hours of output.
* **Resume.** `--resume` replays the prompt plus the partial answer and asks only for the
  remaining tokens. Costs one prefill instead of the whole run. (The server cannot resume
  a finished stream; the job works around that at the message level.)
* **Clean cancellation.** Ctrl-C or SIGTERM closes the stream, which cancels the
  generation server-side within ~1 s instead of leaving it burning disk.
* **Retries** with backoff if the server restarts mid-run.
* **Won't clobber** an existing `output.txt` unless you pass `--resume`.

Note on timing your own scripts: a wall-clock budget check placed inside the streaming
`for` loop cannot fire while blocked waiting for the first token. With a ~10 minute
prefill, a "stop after 2 minutes" check evaluates for the first time at minute 10. Use a
watchdog thread or a socket timeout if you need a hard bound.
