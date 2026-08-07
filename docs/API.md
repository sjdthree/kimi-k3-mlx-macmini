# OpenAI-compatible API

An OpenAI-shaped `/v1` facade over the Kimi-K3 MLX backend. Point any OpenAI SDK or
compatible client at it by setting two variables:

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=sk-your-key
```

No SDK fork, no shim. OpenAPI spec: [`openapi.json`](openapi.json).

---

## Read this first: the backend is disk-bound

**~34 s per generated token; ~70–90 s to first token.** This is not a tuning gap — a
decode step streams ~95.6 GiB of weights for a 2.8-trillion-parameter MoE. It has direct
consequences for clients:

| | implication |
|---|---|
| **Client timeouts** | Defaults of 30–120 s expire before the first token. Set ≥ 900 s. |
| **Streaming** | Strongly preferred. With `stream=false` a client sees nothing for minutes. |
| **Concurrency** | One generation at a time. Concurrent requests queue, then 429. Parallelism makes throughput *worse* — the disk is the contended resource. |
| **`n > 1`** | Multiplies wall-clock linearly. Capped by `--max-n` (default 1). |
| **Chat UIs** | Open WebUI and similar work, but need their request timeout raised. |
| **Coding agents** | A 1k–10k-token task is 9–94 hours. Not practical today. |

The per-token estimate is advertised in model metadata as
`air_llm.capabilities.approx_seconds_per_token` so clients can adapt rather than guess.

## Compatibility matrix

Model: `kimi-k3-2026-07-16` (aliases: `chat`, `reasoning`, `kimi-k3`)

| Feature | Supported | Notes |
|---|---|---|
| `POST /v1/chat/completions` | ✅ | The compatibility baseline |
| Streaming (SSE) | ✅ | `chat.completion.chunk`, terminates `data: [DONE]` |
| `stream_options.include_usage` | ✅ | Usage chunk before `[DONE]` |
| Client-disconnect cancellation | ✅ | Registers ≤ 1 s; generation stops at the next token boundary |
| `temperature`, `top_p`, `seed` | ✅ | `temperature=0` is exact greedy and reproducible |
| `max_tokens` / `max_completion_tokens` | ✅ | Both accepted; capped server-side |
| `stop` | ✅ | String or array; output trimmed at the match |
| `logprobs`, `top_logprobs` | ✅ | From the unmodified distribution, pre-temperature |
| `logit_bias` | ✅ | Token-id keys |
| `user` | ✅ | Recorded for attribution, never logged with content |
| `n` | ⚠️ | Accepted up to `--max-n` (default 1); each choice is a full generation |
| Multi-turn, `system`/`developer`, named messages | ✅ | Rendered into a plain template — see Deviations |
| **`tools` / function calling** | ❌ | **No constrained decoding.** Returns 400 `unsupported_capability` |
| **`response_format: json_schema`** | ❌ | Cannot enforce a schema; returns 400 rather than claiming conformance |
| **`response_format: json_object`** | ❌ | Same reason — valid JSON cannot be guaranteed |
| **Vision / image parts** | ❌ | Checkpoint ships MoonViT; not implemented |
| **`/v1/embeddings`** | ❌ | No embedding model deployed. Returns 404; never routed to a chat model |
| `presence_penalty`, `frequency_penalty` | ❌ | 400 `unsupported_parameter` — they would change sampling, so they are not silently ignored |
| `/v1/completions` (legacy) | ❌ | 404. Use chat completions |
| `/v1/responses` | ❌ | 404. Additive, not implemented in this release |

**Nothing above is faked.** Every ❌ is advertised in `GET /v1/models` *and* returns a
structured error if requested. A capability that cannot be enforced is reported as absent
rather than approximated by prompting — an unenforced JSON schema is worse than an honest
error, because the caller builds on a guarantee that does not hold.

## Examples

### curl

```bash
curl $OPENAI_BASE_URL/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"chat","messages":[{"role":"user","content":"The capital of France is"}],
       "max_tokens":2,"stream":true}' --no-buffer
```

### Python

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="sk-your-key",
                timeout=1800.0)          # required: ~34 s/token

for chunk in client.chat.completions.create(
        model="chat",
        messages=[{"role": "user", "content": "The capital of France is"}],
        max_tokens=4, stream=True,
        stream_options={"include_usage": True}):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### TypeScript

```ts
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:8000/v1",
  apiKey: process.env.OPENAI_API_KEY,
  timeout: 1_800_000,                     // ms
});

const stream = await client.chat.completions.create({
  model: "chat",
  messages: [{ role: "user", content: "The capital of France is" }],
  max_tokens: 4,
  stream: true,
});
for await (const chunk of stream) process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
```

## Model naming and versioning

* **Aliases** — `chat`, `reasoning`, `kimi-k3`. Stable names; the target may change.
  Use when you want "whatever is best for this job".
* **Physical IDs** — `kimi-k3-2026-07-16`. Immutable. The same ID always means the same
  weights; different weights get a different ID. Use for reproducibility.

Deprecation policy: an alias may be repointed with notice; a physical ID is never
repointed or reused. Both appear as selectable entries in `GET /v1/models`.

## Errors

Standard OpenAI envelope, standard status codes:

```json
{"error": {"message": "...", "type": "invalid_request_error",
           "param": "tools", "code": "unsupported_capability"}}
```

| Status | When | Example `code` |
|---|---|---|
| 400 | malformed, or a parameter we cannot honour | `unsupported_capability`, `unsupported_parameter`, `invalid_value` |
| 401 | missing/invalid key | `invalid_api_key` |
| 403 | quota exhausted | `insufficient_quota` |
| 404 | unknown model or endpoint | `model_not_found`, `endpoint_not_found` |
| 413 | input over the token/byte limit | `request_too_large` |
| 429 | busy, rate limit, or concurrency limit | `server_overloaded`, `rate_limit_exceeded`, `concurrency_limit_exceeded` |
| 500 | internal failure | `internal_error` |

Errors never contain stack traces, hostnames, filesystem paths, or scheduler internals —
asserted by the conformance suite. Every response carries `X-Request-ID`; a client-supplied
`X-Request-ID` is echoed and used in logs.

If a stream has already started, an error cannot change the status code, so it is emitted
as a final SSE frame carrying the same envelope, followed by `[DONE]`.

## Backend-native controls

Backend-specific operations live in a separately versioned namespace so compatibility
never blocks a performance feature:

| Endpoint | Purpose |
|---|---|
| `GET /native/v1/engine` | readiness, active/completed/cancelled, cache stats, uptime |
| `GET /native/v1/storage` | resident weight bytes, tensor count, hit rate, cache policy |

Per-request native options use an explicit opt-in that leaves the standard request valid
and portable:

```json
{"model": "chat", "messages": [...],
 "extra_body": {"air_llm": {"priority": "interactive"}}}
```

Standard responses carry no un-namespaced extra fields — asserted by the conformance
suite. Model metadata lives under an `air_llm` block on the model object.

## Operations

| Endpoint | |
|---|---|
| `GET /healthz` | liveness |
| `GET /readyz` | 200 when the engine can serve, 503 while loading; reports `warm` |
| `GET /metrics` | Prometheus |

Metrics: request/error/cancellation counts, queue time, **time-to-first-token**, decode
tokens/sec, prompt/completion tokens, active requests, resident weight bytes, cache hit
ratio, peak unified memory. Histogram buckets are sized for this backend — default
latency buckets top out near 10 s and would put every request in `+Inf`.

Labels are `endpoint`, `model`, `code`, and a **hashed** `key_id`. Raw API keys never
appear in logs or metrics, and prompt/completion content is never logged.

Configuration: `AIR_LLM_API_KEYS` (comma-separated) or `AIR_LLM_API_KEYS_FILE` for
per-key `rpm`, `concurrency`, `max_output_tokens`, and `daily_tokens`. Set
`AIR_LLM_ALLOW_ANONYMOUS=1` to explicitly disable auth for local use — the server refuses
to start with no keys and no explicit opt-out.

## Known deviations

1. **Chat template.** K3 ships an XTML chat format (`encoding_k3.py`) that this project
   does not implement. Messages are flattened into a plain `Role: content` transcript.
   Output is coherent but not chat-tuned; expect base-model behaviour.
2. **`finish_reason: "cancelled"`** is emitted on client disconnect. OpenAI has no such
   value; the alternative was misreporting a truncated generation as `"stop"`.
3. **Prefill cost scales with prompt length** more than on batched backends — every
   prompt token routes independently through the MoE.
4. **`system_fingerprint` is not emitted.** There is no meaningful backend fingerprint to
   report, and a fabricated one would be worse than its absence.

## Running

```bash
export AIR_LLM_API_KEYS=sk-your-key
python -m air_llm.serve \
  --model /path/to/Kimi-K3 --manifest manifest.jsonl.gz \
  --cache ~/k3_dense --mode runtime --resident-gb 32 \
  --port 8000 --max-output-tokens 256
```

The model loads once and stays resident, so the pinned weight cache is populated by the
first request and **warm for every one after it** — unlike the CLI, which re-populates per
invocation. Measured: 24 GiB resident held across requests with the hit rate climbing.

For HTTP-surface work without waiting on the model:

```bash
AIR_LLM_ALLOW_ANONYMOUS=1 python -m air_llm.serve --mock --port 8000
```

## Conformance

`tests/test_openai_conformance.py` runs the **official OpenAI Python SDK** against the
server (29 checks, ~10 s on the mock engine): SDK construction with only `base_url` and
`api_key`, model listing, non-streaming and streaming shapes, raw SSE framing and `[DONE]`,
`include_usage`, disconnect cancellation, the full error taxonomy, logprobs, no internal
leakage, no un-namespaced response fields, and the published OpenAPI document.
