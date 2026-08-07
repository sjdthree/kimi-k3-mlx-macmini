"""Prometheus metrics.

Label cardinality is kept deliberately low: `model`, `endpoint`, `code`, and a hashed
`key_id`. No prompt or completion content is ever recorded, and raw API keys never
appear — tenant attribution goes through the same non-reversible key id used in logs.

Histogram buckets are sized for THIS backend. Default latency buckets top out around
10 s, which would put every single request in +Inf here: time-to-first-token is ~70 s and
a 16-token response is ~9 minutes.
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

NS = "air_llm"

requests_total = Counter(
    f"{NS}_requests_total", "Requests received",
    ["endpoint", "model", "code", "key_id"])

errors_total = Counter(
    f"{NS}_errors_total", "Requests that returned an error",
    ["endpoint", "type", "code"])

cancellations_total = Counter(
    f"{NS}_cancellations_total", "Generations cancelled by client disconnect", ["model"])

# seconds -- sized for a disk-bound backend
_SLOW = (1, 5, 15, 30, 60, 90, 120, 300, 600, 1200, 2400, 4800, float("inf"))

queue_seconds = Histogram(
    f"{NS}_queue_seconds", "Time spent waiting for the generation slot",
    ["model"], buckets=(0.01, 0.1, 1, 5, 15, 60, 300, 1200, float("inf")))

ttft_seconds = Histogram(
    f"{NS}_time_to_first_token_seconds", "Time to first generated token",
    ["model"], buckets=_SLOW)

request_seconds = Histogram(
    f"{NS}_request_seconds", "Total request duration", ["endpoint", "model"],
    buckets=_SLOW)

decode_tokens_per_second = Histogram(
    f"{NS}_decode_tokens_per_second", "Sustained decode rate",
    ["model"], buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.5, 1, 10, float("inf")))

prompt_tokens_total = Counter(f"{NS}_prompt_tokens_total", "Prompt tokens", ["model"])
completion_tokens_total = Counter(f"{NS}_completion_tokens_total", "Generated tokens",
                                  ["model"])

active_requests = Gauge(f"{NS}_active_requests", "In-flight generations")
queued_requests = Gauge(f"{NS}_queued_requests", "Requests waiting for the slot")

bytes_read_total = Counter(f"{NS}_weight_bytes_read_total",
                           "Weight bytes read from storage", ["model"])
resident_bytes = Gauge(f"{NS}_resident_weight_bytes",
                       "Weights pinned in unified memory")
cache_hit_ratio = Gauge(f"{NS}_weight_cache_hit_ratio",
                        "Resident weight cache hit ratio")
peak_memory_bytes = Gauge(f"{NS}_peak_unified_memory_bytes",
                          "Peak unified memory observed")
model_ready = Gauge(f"{NS}_model_ready", "1 when the engine can serve")
