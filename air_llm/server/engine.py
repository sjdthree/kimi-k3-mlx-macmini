"""Generation engine: one model, one generation at a time, cancellable.

Why single-flight
-----------------
A decode step streams ~95.6 GiB of weights. Two concurrent generations do not overlap
usefully -- they contend for the same disk and each runs at roughly half speed, so
concurrency is strictly worse than a queue. The engine therefore serialises generation
and rejects (429) or queues rather than pretending to parallelise.

Why a long-lived process is a real win
--------------------------------------
`--resident-gb` pins dense weights in unified memory. In the CLI each invocation
re-populates that cache from scratch. Here the model is built once at startup and lives
for the process lifetime, so only the *first* request pays for it and every subsequent
request starts warm. That is the main performance argument for the server beyond
compatibility.

Cancellation
------------
A dropped client must stop the generation promptly -- an abandoned request costs minutes
of disk and GPU.

The signal is NOT `request.is_disconnected()`. Starlette's StreamingResponse runs its own
`listen_for_disconnect` task which consumes the receive channel, so a generator that also
polls `is_disconnected()` never observes the disconnect -- Starlette already took the
message. What it does instead is cancel the task group, which throws `CancelledError`
into the streaming generator, which in turn closes the `async for` driving this engine
and raises `GeneratorExit` at our `yield`.

So cancellation is detected structurally: `_run` tracks whether it reached its natural
end, and anything else (GeneratorExit, CancelledError, client hang-up, upstream error)
counts as a cancellation and stops the token loop. `is_disconnected()` is still polled
because it does work for the non-streaming path.

Worst-case latency is one token (~34 s) -- the loop can only be interrupted between
tokens without threading cancellation through 93 layer calls.
"""
from __future__ import annotations

import asyncio
import codecs
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger("air_llm.server")


@dataclass
class GenParams:
    prompt_tokens: list[int]
    max_tokens: int = 16
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    logprobs: bool = False
    top_logprobs: int = 0
    logit_bias: dict[int, float] | None = None


@dataclass
class Step:
    """One generated token."""
    token_id: int
    text: str
    index: int
    logprob: float | None = None
    top: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class Finished:
    reason: str                      # "stop" | "length" | "cancelled"
    prompt_tokens: int
    completion_tokens: int


class Cancelled(Exception):
    pass


class Engine:
    def __init__(self, cfg, src, tokenizer, *, max_memory_gb: float = 48.0,
                 queue_size: int = 2, max_output_tokens: int = 4096,
                 max_input_tokens: int = 8192, token_log=None):
        from ..models.kimi_k3.model import KimiK3Model
        self.cfg = cfg
        self.src = src
        self.tokenizer = tokenizer
        self.model = KimiK3Model(cfg, src, max_memory_gb=max_memory_gb)
        self.max_output_tokens = max_output_tokens
        self.max_input_tokens = max_input_tokens
        # Per-token diagnostics. KimiK3Model.generate already writes these via its
        # `log_path`, but the CLI path is not the one long runs go through -- the server
        # is. Without this, a slow run leaves only wall-clock timings, which cannot
        # separate "the KV cache grew" from "routing hit more experts". See
        # docs/kv-cache/README.md.
        self.token_log = Path(token_log) if token_log else None
        self._prev_cache = (0, 0)     # (hits, misses), for per-token deltas

        self._worker_lock = threading.Lock()
        self._lock = asyncio.Semaphore(1)
        self._queue_slots = asyncio.Semaphore(max(1, queue_size))
        self._ready = False
        self._warm = False
        self.active = 0
        self.completed = 0
        self.cancelled = 0
        self.started_at = time.time()

    # ------------------------------------------------------------------ lifecycle
    async def warmup(self) -> None:
        """Verify the weights are reachable without running a full forward pass."""
        def _probe():
            self.model.embed([0])
            _ = self.model._g("language_model.model.norm.weight")
        await asyncio.to_thread(_probe)
        self._ready = True
        log.info("engine ready model=%s layers=%d", "kimi-k3", self.cfg.num_hidden_layers)

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def warm(self) -> bool:
        """True once at least one generation has populated the resident cache."""
        return self._warm

    def _log_token(self, index: int, ids: list[int], token_id: int,
                   elapsed: float, state) -> None:
        """Append one JSON record per generated token.

        `cache_bytes` and `context` move with CONTEXT LENGTH. The routing side needs more
        care than it first appears:

        `experts_loaded` and `expert_bytes` are STRUCTURALLY CONSTANT during decode --
        every token routes top_k experts in each of the 92 MoE layers and loads 3
        projections each, so the count is always 92*16*3 = 4416 and the byte total
        follows from fixed shapes. Measured over 85 decode steps of the 2026-08-09
        after-run: one distinct value each, while s/token varied by 28.2 s. They cannot
        explain any variance, and logging them alone was a design error.

        What actually varies is WHICH experts get routed, and therefore whether each read
        hits the resident cache or goes to disk. That is `hits`/`misses`, so those are
        logged as per-token deltas. A slowdown tracking `cache_misses` is a routing
        problem; one tracking `context` is a KV-cache problem.

        Opened in append mode and flushed per token so an interrupted run keeps every
        record it produced, matching how run_long_job.py treats its own output.
        """
        try:
            from ..models.kimi_k3.model import _peak_memory
            mem = state.memory_report()
            cs = self.src.cache_stats()
            ph, pm = self._prev_cache
            self._prev_cache = (cs["hits"], cs["misses"])
            hits, misses = cs["hits"] - ph, cs["misses"] - pm
            rec = {
                "index": index,
                "token_id": token_id,
                "step_tokens": len(ids),          # prompt length on step 0, else 1
                "context": state.token_position,
                "elapsed_s": round(elapsed, 3),
                # per-token deltas -- these are the ones that actually vary
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate": round(hits / (hits + misses), 4) if hits + misses else 0.0,
                "bytes_read": self.src.bytes_read,
                "expert_bytes": self.src.expert_bytes,
                "experts_loaded": self.src.experts_loaded,
                "cache_bytes": mem["total"],
                "mla_cache_bytes": mem["mla_cache"],
                "kda_state_bytes": mem["kda_recurrent"] + mem["kda_conv"],
                # Peak allocator high-water mark. Without this the only way to learn what
                # a long prefill actually costs is to trip the ceiling and read the
                # MemoryError -- which is how the 2026-08-10 long-prompt attempts were
                # diagnosed, badly. A successful run should report its own headroom.
                "peak_bytes": _peak_memory(),
            }
            self.token_log.parent.mkdir(parents=True, exist_ok=True)
            with self.token_log.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            # Diagnostics must never be able to kill a generation that is hours in.
            log.warning("token log write failed", exc_info=True)

    def stats(self) -> dict:
        cs = self.src.cache_stats()
        return {
            "active": self.active,
            "completed": self.completed,
            "cancelled": self.cancelled,
            "warm": self._warm,
            "resident_bytes": cs["resident_bytes"],
            "cache_hit_rate": cs["hit_rate"],
            "uptime_s": time.time() - self.started_at,
        }

    # ----------------------------------------------------------------- generation
    async def generate(self, p: GenParams, is_disconnected) -> AsyncIterator[object]:
        """Yield `Step` per token, then a single `Finished`.

        `is_disconnected` is an async predicate; it is polled between tokens so a
        vanished client stops the work rather than paying for a full response nobody
        will read.
        """
        from .errors import overloaded, request_too_large

        if len(p.prompt_tokens) > self.max_input_tokens:
            raise request_too_large(
                f"Input of {len(p.prompt_tokens)} tokens exceeds the configured maximum "
                f"of {self.max_input_tokens}.", param="messages")

        if self._queue_slots.locked():
            raise overloaded()
        acquired = self._queue_slots.acquire()
        try:
            await asyncio.wait_for(acquired, timeout=0.001)
        except (asyncio.TimeoutError, Exception):
            raise overloaded()

        try:
            async with self._lock:
                self.active += 1
                try:
                    async for item in self._run(p, is_disconnected):
                        yield item
                finally:
                    self.active -= 1
        finally:
            self._queue_slots.release()

    async def _run(self, p: GenParams, is_disconnected) -> AsyncIterator[object]:
        from ..models.kimi_k3.recurrent_state import KimiK3State
        from ..models.kimi_k3.sampling import choose, full_logits
        import mlx.core as mx

        state = KimiK3State()
        produced: list[int] = []
        text_filter = TextFilter(p.stop)
        cancel = threading.Event()
        finish = "length"
        finished_cleanly = False

        def compute_step(ids: list[int]):
            """One forward pass + one sampled token. Runs off the event loop."""
            from ..models.kimi_k3.model import _reset_peak
            _reset_peak()
            if self.token_log is not None:
                self.src.reset_counters()
                t0 = time.time()
            h = self.model.forward_token(self.model.embed(ids), state)
            mx.eval(h)
            lg, _ = full_logits(self.model, h[:, -1, :])
            c = choose(lg, temperature=p.temperature, top_p=p.top_p,
                       seed=None if p.seed is None else p.seed + len(produced),
                       n_logprobs=p.top_logprobs if p.logprobs else 0,
                       logit_bias=p.logit_bias)
            state.advance(len(ids))
            if self.token_log is not None:
                self._log_token(len(produced), ids, c.token_id, time.time() - t0, state)
            return c

        def step(ids):
            # asyncio cancellation cannot stop a running Metal/Python worker.
            # Keep mutual exclusion in the worker itself until computation returns.
            with self._worker_lock:
                if cancel.is_set():
                    raise Cancelled()
                return compute_step(ids)

        n_max = min(p.max_tokens, self.max_output_tokens)
        try:
            for i in range(n_max):
                if cancel.is_set() or await is_disconnected():
                    finish = "cancelled"
                    break

                ids = p.prompt_tokens if i == 0 else [produced[-1]]
                c = await asyncio.to_thread(step, ids)
                self._warm = True

                produced.append(c.token_id)
                if c.token_id in self._eos_ids():
                    finish = "stop"
                    break
                raw = token_bytes(self.tokenizer, c.token_id)
                piece = text_filter.feed(raw)
                yield Step(token_id=c.token_id, text=piece, index=i,
                           logprob=c.logprob if p.logprobs else None,
                           top=c.top)
                if text_filter.stopped:
                    finish = "stop"
                    break
            else:
                finish = "length"

            tail = text_filter.finish()
            if tail:
                yield Step(token_id=-1, text=tail, index=len(produced))
            self.completed += 1
            yield Finished(reason=finish, prompt_tokens=len(p.prompt_tokens),
                           completion_tokens=len(produced))
            finished_cleanly = True
        finally:
            cancel.set()
            # GeneratorExit (consumer closed us), CancelledError (Starlette cancelled the
            # task group on client disconnect), or an early `break` all land here.
            if not finished_cleanly:
                self.cancelled += 1
                log.info("generation cancelled after %d token(s)", len(produced))

    def _eos_ids(self) -> set[int]:
        sp = self.tokenizer.special_tokens
        return {i for n, i in sp.items() if n in ("[EOS]", "<|end_of_msg|>")}

    @staticmethod
    def _hit_stop(text: str, stops: list[str]) -> bool:
        return any(s and s in text for s in stops)

    @staticmethod
    def trim_at_stop(text: str, stops: list[str]) -> str:
        cut = len(text)
        for s in stops:
            if s and (i := text.find(s)) != -1:
                cut = min(cut, i)
        return text[:cut]


def token_bytes(tokenizer, token_id):
    if hasattr(tokenizer, "decode_single_token_bytes"):
        return tokenizer.decode_single_token_bytes(token_id)
    return tokenizer.decode([token_id]).encode("utf-8")


class TextFilter:
    """Incremental UTF-8 decoding and stop matching before text is emitted."""
    def __init__(self, stops):
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.stops = stops
        self.pending = ""
        self.stopped = False

    def feed(self, raw):
        if self.stopped:
            return ""
        self.pending += self.decoder.decode(raw)
        cuts = [self.pending.index(s) for s in self.stops if s in self.pending]
        if cuts:
            out = self.pending[:min(cuts)]
            self.pending = ""
            self.stopped = True
            return out
        keep = max((n for s in self.stops for n in range(1, len(s))
                    if self.pending.endswith(s[:n])), default=0)
        cut = len(self.pending) - keep
        out, self.pending = self.pending[:cut], self.pending[cut:]
        return out

    def finish(self):
        if self.stopped:
            return ""
        out = self.feed(b"") + self.pending + self.decoder.decode(b"", final=True)
        self.pending = ""
        return out
