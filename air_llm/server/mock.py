"""A stub engine that satisfies the Engine interface without any weights.

The real backend takes ~34 s per token, which makes it useless for exercising HTTP
behaviour: SSE framing, cancellation, auth, error shapes and SDK conformance have nothing
to do with the model and everything to do with the server. This mock lets the conformance
suite run those in seconds.

It is deliberately NOT a fake model: it emits deterministic filler text. Anything that
depends on model quality is tested against the real engine instead.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .engine import Finished, GenParams, Step, TextFilter


class _Tok:
    """Minimal tokenizer stand-in: whitespace tokens, stable ids."""

    special_tokens = {"[EOS]": 999999}

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        return [abs(hash(w)) % 100000 for w in text.split()] or [0]

    def decode(self, ids) -> str:
        return "".join(_WORDS[i % len(_WORDS)] for i in ids)


_WORDS = [" the", " quick", " brown", " fox", " jumps", " over", " lazy", " dog",
          " and", " then", " stops", "."]


class _Src:
    def cache_stats(self):
        return {"hits": 0, "misses": 0, "hit_rate": 0.0,
                "resident_bytes": 0, "entries": 0}


class MockEngine:
    def __init__(self, delay: float = 0.01):
        self.tokenizer = _Tok()
        self.src = _Src()
        self.delay = delay
        self._ready = False
        self._warm = False
        self.active = 0
        self.completed = 0
        self.cancelled = 0
        self.started_at = time.time()
        self.max_output_tokens = 4096
        self.max_input_tokens = 8192
        self._lock = asyncio.Semaphore(1)

    async def warmup(self):
        self._ready = True

    @property
    def ready(self):
        return self._ready

    @property
    def warm(self):
        return self._warm

    def stats(self):
        return {"active": self.active, "completed": self.completed,
                "cancelled": self.cancelled, "warm": self._warm,
                "resident_bytes": 0, "cache_hit_rate": 0.0,
                "uptime_s": time.time() - self.started_at, "mock": True}

    async def generate(self, p: GenParams, is_disconnected) -> AsyncIterator[object]:
        from .errors import request_too_large
        if len(p.prompt_tokens) > self.max_input_tokens:
            raise request_too_large(
                f"Input of {len(p.prompt_tokens)} tokens exceeds the configured maximum "
                f"of {self.max_input_tokens}.", param="messages")

        async with self._lock:
            self.active += 1
            produced, text, finish = [], "", "length"
            finished_cleanly = False
            text_filter = TextFilter(p.stop)
            try:
                n = min(p.max_tokens, self.max_output_tokens)
                for i in range(n):
                    if await is_disconnected():
                        finish = "cancelled"
                        break
                    await asyncio.sleep(self.delay)
                    self._warm = True
                    tid = (i * 7 + 3) % 100000
                    piece = text_filter.feed(self.tokenizer.decode([tid]).encode())
                    produced.append(tid)
                    text += piece
                    top = [(tid, -0.1), ((tid + 1) % 100000, -2.3)][:max(p.top_logprobs, 0)]
                    yield Step(token_id=tid, text=piece, index=i,
                               logprob=-0.1 if p.logprobs else None, top=top)
                    if text_filter.stopped:
                        finish = "stop"
                        break
                tail = text_filter.finish()
                if tail:
                    yield Step(token_id=-1, text=tail, index=len(produced))
                self.completed += 1
                yield Finished(reason=finish, prompt_tokens=len(p.prompt_tokens),
                               completion_tokens=len(produced))
                finished_cleanly = True
            finally:
                # Mirrors Engine._run: GeneratorExit / CancelledError from a client
                # hang-up lands here. Polling is_disconnected() alone never fires under
                # StreamingResponse, which owns the receive channel.
                if not finished_cleanly:
                    self.cancelled += 1
                self.active -= 1

    @staticmethod
    def trim_at_stop(text: str, stops: list[str]) -> str:
        cut = len(text)
        for s in stops:
            if s and (i := text.find(s)) != -1:
                cut = min(cut, i)
        return text[:cut]
