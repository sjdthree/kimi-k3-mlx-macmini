"""OpenAI-compatible HTTP facade for the Kimi-K3 MLX backend.

    OPENAI_BASE_URL=http://host:8000/v1
    OPENAI_API_KEY=<key>

Design contract
---------------
* `/v1/*` is the stable external surface and holds no backend-specific fields. Anything
  extra lives under a namespaced `air_llm` block or the `extra_body.air_llm` opt-in, so a
  strict OpenAI client sees exactly the shapes it expects.
* `/native/v1/*` is where backend-specific controls live (cache residency, storage tiers,
  engine stats). Versioned separately so compatibility never blocks a performance feature.
* Operational endpoints (`/healthz`, `/readyz`, `/metrics`) sit outside `/v1`.

Run:
    python -m air_llm.serve --model /path/to/Kimi-K3 --cache ~/k3_dense --resident-gb 32
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import chat as C
from . import errors as E
from . import metrics as M
from . import registry as R
from .auth import Auth, KeyPolicy
from .engine import Engine, Finished, GenParams, Step
from .schemas import ChatRequest, CompletionsRequest, EmbeddingsRequest

log = logging.getLogger("air_llm.server")

STATE: dict[str, Any] = {"engine": None, "auth": None, "config": {}}


# --------------------------------------------------------------------- middleware

async def request_context(request: Request, call_next):
    """Assign/propagate a request id and stamp it on every response."""
    rid = request.headers.get("x-request-id") or "req_" + uuid.uuid4().hex[:24]
    request.state.request_id = rid
    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled rid=%s path=%s", rid, request.url.path)
        raise
    response.headers["X-Request-ID"] = rid
    log.info("rid=%s %s %s -> %s in %.3fs", rid, request.method, request.url.path,
             response.status_code, time.time() - t0)
    return response


def require_auth(request: Request) -> tuple[str, KeyPolicy]:
    auth: Auth = STATE["auth"]
    kid, policy = auth.authenticate(request)
    auth.check_rate(kid, policy)
    return kid, policy


# --------------------------------------------------------------------------- app

def create_app(engine: Engine | None = None, config: dict | None = None) -> FastAPI:
    app = FastAPI(
        title="Kimi-K3 MLX local API",
        version="1.0.0",
        description="A documented subset of the OpenAI Chat Completions API over the "
                    "Kimi-K3 MLX backend. Disk-bound: the best recorded decode was "
                    "~32.4 s per token, and prefill can take many minutes.",
        docs_url="/docs", openapi_url="/openapi.json",
    )
    STATE["engine"] = engine
    STATE["auth"] = Auth()
    STATE["config"] = config or {}
    app.middleware("http")(request_context)
    E.install(app)

    v1 = APIRouter(prefix="/v1", tags=["openai"])
    native = APIRouter(prefix="/native/v1", tags=["native"])

    # ------------------------------------------------------------------ models
    @v1.get("/models")
    async def list_models(auth=Depends(require_auth)):
        return {"object": "list", "data": R.list_models()}

    @v1.get("/models/{model_id:path}")
    async def get_model(model_id: str, auth=Depends(require_auth)):
        m = R.resolve(model_id)
        return m.as_openai(model_id=model_id)

    # ---------------------------------------------------------- chat completions
    @v1.post("/chat/completions")
    async def chat_completions(req: ChatRequest, request: Request,
                               auth=Depends(require_auth)):
        kid, policy = auth
        eng: Engine = STATE["engine"]
        if eng is None or not eng.ready:
            raise E.APIError(503, "The model is still loading. Retry shortly.",
                             type="api_error", code="model_not_ready")

        model = R.resolve(req.model)
        cfg = STATE["config"]
        C.validate(req, model, max_n=cfg.get("max_n", 1))

        n_out = min(req.resolved_max_tokens(cfg.get("default_max_tokens", 16)),
                    policy.max_output_tokens, model.capabilities.max_output_tokens)

        prompt = C.render_prompt(req.messages, caps=model.capabilities)
        tok = eng.tokenizer
        prompt_ids = tok.encode(prompt)

        params = GenParams(
            prompt_tokens=prompt_ids,
            max_tokens=n_out,
            temperature=req.temperature or 0.0,
            top_p=req.top_p if req.top_p is not None else 1.0,
            seed=req.seed,
            stop=req.stop_list(),
            logprobs=bool(req.logprobs),
            top_logprobs=req.top_logprobs or 0,
            logit_bias=C.parse_logit_bias(req.logit_bias),
        )

        cid, created = C.new_id(), int(time.time())
        mid = req.model
        want_usage = bool(req.stream_options and req.stream_options.include_usage)

        STATE["auth"].enter(kid, policy, params.max_tokens)
        if req.stream:
            return StreamingResponse(
                _stream(eng, params, request, cid, created, mid, kid, want_usage,
                        bool(req.logprobs)),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                         "X-Request-ID": request.state.request_id})
        try:
            return await _blocking(eng, params, request, cid, created, mid,
                                   bool(req.logprobs))
        finally:
            STATE["auth"].leave(kid)

    # ------------------------------------------------------------- embeddings
    @v1.post("/embeddings")
    async def embeddings(req: EmbeddingsRequest, auth=Depends(require_auth)):
        # Spec requirement: never route embeddings to a chat-only model. There is no
        # embedding model deployed, so this is a 404 on the model, not a bad response.
        R.resolve(req.model)          # 404s for unknown ids first
        raise E.APIError(
            404,
            f"The model '{req.model}' does not support embeddings, and no embedding "
            f"model is deployed. Requests are not routed to chat-only models. "
            f"See GET /v1/models for capabilities.",
            type="invalid_request_error", param="model",
            code="model_not_found")

    # ------------------------------------------------------------- completions
    @v1.post("/completions")
    async def completions(req: CompletionsRequest, auth=Depends(require_auth)):
        raise E.APIError(
            404,
            "The legacy /v1/completions endpoint is not implemented. Use "
            "/v1/chat/completions, which is the compatibility baseline.",
            type="invalid_request_error", code="endpoint_not_found")

    @v1.post("/responses")
    async def responses(request: Request, auth=Depends(require_auth)):
        raise E.APIError(
            404,
            "The /v1/responses endpoint is not implemented in this release. Use "
            "/v1/chat/completions.",
            type="invalid_request_error", code="endpoint_not_found")

    # ----------------------------------------------------------------- native
    @native.get("/engine")
    async def engine_stats(auth=Depends(require_auth)):
        eng: Engine = STATE["engine"]
        if eng is None:
            return {"ready": False}
        return {"ready": eng.ready, **eng.stats(), "config": STATE["config"]}

    @native.get("/storage")
    async def storage(auth=Depends(require_auth)):
        eng: Engine = STATE["engine"]
        cs = eng.src.cache_stats()
        return {
            "resident_bytes": cs["resident_bytes"],
            "resident_tensors": cs["entries"],
            "hit_rate": cs["hit_rate"],
            "policy": "fill-and-pin (LRU is pathological for a cyclic layer scan)",
        }

    app.include_router(v1)
    app.include_router(native)

    # ------------------------------------------------------------ operational
    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz():
        eng: Engine = STATE["engine"]
        ready = bool(eng and eng.ready)
        M.model_ready.set(1 if ready else 0)
        return JSONResponse({"status": "ready" if ready else "loading",
                             "warm": bool(eng and eng.warm)},
                            status_code=200 if ready else 503)

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        eng: Engine = STATE["engine"]
        if eng is not None:
            s = eng.stats()
            M.resident_bytes.set(s["resident_bytes"])
            M.cache_hit_ratio.set(s["cache_hit_rate"])
            M.active_requests.set(s["active"])
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# ------------------------------------------------------------------- generation

def _decorate_top(steps, tok):
    """Attach decoded strings to top-logprob token ids."""
    for s in steps:
        from .engine import token_bytes
        s.raw_bytes = token_bytes(tok, s.token_id) if s.token_id >= 0 else b""
        s.top_decoded = [(token_bytes(tok, t), lp) for t, lp in s.top]
    return steps


async def _blocking(eng: Engine, params: GenParams, request: Request, cid: str,
                    created: int, mid: str, want_logprobs: bool) -> JSONResponse:
    async def disconnected():
        return await request.is_disconnected()

    steps: list[Step] = []
    fin: Finished | None = None
    t0 = time.time()
    ttft = None
    async for item in eng.generate(params, disconnected):
        if isinstance(item, Step):
            if ttft is None:
                ttft = time.time() - t0
                M.ttft_seconds.labels(mid).observe(ttft)
            steps.append(item)
        else:
            fin = item

    assert fin is not None
    text = "".join(s.text for s in steps)
    if params.stop:
        text = eng.trim_at_stop(text, params.stop)

    dt = time.time() - t0
    M.request_seconds.labels("chat.completions", mid).observe(dt)
    if fin.completion_tokens and dt > 0:
        M.decode_tokens_per_second.labels(mid).observe(fin.completion_tokens / dt)
    M.prompt_tokens_total.labels(mid).inc(fin.prompt_tokens)
    M.completion_tokens_total.labels(mid).inc(fin.completion_tokens)
    if fin.reason == "cancelled":
        M.cancellations_total.labels(mid).inc()

    lp = None
    if want_logprobs:
        lp = C.logprobs_payload(_decorate_top(steps, eng.tokenizer))

    body = C.completion(cid, created, mid, text, fin.reason,
                        C.usage_of(fin.prompt_tokens, fin.completion_tokens),
                        logprobs=lp)
    M.requests_total.labels("chat.completions", mid, "200", "-").inc()
    return JSONResponse(body, headers={"X-Request-ID": request.state.request_id})


async def _stream(eng: Engine, params: GenParams, request: Request, cid: str,
                  created: int, mid: str, kid: str, want_usage: bool,
                  want_logprobs: bool):
    """SSE generator: role chunk, content deltas, finish chunk, optional usage, [DONE]."""
    t0 = time.time()
    ttft = None
    fin: Finished | None = None
    emitted = 0

    async def disconnected():
        return await request.is_disconnected()

    try:
        yield C.sse(C.chunk(cid, created, mid, delta={"role": "assistant", "content": ""}))
        async for item in eng.generate(params, disconnected):
            if isinstance(item, Step):
                if ttft is None:
                    ttft = time.time() - t0
                    M.ttft_seconds.labels(mid).observe(ttft)
                lp = None
                if want_logprobs and item.logprob is not None:
                    _decorate_top([item], eng.tokenizer)
                    lp = C.logprobs_payload([item])
                emitted += 1
                yield C.sse(C.chunk(cid, created, mid, delta={"content": item.text},
                                    logprobs=lp))
            else:
                fin = item

        reason = fin.reason if fin else "stop"
        yield C.sse(C.chunk(cid, created, mid, delta={}, finish=reason))
        if want_usage and fin:
            yield C.sse(C.chunk(cid, created, mid,
                                usage=C.usage_of(fin.prompt_tokens,
                                                 fin.completion_tokens)))
        yield C.DONE

        dt = time.time() - t0
        M.request_seconds.labels("chat.completions", mid).observe(dt)
        if fin:
            M.prompt_tokens_total.labels(mid).inc(fin.prompt_tokens)
            M.completion_tokens_total.labels(mid).inc(fin.completion_tokens)
            if fin.completion_tokens and dt > 0:
                M.decode_tokens_per_second.labels(mid).observe(fin.completion_tokens / dt)
            if fin.reason == "cancelled":
                M.cancellations_total.labels(mid).inc()
        M.requests_total.labels("chat.completions", mid, "200", "-").inc()

    except asyncio.CancelledError:
        # Client vanished mid-stream. The engine sees the disconnect on its next poll.
        M.cancellations_total.labels(mid).inc()
        log.info("stream cancelled rid=%s after %d tokens",
                 request.state.request_id, emitted)
        raise
    except E.APIError as exc:
        # An error after headers are sent cannot change the status code, so surface it
        # inside the stream where an SSE client can still see it.
        yield C.sse(exc.envelope())
        yield C.DONE
    except Exception:
        log.exception("stream failed rid=%s", request.state.request_id)
        yield C.sse(E.internal_error(request.state.request_id).envelope())
        yield C.DONE
    finally:
        STATE["auth"].leave(kid)
