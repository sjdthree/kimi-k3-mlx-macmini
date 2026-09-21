"""Conformance suite: the OFFICIAL OpenAI Python SDK against our server.

This is the acceptance bar the spec sets — "the official SDK works by setting only
base_url and api_key". So the tests use the real `openai` package, not hand-rolled
requests: the SDK validates response shapes itself, and it fails loudly on anything it
doesn't recognise.

Runs against the mock engine so HTTP behaviour is tested in seconds. Model quality is
covered by the real-checkpoint suites; nothing here depends on the weights.

    python tests/test_openai_conformance.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("AIR_LLM_API_KEYS", "sk-test-key,sk-other")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from openai import OpenAI  # noqa: E402
from openai import APIStatusError, AuthenticationError, NotFoundError  # noqa: E402

from air_llm.server.app import create_app  # noqa: E402
from air_llm.server.mock import MockEngine  # noqa: E402

FAILS = 0
PORT = 8137
BASE = f"http://127.0.0.1:{PORT}"


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


class Server(threading.Thread):
    daemon = True

    def __init__(self):
        super().__init__()
        engine = MockEngine(delay=0.01)
        app = create_app(engine, {"max_n": 1, "default_max_tokens": 8})

        @app.on_event("startup")
        async def _s():
            await engine.warmup()

        self.engine = engine
        self.config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def wait_ready(self, timeout=20):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                if httpx.get(f"{BASE}/readyz", timeout=1).status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False


def main():
    srv = Server()
    srv.start()
    if not srv.wait_ready():
        print("server failed to start")
        return 1

    print("OpenAI SDK conformance suite")
    print("=" * 72)

    # ---- 1. SDK works with only base_url + api_key
    client = OpenAI(base_url=f"{BASE}/v1", api_key="sk-test-key", timeout=60.0)
    check("1. official SDK constructs with only base_url + api_key", True)

    # ---- 2. models.list
    models = client.models.list()
    ids = {m.id for m in models.data}
    check("2. models.list() returns the deployment id and its aliases",
          {"kimi-k3-local", "chat", "reasoning", "kimi-k3"} <= ids, f"{sorted(ids)}")
    raw = httpx.get(f"{BASE}/v1/models",
                    headers={"Authorization": "Bearer sk-test-key"}).json()
    caps = next(d["air_llm"]["capabilities"] for d in raw["data"] if d["id"] == "chat")
    check("   capabilities advertised in namespaced metadata",
          caps["tools"] is False and caps["embeddings"] is False
          and caps["context_length"] > 0,
          f"tools={caps['tools']} json_schema={caps['json_schema']} "
          f"ctx={caps['context_length']}")

    # ---- 3. non-streaming chat
    r = client.chat.completions.create(
        model="chat", messages=[{"role": "user", "content": "hello"}], max_tokens=5)
    check("3. non-streaming completion is OpenAI-shaped",
          r.object == "chat.completion" and r.id.startswith("chatcmpl-")
          and r.choices[0].message.role == "assistant"
          and r.choices[0].finish_reason in ("stop", "length"),
          f"id={r.id[:20]} finish={r.choices[0].finish_reason}")
    check("   usage is reported",
          r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens,
          f"{r.usage.prompt_tokens}+{r.usage.completion_tokens}={r.usage.total_tokens}")

    # ---- 4. streaming
    chunks, content = [], ""
    stream = client.chat.completions.create(
        model="chat", messages=[{"role": "user", "content": "hi"}],
        max_tokens=6, stream=True)
    for ch in stream:
        chunks.append(ch)
        if ch.choices and ch.choices[0].delta.content:
            content += ch.choices[0].delta.content
    check("4. streaming yields chat.completion.chunk objects",
          len(chunks) >= 3 and all(c.object == "chat.completion.chunk" for c in chunks),
          f"{len(chunks)} chunks, content={content!r}")
    check("   first chunk carries the assistant role",
          chunks[0].choices[0].delta.role == "assistant")
    check("   final chunk carries finish_reason",
          any(c.choices and c.choices[0].finish_reason for c in chunks))

    # raw SSE framing: the SDK hides [DONE], so check the wire directly
    with httpx.stream("POST", f"{BASE}/v1/chat/completions",
                      headers={"Authorization": "Bearer sk-test-key"},
                      json={"model": "chat", "stream": True, "max_tokens": 3,
                            "messages": [{"role": "user", "content": "x"}]},
                      timeout=30) as resp:
        body = "".join(resp.iter_text())
    check("   SSE terminates with data: [DONE]", body.rstrip().endswith("data: [DONE]"))
    check("   every SSE frame is `data: ` prefixed",
          all(l.startswith("data: ") for l in body.strip().split("\n\n") if l.strip()))

    # ---- stream_options.include_usage
    usage_seen = None
    for ch in client.chat.completions.create(
            model="chat", messages=[{"role": "user", "content": "u"}],
            max_tokens=4, stream=True,
            stream_options={"include_usage": True}):
        if ch.usage:
            usage_seen = ch.usage
    check("   stream_options.include_usage emits a usage chunk",
          usage_seen is not None and usage_seen.completion_tokens > 0,
          f"{usage_seen}")

    # ---- 5. client disconnect cancels generation
    before = srv.engine.cancelled
    slow = MockEngineDelayProbe(srv)
    with httpx.Client(timeout=5) as c:
        try:
            with c.stream("POST", f"{BASE}/v1/chat/completions",
                          headers={"Authorization": "Bearer sk-test-key"},
                          json={"model": "chat", "stream": True, "max_tokens": 200,
                                "messages": [{"role": "user", "content": "long"}]}) as r:
                for i, _ in enumerate(r.iter_lines()):
                    if i > 3:
                        break          # hang up early
        except Exception:
            pass
    deadline = time.time() + 10
    while srv.engine.cancelled == before and time.time() < deadline:
        time.sleep(0.2)
    check("5. client disconnect cancels the generation",
          srv.engine.cancelled > before,
          f"cancelled {before} -> {srv.engine.cancelled} in "
          f"{10 - max(deadline - time.time(), 0):.1f}s")

    # ---- 6/7. unsupported capabilities produce structured errors, not silence
    def expect_error(label, payload, want_param, want_code):
        resp = httpx.post(f"{BASE}/v1/chat/completions",
                          headers={"Authorization": "Bearer sk-test-key"},
                          json=payload, timeout=30)
        ok = resp.status_code == 400
        body = resp.json().get("error", {})
        ok = ok and body.get("param") == want_param and body.get("code") == want_code
        check(label, ok, f"{resp.status_code} {body.get('code')} param={body.get('param')}")

    base_msg = [{"role": "user", "content": "x"}]
    expect_error("6. tools request -> unsupported_capability (not silently ignored)",
                 {"model": "chat", "messages": base_msg,
                  "tools": [{"type": "function",
                             "function": {"name": "f", "parameters": {}}}]},
                 "tools", "unsupported_capability")
    expect_error("7. response_format json_schema -> unsupported_capability",
                 {"model": "chat", "messages": base_msg,
                  "response_format": {"type": "json_schema",
                                      "json_schema": {"name": "s", "schema": {}}}},
                 "response_format", "unsupported_capability")
    expect_error("   response_format json_object -> unsupported_capability",
                 {"model": "chat", "messages": base_msg,
                  "response_format": {"type": "json_object"}},
                 "response_format", "unsupported_capability")
    expect_error("   presence_penalty -> unsupported_parameter",
                 {"model": "chat", "messages": base_msg, "presence_penalty": 1.0},
                 "presence_penalty", "unsupported_parameter")

    # ---- 8. embeddings must not be routed to a chat model
    try:
        client.embeddings.create(model="chat", input=["a", "b"])
        check("8. embeddings on a chat model is refused", False, "it succeeded")
    except NotFoundError as e:
        check("8. embeddings on a chat model returns 404 model_not_found",
              e.body.get("code") == "model_not_found", f"{e.body.get('code')}")

    # ---- 11. error taxonomy
    try:
        OpenAI(base_url=f"{BASE}/v1", api_key="wrong").models.list()
        check("11. bad key -> 401", False, "it succeeded")
    except AuthenticationError as e:
        check("11. bad key -> 401 invalid_api_key",
              e.body.get("code") == "invalid_api_key", f"{e.body.get('code')}")

    try:
        client.chat.completions.create(model="does-not-exist",
                                       messages=base_msg, max_tokens=1)
        check("    unknown model -> 404", False, "it succeeded")
    except NotFoundError as e:
        check("    unknown model -> 404 model_not_found",
              e.body.get("code") == "model_not_found" and e.body.get("param") == "model")

    r = httpx.post(f"{BASE}/v1/chat/completions",
                   headers={"Authorization": "Bearer sk-test-key"},
                   json={"model": "chat"}, timeout=10)
    check("    malformed body -> 400 with OpenAI error envelope",
          r.status_code == 400 and set(r.json()["error"]) >= {"message", "type", "code"},
          f"{r.status_code} {r.json().get('error', {}).get('code')}")

    check("    X-Request-ID present on responses", bool(r.headers.get("X-Request-ID")))
    rid = "req_client_supplied_123"
    r2 = httpx.get(f"{BASE}/v1/models",
                   headers={"Authorization": "Bearer sk-test-key", "X-Request-ID": rid})
    check("    client-supplied X-Request-ID is echoed",
          r2.headers.get("X-Request-ID") == rid)

    # ---- errors never leak internals
    body_text = json.dumps(r.json())
    leaked = [s for s in ("Traceback", "/Users/", "/Volumes/", "site-packages")
              if s in body_text]
    check("    errors leak no paths or stack traces", not leaked, f"{leaked}")

    # ---- logprobs
    rl = client.chat.completions.create(
        model="chat", messages=base_msg, max_tokens=3, logprobs=True, top_logprobs=2)
    lp = rl.choices[0].logprobs
    check("    logprobs + top_logprobs returned in OpenAI shape",
          lp is not None and len(lp.content) > 0
          and len(lp.content[0].top_logprobs) == 2,
          f"{len(lp.content) if lp else 0} entries")

    check("    logprob bytes are the token's raw UTF-8 bytes",
          all(c.bytes == list(c.token.encode()) for c in lp.content))

    # ---- stop strings: matched across token boundaries, never partially emitted.
    # The mock emits " fox", " stops", " over", " the", ...; "s ov" spans two tokens.
    rs = client.chat.completions.create(
        model="chat", messages=base_msg, max_tokens=8, stop=["s ov"])
    check("    stop spanning two tokens trims there, finish_reason=stop",
          rs.choices[0].message.content == " fox stop"
          and rs.choices[0].finish_reason == "stop",
          f"{rs.choices[0].message.content!r} {rs.choices[0].finish_reason}")
    streamed = "".join(
        ch.choices[0].delta.content or "" for ch in client.chat.completions.create(
            model="chat", messages=base_msg, max_tokens=8, stop=["s ov"], stream=True)
        if ch.choices)
    check("    streamed text never includes a partial stop match", streamed == " fox stop",
          f"{streamed!r}")

    # ---- requests the backend cannot honour exactly are refused, not approximated
    expect_error("    unknown request field -> unsupported_parameter",
                 {"model": "chat", "messages": base_msg, "not_a_field": 1},
                 "not_a_field", "unsupported_parameter")
    expect_error("    extra_body -> unsupported_parameter",
                 {"model": "chat", "messages": base_msg,
                  "extra_body": {"air_llm": {"priority": "interactive"}}},
                 "extra_body", "unsupported_parameter")
    expect_error("    n=2 -> invalid_value",
                 {"model": "chat", "messages": base_msg, "n": 2}, "n", "invalid_value")
    expect_error("    stop with logprobs -> unsupported_parameter",
                 {"model": "chat", "messages": base_msg, "stop": ["x"], "logprobs": True},
                 "logprobs", "unsupported_parameter")
    expect_error("    five stop strings -> 400",
                 {"model": "chat", "messages": base_msg, "stop": list("abcde")},
                 "stop", None)
    expect_error("    unknown content part type -> unsupported_capability",
                 {"model": "chat",
                  "messages": [{"role": "user", "content": [{"type": "file", "x": 1}]}]},
                 "messages[].content", "unsupported_capability")

    # ---- a UTF-8 character split across two tokens is emitted once, whole
    from air_llm.server.engine import TextFilter
    tf = TextFilter([])
    parts = [tf.feed(b"\xe2\x82"), tf.feed(b"\xac!"), tf.finish()]
    check("    multibyte character split across tokens decodes intact",
          parts == ["", "€!", ""], f"{parts}")

    # ---- daily quota is reserved up front, so a request cannot overrun it
    from air_llm.server.auth import Auth, KeyPolicy
    from air_llm.server.errors import APIError
    quota = Auth()
    policy = KeyPolicy(daily_tokens=10)
    quota.enter("key_q", policy, 8)
    quota.leave("key_q")
    try:
        quota.enter("key_q", policy, 8)
        check("    request over the remaining daily quota -> 403", False, "admitted")
    except APIError as e:
        check("    request over the remaining daily quota -> 403 insufficient_quota",
              e.status == 403 and e.code == "insufficient_quota")

    # ---- operational endpoints outside /v1
    check("    /healthz outside the /v1 namespace",
          httpx.get(f"{BASE}/healthz").status_code == 200)
    check("    /readyz reports readiness",
          httpx.get(f"{BASE}/readyz").json()["status"] == "ready")
    mtext = httpx.get(f"{BASE}/metrics").text
    check("    /metrics exposes Prometheus counters",
          "air_llm_requests_total" in mtext and "air_llm_time_to_first_token" in mtext)

    # ---- native namespace is separate
    nat = httpx.get(f"{BASE}/native/v1/engine",
                    headers={"Authorization": "Bearer sk-test-key"}).json()
    check("    /native/v1 exposes backend controls separately", nat.get("ready") is True)

    # ---- standard response shape carries no un-namespaced extras
    plain = httpx.post(f"{BASE}/v1/chat/completions",
                       headers={"Authorization": "Bearer sk-test-key"},
                       json={"model": "chat", "messages": base_msg,
                             "max_tokens": 2}).json()
    allowed = {"id", "object", "created", "model", "choices", "usage",
               "system_fingerprint", "service_tier"}
    extra = set(plain) - allowed
    check("    chat.completion carries no non-standard top-level fields", not extra,
          f"extra={extra}")

    # ---- OpenAPI document is published
    spec = httpx.get(f"{BASE}/openapi.json").json()
    check("    OpenAPI spec published with the documented paths",
          {"/v1/chat/completions", "/v1/models"} <= set(spec["paths"]),
          f"{len(spec['paths'])} paths")
    published = json.loads((Path(__file__).resolve().parent.parent / "docs" / "openapi.json")
                           .read_text())
    check("    docs/openapi.json matches the served spec (regenerate if this fails)",
          published == spec)

    print("=" * 72)
    print(f"CONFORMANCE: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


class MockEngineDelayProbe:
    """Slows the mock so the disconnect test has time to hang up mid-stream."""

    def __init__(self, srv):
        srv.engine.delay = 0.15


if __name__ == "__main__":
    sys.exit(main())
