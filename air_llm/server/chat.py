"""`/v1/chat/completions` — the compatibility baseline.

Prompt construction
-------------------
K3 ships an XTML chat format (`encoding_k3.py`) that this project does not implement, so
messages are rendered with a plain, documented template. That is a **known deviation**
and is recorded in the compatibility matrix rather than presented as chat-tuned
behaviour.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Any

from .errors import invalid_request, unsupported_capability, unsupported_parameter
from .registry import ServedModel
from .schemas import ChatRequest, Message


def new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


# ------------------------------------------------------------------ prompt render

ROLE_TAG = {"system": "System", "developer": "System", "user": "User",
            "assistant": "Assistant", "tool": "Tool", "function": "Tool"}


def _content_text(m: Message, *, caps) -> str:
    if m.content is None:
        return ""
    if isinstance(m.content, str):
        return m.content
    parts: list[str] = []
    for p in m.content:
        t = p.get("type")
        if t == "text":
            parts.append(str(p.get("text", "")))
        elif t in ("image_url", "input_image", "image"):
            if not caps.vision:
                raise unsupported_capability(
                    "messages[].content",
                    "This model is text-only; image content parts are not supported. "
                    "The checkpoint ships a vision tower but it is not implemented.")
        elif t in ("input_audio", "audio"):
            raise unsupported_capability("messages[].content",
                                         "Audio content parts are not supported.")
        else:
            parts.append(str(p.get("text", "")))
    return "".join(parts)


def render_prompt(messages: list[Message], *, caps) -> str:
    """Flatten messages into a single prompt string.

    Named messages keep their name (`User (alice):`) so multi-party transcripts survive
    the round trip.
    """
    lines: list[str] = []
    for m in messages:
        if m.role in ("tool", "function"):
            raise unsupported_capability(
                "messages[].role",
                "Tool messages are not supported because this backend cannot generate "
                "tool calls (no constrained decoding). See /v1/models capabilities.")
        if m.tool_calls:
            raise unsupported_capability(
                "messages[].tool_calls",
                "Assistant tool_calls are not supported by this backend.")
        tag = ROLE_TAG.get(m.role, "User")
        if m.name:
            tag = f"{tag} ({m.name})"
        text = _content_text(m, caps=caps)
        lines.append(f"{tag}: {text}" if text else f"{tag}:")
    lines.append("Assistant:")
    return "\n".join(lines)


# --------------------------------------------------------------------- validation

def validate(req: ChatRequest, model: ServedModel, *, max_n: int) -> None:
    """Reject anything we cannot honour, naming the parameter.

    Ordered so the most consequential rejections come first: a caller who asked for
    tools cares about that far more than about `presence_penalty`.
    """
    caps = model.capabilities

    if req.tools or req.functions:
        raise unsupported_capability(
            "tools",
            "Function/tool calling is not supported by this deployment. The backend has "
            "no constrained decoding, and emitting tool calls by prompting alone cannot "
            "be made reliable. `/v1/models` reports capabilities.tools = false.")
    if req.tool_choice not in (None, "none") or req.function_call not in (None, "none"):
        raise unsupported_capability("tool_choice",
                                     "Tool calling is not supported by this deployment.")

    if req.response_format is not None:
        t = req.response_format.type
        if t == "json_schema" and not caps.json_schema:
            raise unsupported_capability(
                "response_format",
                "Strict JSON Schema output is not supported: this backend has no "
                "constrained decoding, so schema conformance cannot be enforced. "
                "Returning unenforced JSON would misrepresent the guarantee.")
        if t == "json_object" and not caps.json_object:
            raise unsupported_capability(
                "response_format",
                "response_format=json_object is not supported: without constrained "
                "decoding, valid JSON cannot be guaranteed.")

    if req.presence_penalty not in (None, 0.0):
        raise unsupported_parameter(
            "presence_penalty",
            "presence_penalty is not implemented; it would materially change sampling. "
            "Omit it or set it to 0.")
    if req.frequency_penalty not in (None, 0.0):
        raise unsupported_parameter(
            "frequency_penalty",
            "frequency_penalty is not implemented; it would materially change sampling. "
            "Omit it or set it to 0.")

    if req.n is not None and req.n > max_n:
        raise invalid_request(
            f"n={req.n} exceeds the configured maximum of {max_n}. Each additional "
            f"choice costs a full generation (~{caps.approx_seconds_per_token:.0f} s per "
            f"token).", param="n", code="invalid_value")

    if req.top_logprobs and not req.logprobs:
        raise invalid_request(
            "`top_logprobs` requires `logprobs` to be true.", param="top_logprobs",
            code="invalid_value")

    if req.logit_bias:
        for k in req.logit_bias:
            try:
                int(k)
            except ValueError:
                raise invalid_request(f"logit_bias keys must be token ids; got {k!r}.",
                                      param="logit_bias", code="invalid_value")


def parse_logit_bias(raw: dict[str, float] | None) -> dict[int, float] | None:
    return None if not raw else {int(k): float(v) for k, v in raw.items()}


# ----------------------------------------------------------------------- responses

def logprobs_payload(steps) -> dict:
    """OpenAI `choices[].logprobs` for chat: {"content": [ {token, logprob, top_logprobs} ]}"""
    content = []
    for s in steps:
        if s.logprob is None:
            continue
        content.append({
            "token": s.text,
            "logprob": s.logprob,
            "bytes": list(s.text.encode("utf-8")),
            "top_logprobs": [{"token": t, "logprob": lp, "bytes": list(t.encode())}
                             for t, lp in s.top_decoded],
        })
    return {"content": content}


def completion(cid: str, created: int, model_id: str, text: str, finish: str,
               usage: dict, *, logprobs: dict | None = None, index: int = 0) -> dict:
    choice: dict[str, Any] = {
        "index": index,
        "message": {"role": "assistant", "content": text},
        "finish_reason": finish,
        "logprobs": logprobs,
    }
    return {
        "id": cid,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [choice],
        "usage": usage,
    }


def chunk(cid: str, created: int, model_id: str, *, delta: dict | None = None,
          finish: str | None = None, index: int = 0,
          usage: dict | None = None, logprobs: dict | None = None) -> dict:
    d: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [] if usage is not None and delta is None and finish is None else [{
            "index": index,
            "delta": delta or {},
            "finish_reason": finish,
            "logprobs": logprobs,
        }],
    }
    if usage is not None:
        d["usage"] = usage
    return d


def usage_of(prompt_tokens: int, completion_tokens: int) -> dict:
    return {"prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}


def sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n"


DONE = "data: [DONE]\n\n"
