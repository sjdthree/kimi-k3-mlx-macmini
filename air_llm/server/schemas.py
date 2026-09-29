"""OpenAI Chat Completions request/response schemas.

Validation is deliberately permissive about *shape* and strict about *semantics*: we
accept the full OpenAI request body so SDK clients never fail schema validation, then
reject — with a named `param` — any field we cannot actually honour. Silently dropping
`tools` or `response_format` would produce a plausible answer that ignored the caller's
requirements, which is worse than an error.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------------ requests

class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImagePart(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["image_url"]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")   # tolerate unknown fields from new SDKs

    model: str
    messages: list[Message] = Field(min_length=1)

    # generation controls
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    logprobs: bool | None = None
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    logit_bias: dict[str, float] | None = None
    user: str | None = None
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    # Kimi-K3 thinking effort, as on Moonshot's API. None means the model default (max).
    reasoning_effort: Literal["low", "high", "max"] | None = None

    # features this backend cannot honour -- accepted, then rejected by name
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any | None = None
    functions: list[dict[str, Any]] | None = None
    function_call: Any | None = None
    response_format: ResponseFormat | None = None

    # explicit, namespaced opt-in for backend-native controls
    extra_body: dict[str, Any] | None = None

    def resolved_max_tokens(self, default: int) -> int:
        return self.max_completion_tokens or self.max_tokens or default

    def stop_list(self) -> list[str]:
        if self.stop is None:
            return []
        return [self.stop] if isinstance(self.stop, str) else list(self.stop)


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: str | list[str] | list[int] | list[list[int]]
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = None
    user: str | None = None


class CompletionsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str | list[str] | list[int] | list[list[int]]
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: str | list[str] | None = None
    seed: int | None = None
    n: int | None = Field(default=None, ge=1)
    stream: bool | None = None
    stream_options: StreamOptions | None = None
    logprobs: int | None = None
    user: str | None = None
