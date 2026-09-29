"""Kimi-K3 chat format (XTML) for text-only conversations.

Ported from Moonshot's `encoding_k3.py` (Hugging Face `moonshotai/Kimi-K3`, revision
c5d1dd4c, sha256 b9cb7ae1...5333); see NOTICE and the Kimi K3 license in
licenses/LICENSE.kimi-k3. Only the subset this backend can honour is ported: system,
user and assistant messages, the thinking channel and `thinking_effort`. Tools, images
and `response_format` are refused rather than approximated.

A prompt is a list of segments, and each segment is tokenized on its own. That matters:
upstream encodes `<|open|>`, `message`, ` role`, `="`, `user`, `"` and `<|sep|>` as
separate pieces, so tokenizing the joined string would merge BPE tokens across piece
boundaries and produce different ids. Structural markers are special tokens; message
text is always ordinary text, so a user who types `<|open|>` cannot forge structure.

The generation prompt ends inside the assistant's thinking channel
(`<|open|>message role="assistant"<|sep|><|open|>think<|sep|>`), so the model first
writes its reasoning, then closes `think`, opens `response`, writes the answer and ends
with `<|end_of_msg|>`. `ChannelParser` follows those markers token by token.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OPEN_TOKEN = "<|open|>"
CLOSE_TOKEN = "<|close|>"
SEP_TOKEN = "<|sep|>"
END_OF_MSG_TOKEN = "<|end_of_msg|>"

THINKING_EFFORTS = ("low", "high", "max")
# Upstream `apply_chat_template` sets this when the caller gives none.
DEFAULT_THINKING_EFFORT = "max"


@dataclass(frozen=True)
class Segment:
    text: str
    special: bool = False


def _get(message: Any, key: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(key, default)
    value = getattr(message, key, None)
    if value is None:
        extra = getattr(message, "model_extra", None) or {}
        value = extra.get(key, default)
    return default if value is None else value


def _text(value: Any) -> list[Segment]:
    value = str(value)
    return [Segment(value)] if value else []


def _control(value: str) -> list[Segment]:
    return [Segment(value, special=True)]


def _attr(key: str, value: Any) -> list[Segment]:
    escaped = str(value).replace("&", "&amp;").replace('"', "&quot;")
    return _text(f" {key}") + _text('="') + _text(escaped) + _text('"')


def _open_tag(tag: str, attrs: list[tuple[str, Any]] = ()) -> list[Segment]:
    out = _control(OPEN_TOKEN) + _text(tag)
    for key, value in attrs:
        out += _attr(key, value)
    return out + _control(SEP_TOKEN)


def _close_tag(tag: str) -> list[Segment]:
    return _control(CLOSE_TOKEN) + _text(tag) + _control(SEP_TOKEN)


def _content(content: Any) -> list[Segment]:
    if content is None:
        return []
    if isinstance(content, str):
        return _text(content)
    out: list[Segment] = []
    for part in content:
        kind = part.get("type")
        if kind != "text":
            raise ValueError(f"content part type {kind!r} is not supported (text only)")
        out += _text(part.get("text", ""))
    return out


def _message(role: str, name: Any, body: list[Segment]) -> list[Segment]:
    attrs = [("role", role)]
    if name:
        attrs.append(("name", name))
    return _open_tag("message", attrs) + body + _close_tag("message") + _control(END_OF_MSG_TOKEN)


def _internal_system(kind: str, body: str) -> list[Segment]:
    return (_open_tag("message", [("role", "system"), ("type", kind)])
            + _text(body.strip()) + _close_tag("message") + _control(END_OF_MSG_TOKEN))


def render(messages: list[Any], *, thinking: bool = True,
           thinking_effort: str | None = DEFAULT_THINKING_EFFORT,
           add_generation_prompt: bool = True) -> list[Segment]:
    """Segments for `messages`, matching upstream `build_chat_segments` for text chat.

    Messages may be dicts or objects with `role`, `content`, `name` and, for assistant
    turns, `reasoning_content` (or `reasoning`). `developer` is treated as `system`.
    """
    out: list[Segment] = []
    if thinking and thinking_effort is not None:
        if thinking_effort not in THINKING_EFFORTS:
            raise ValueError(f"thinking_effort must be one of {THINKING_EFFORTS}, "
                             f"got {thinking_effort!r}")
        out += _internal_system(
            "thinking-effort",
            "`thinking_effort` guides on how much to think in your "
            "thinking channel (not including the response channel), "
            "supported values include `low`, `medium`, `high`, and `max`.\n"
            f"Now the system is invoked with `thinking_effort={thinking_effort}`.")

    for m in messages:
        role = _get(m, "role")
        if role == "developer":
            role = "system"
        if role in ("tool", "function") or _get(m, "tool_calls") or _get(m, "tools"):
            raise ValueError("tool messages and tool calls are not supported")
        name = _get(m, "name")
        if role in ("user", "system"):
            out += _message(role, name, _content(_get(m, "content")))
        elif role == "assistant":
            body: list[Segment] = []
            if thinking:
                reasoning = _get(m, "reasoning_content") or _get(m, "reasoning")
                body += _open_tag("think")
                if reasoning is not None and str(reasoning).strip():
                    body += _text(reasoning)
                body += _close_tag("think")
            body += _open_tag("response") + _content(_get(m, "content")) + _close_tag("response")
            out += _message("assistant", name, body)
        else:
            raise ValueError(f"unsupported role {role!r}")

    if add_generation_prompt:
        out += _open_tag("message", [("role", "assistant")])
        out += _open_tag("think" if thinking else "response")
    return out


def to_text(segments: list[Segment]) -> str:
    return "".join(s.text for s in segments)


def encode(tokenizer, segments: list[Segment]) -> list[int]:
    """Token ids, one tokenizer call per segment (see the module docstring)."""
    ids: list[int] = []
    for s in segments:
        ids += tokenizer.encode(s.text, allow_special=s.special)
    return ids


def encode_chat(tokenizer, messages: list[Any], **kwargs) -> list[int]:
    return encode(tokenizer, render(messages, **kwargs))


class ChannelParser:
    """Assigns each generated token to "reasoning", "content" or None (structure).

    Feed token ids in order. Tag names arrive as ordinary text between an open/close
    marker and `<|sep|>`, so tokens are buffered from the marker to the separator and
    all reported as structure. Text outside any recognised channel is treated as
    content, so an unexpected shape degrades to visible text rather than lost text.
    """

    def __init__(self, tokenizer, *, thinking: bool = True):
        sp = getattr(tokenizer, "special_tokens", {}) or {}
        self.open_id = sp.get(OPEN_TOKEN)
        self.close_id = sp.get(CLOSE_TOKEN)
        self.sep_id = sp.get(SEP_TOKEN)
        self.end_ids = {i for n, i in sp.items() if n in (END_OF_MSG_TOKEN, "[EOS]")}
        self.tokenizer = tokenizer
        # A tokenizer without the structural tokens cannot have produced this format
        # (the server's mock engine), so everything it emits is content.
        formatted = thinking and None not in (self.open_id, self.close_id, self.sep_id)
        self.channel = "reasoning" if formatted else "content"
        self._tag: str | None = None      # "open" / "close" while inside a marker
        self._tag_ids: list[int] = []

    def feed(self, token_id: int) -> str | None:
        if token_id in self.end_ids:
            return None
        if token_id in (self.open_id, self.close_id):
            self._tag = "open" if token_id == self.open_id else "close"
            self._tag_ids = []
            return None
        if self._tag is not None:
            if token_id == self.sep_id:
                name = self.tokenizer.decode(self._tag_ids).strip().split(" ", 1)[0]
                if self._tag == "open" and name == "think":
                    self.channel = "reasoning"
                elif self._tag == "open" and name == "response":
                    self.channel = "content"
                elif self._tag == "close" and name in ("think", "response"):
                    self.channel = None
                self._tag = None
            else:
                self._tag_ids.append(token_id)
            return None
        return self.channel or "content"
