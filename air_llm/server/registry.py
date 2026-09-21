"""Local deployment aliases. IDs do not attest checkpoint identity; record hashes separately."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

OWNER = "air-llm"


@dataclass(frozen=True)
class Capabilities:
    chat: bool = True
    streaming: bool = True
    tools: bool = False
    json_object: bool = False
    json_schema: bool = False
    vision: bool = False
    embeddings: bool = False
    logprobs: bool = True
    sampling: bool = True
    context_length: int = 8192
    max_output_tokens: int = 4096
    # Deliberately part of the advertised contract: clients and humans both need it.
    approx_seconds_per_token: float = 34.0
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "notes"}
        d["notes"] = list(self.notes)
        return d


@dataclass(frozen=True)
class ServedModel:
    id: str
    capabilities: Capabilities
    created: int = field(default_factory=lambda: int(time.time()))
    aliases: tuple[str, ...] = ()
    description: str = ""

    def as_openai(self, model_id: str | None = None) -> dict:
        """OpenAI `model` object. Extra keys live under a namespaced block so the
        standard shape is untouched for clients that validate it strictly."""
        return {
            "id": model_id or self.id,
            "object": "model",
            "created": self.created,
            "owned_by": OWNER,
            "air_llm": {
                "deployment_id": self.id,
                "aliases": list(self.aliases),
                "description": self.description,
                "capabilities": self.capabilities.as_dict(),
            },
        }


K3 = ServedModel(
    id="kimi-k3-local",
    aliases=("chat", "reasoning", "kimi-k3"),
    description="Kimi-K3, 2.8T-parameter MoE (896 experts, top-16), 93 layers, "
                "streamed from disk on Apple Silicon via MLX.",
    capabilities=Capabilities(
        chat=True,
        streaming=True,
        # No constrained decoding exists in this backend. Advertising tools or strict
        # JSON here would mean prompting and hoping, which is exactly the failure mode
        # the API contract is supposed to prevent.
        tools=False,
        json_object=False,
        json_schema=False,
        # The checkpoint ships a MoonViT tower, but this implementation is text-only.
        vision=False,
        embeddings=False,
        logprobs=True,
        sampling=True,
        context_length=8192,
        max_output_tokens=4096,
        approx_seconds_per_token=34.0,
        notes=(
            "Historical best decode ~32.4 s/token. Prefill can exceed 25 minutes. Set generous "
            "client timeouts (>= 900 s) and prefer stream=true.",
            "Executes one generation at a time; concurrent requests queue and may 429.",
            "No constrained decoding: tools and response_format json_object/json_schema "
            "return unsupported_capability rather than an unenforced approximation.",
            "Text-only. The checkpoint's vision tower is not implemented.",
            "Only n=1 is supported. Limits are configuration, not validated context capacity.",
        ),
    ),
)

_MODELS: dict[str, ServedModel] = {K3.id: K3}
_ALIASES: dict[str, str] = {a: K3.id for a in K3.aliases}


def resolve(name: str) -> ServedModel:
    """Alias or physical id -> ServedModel. Raises APIError(404) if unknown."""
    from .errors import model_not_found
    physical = _ALIASES.get(name, name)
    m = _MODELS.get(physical)
    if m is None:
        raise model_not_found(name)
    return m


def list_models() -> list[dict]:
    """Every physical id and every alias, as separate selectable entries.

    Listing aliases as their own objects is what lets a UI's model picker show `chat`
    without the user needing to know the versioned id.
    """
    out = [m.as_openai() for m in _MODELS.values()]
    for alias, physical in _ALIASES.items():
        out.append(_MODELS[physical].as_openai(model_id=alias))
    return sorted(out, key=lambda d: d["id"])


def embedding_models() -> list[ServedModel]:
    return [m for m in _MODELS.values() if m.capabilities.embeddings]
