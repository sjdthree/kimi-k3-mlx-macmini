"""Bearer-token auth with per-key limits.

Keys come from `AIR_LLM_API_KEYS` (comma-separated) or a JSON file via
`AIR_LLM_API_KEYS_FILE`, allowing per-key quotas:

    {"sk-alice": {"rpm": 10, "max_output_tokens": 256, "concurrency": 1}}

Keys are compared with `secrets.compare_digest` and never logged; logs and metrics carry
a short non-reversible key id instead, so a tenant is attributable without the secret
appearing anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Request

from .errors import authentication_error, permission_error, rate_limit


@dataclass
class KeyPolicy:
    rpm: int = 60
    max_output_tokens: int = 4096
    concurrency: int = 1
    daily_tokens: int | None = None


@dataclass
class KeyState:
    calls: deque = field(default_factory=deque)
    in_flight: int = 0
    tokens_today: int = 0
    day: int = field(default_factory=lambda: int(time.time() // 86400))


def key_id(key: str) -> str:
    """Stable, non-reversible identifier for logs and metric labels."""
    return "key_" + hashlib.sha256(key.encode()).hexdigest()[:12]


class Auth:
    def __init__(self):
        self.keys: dict[str, KeyPolicy] = {}
        self.state: dict[str, KeyState] = defaultdict(KeyState)
        self.disabled = False
        self._load()

    def _load(self) -> None:
        path = os.environ.get("AIR_LLM_API_KEYS_FILE")
        if path and os.path.isfile(path):
            for k, v in json.loads(open(path).read()).items():
                self.keys[k] = KeyPolicy(**v) if isinstance(v, dict) else KeyPolicy()
        raw = os.environ.get("AIR_LLM_API_KEYS", "")
        for k in (s.strip() for s in raw.split(",")):
            if k:
                self.keys.setdefault(k, KeyPolicy())
        if not self.keys:
            # Explicit, loud opt-out for local development.
            self.disabled = os.environ.get("AIR_LLM_ALLOW_ANONYMOUS") == "1"

    # ------------------------------------------------------------------ checking
    def authenticate(self, request: Request) -> tuple[str, KeyPolicy]:
        if self.disabled:
            return "anonymous", KeyPolicy()

        header = request.headers.get("authorization", "")
        if not header.lower().startswith("bearer "):
            raise authentication_error(
                "You didn't provide an API key. You need to provide your API key in an "
                "Authorization header using Bearer auth "
                "(i.e. Authorization: Bearer YOUR_KEY).")
        presented = header[7:].strip()

        for k, policy in self.keys.items():
            if secrets.compare_digest(k, presented):
                return key_id(k), policy
        raise authentication_error()

    def check_rate(self, kid: str, policy: KeyPolicy) -> None:
        st = self.state[kid]
        now = time.time()
        while st.calls and now - st.calls[0] > 60:
            st.calls.popleft()
        if len(st.calls) >= policy.rpm:
            raise rate_limit(
                f"Rate limit reached for this key: {policy.rpm} requests per minute. "
                f"Please retry shortly.")
        st.calls.append(now)

        today = int(now // 86400)
        if st.day != today:
            st.day, st.tokens_today = today, 0
        if policy.daily_tokens and st.tokens_today >= policy.daily_tokens:
            raise permission_error(
                f"Daily token quota of {policy.daily_tokens} exhausted for this key.")

    def enter(self, kid: str, policy: KeyPolicy) -> None:
        st = self.state[kid]
        if st.in_flight >= policy.concurrency:
            raise rate_limit(
                f"This key allows {policy.concurrency} concurrent request(s).",
                code="concurrency_limit_exceeded")
        st.in_flight += 1

    def leave(self, kid: str, tokens: int = 0) -> None:
        st = self.state[kid]
        st.in_flight = max(0, st.in_flight - 1)
        st.tokens_today += tokens
