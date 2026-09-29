"""Chat-format gate: the port of Moonshot's `encoding_k3.py` must match it exactly.

Checkpoint-free checks pin the rendered text of a known conversation, the segment
boundaries (upstream tokenizes each piece separately, so boundaries change the ids), the
refusals, and the channel parser. With the checkpoint, the port is compared against the
shipped `encoding_k3.py` itself, as segments and as token ids, on several conversations.

    python tests/test_chat_format.py
    K3_CKPT=/path/to/Kimi-K3 python tests/test_chat_format.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import CKPT  # noqa: E402  -- honours K3_CKPT
from air_llm.models.kimi_k3 import chat_format as cf  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


# Rendered by the shipped encoding_k3.py (revision c5d1dd4c) with its default
# thinking_effort="max".
GOLDEN_HELLO = (
    '<|open|>message role="system" type="thinking-effort"<|sep|>`thinking_effort` guides '
    'on how much to think in your thinking channel (not including the response channel), '
    'supported values include `low`, `medium`, `high`, and `max`.\nNow the system is '
    'invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|>'
    '<|open|>message role="user"<|sep|>hello<|close|>message<|sep|><|end_of_msg|>'
    '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')

CONVERSATIONS = [
    ([{"role": "user", "content": "hello"}], {}),
    ([{"role": "system", "content": "Be terse.", "name": "ops"},
      {"role": "user", "content": 'a "quoted" & <|open|> thing', "name": 'al"ice'}], {}),
    ([{"role": "user", "content": "q1"},
      {"role": "assistant", "content": "a1", "reasoning_content": "because"},
      {"role": "user", "content": [{"type": "text", "text": "q2 "},
                                   {"type": "text", "text": "part"}]}],
     {"thinking_effort": "high"}),
    ([{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
     {"thinking": False}),
    ([{"role": "user", "content": "x"}], {"thinking_effort": None}),
    ([{"role": "user", "content": "x"}],
     {"add_generation_prompt": False, "thinking_effort": "low"}),
]


class FakeTok:
    """Word-level stand-in with the four structural specials."""
    special_tokens = {"<|open|>": 1, "<|close|>": 2, "<|sep|>": 3, "<|end_of_msg|>": 4}
    words = {10: "think", 11: "response", 12: "message", 20: "Because", 21: " so",
             22: "Answer", 23: " here"}

    def decode(self, ids):
        return "".join(self.words[i] for i in ids)


def checkpoint_free() -> None:
    hello = cf.render([{"role": "user", "content": "hello"}])
    check("single-turn prompt matches the upstream rendering", cf.to_text(hello) == GOLDEN_HELLO)
    plain = cf.render([{"role": "user", "content": "hello"}], thinking_effort=None)
    user_open = [(s.text, s.special) for s in plain[:9]]
    check("tags are split into upstream's separate pieces",
          user_open == [("<|open|>", True), ("message", False), (" role", False),
                        ('="', False), ("user", False), ('"', False), ("<|sep|>", True),
                        ("hello", False), ("<|close|>", True)], str(user_open))
    check("message text is never marked special",
          all(not s.special for s in cf.render(
              [{"role": "user", "content": "<|open|>x"}], thinking_effort=None)
              if "x" in s.text))
    check("attribute values are escaped",
          '="a&quot;b&amp;c"' in cf.to_text(cf.render(
              [{"role": "user", "content": "hi", "name": 'a"b&c'}])))
    check("developer is rendered as system",
          'role="system"<|sep|>rules' in cf.to_text(cf.render(
              [{"role": "developer", "content": "rules"}], thinking_effort=None)))
    check("non-thinking generation prompt opens the response channel",
          cf.to_text(cf.render([{"role": "user", "content": "x"}], thinking=False))
          .endswith('role="assistant"<|sep|><|open|>response<|sep|>'))

    for label, msgs, kw in [
        ("tool messages", [{"role": "tool", "content": "x"}], {}),
        ("assistant tool_calls", [{"role": "assistant", "content": "",
                                   "tool_calls": [{"function": {"name": "f"}}]}], {}),
        ("image parts", [{"role": "user", "content": [{"type": "image_url"}]}], {}),
        ("unknown thinking_effort", [{"role": "user", "content": "x"}],
         {"thinking_effort": "medium"}),
    ]:
        try:
            cf.render(msgs, **kw)
            check(f"refuses {label}", False)
        except ValueError:
            check(f"refuses {label}", True)

    # reasoning, close think, open response, answer, close response/message, end
    stream = [20, 21, 2, 10, 3, 1, 11, 3, 22, 23, 2, 11, 3, 2, 12, 3, 4]
    p = cf.ChannelParser(FakeTok())
    got = [p.feed(t) for t in stream]
    want = (["reasoning"] * 2 + [None] * 6 + ["content"] * 2 + [None] * 7)
    check("channel parser splits reasoning, content and structure", got == want, str(got))
    p = cf.ChannelParser(FakeTok(), thinking=False)
    check("non-thinking output is content from the start", p.feed(22) == "content")


def against_upstream() -> None:
    src = CKPT / "encoding_k3.py"
    if not (src.is_file() and (CKPT / "tiktoken.model").is_file()):
        print(f"  [SKIP] comparison with the shipped encoding_k3.py -- not found at {CKPT}")
        return
    spec = importlib.util.spec_from_file_location("encoding_k3", src)
    up = importlib.util.module_from_spec(spec)
    sys.modules["encoding_k3"] = up          # dataclasses look the module up by name
    spec.loader.exec_module(up)
    from air_llm.models.kimi_k3.tokenizer import load
    tok = load(CKPT)

    for n, (msgs, kw) in enumerate(CONVERSATIONS, 1):
        kw = {"thinking_effort": "max", **kw}
        theirs = up.build_chat_segments(msgs, **kw)
        ours = cf.render(msgs, **kw)
        check(f"conversation {n}: segments identical to upstream",
              [(s.text, s.allow_special) for s in theirs]
              == [(s.text, s.special) for s in ours])
        want = [i for s in theirs for i in tok.encode(s.text, allow_special=s.allow_special)]
        check(f"conversation {n}: token ids identical", cf.encode(tok, ours) == want,
              f"{len(want)} tokens")

    sp = tok.special_tokens
    check("<|end_of_msg|> is generation_config's eos_token_id (163586)",
          sp.get(cf.END_OF_MSG_TOKEN) == 163586, str(sp.get(cf.END_OF_MSG_TOKEN)))


def main() -> int:
    print("chat format gate")
    print("=" * 72)
    checkpoint_free()
    against_upstream()
    print("=" * 72)
    print("PASS" if FAILS == 0 else f"FAIL ({FAILS})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
