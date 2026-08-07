"""Kimi-K3 tokenizer shim: TikToken directly, no transformers version dependency.

K3 ships `tokenization_kimi.py`, a `PreTrainedTokenizer` subclass targeting transformers
4.56. It imports `transformers.convert_slow_tokenizer.bytes_to_unicode` and
`transformers.tokenization_utils.PreTrainedTokenizer`, both of which moved in 5.x, so
`AutoTokenizer.from_pretrained(trust_remote_code=True)` fails in this environment.

None of that machinery is needed to turn text into ids. The actual tokenizer is a plain
`tiktoken.Encoding` built from three things the checkpoint provides directly:

  * `tiktoken.model`      -- the mergeable BPE ranks
  * `pat_str`             -- the split regex, transcribed below from tokenization_kimi.py
  * `tokenizer_config.json` -- `added_tokens_decoder`, mapping special ids to strings

So this shim reconstructs the Encoding itself. It is deliberately a *reimplementation of
the loading*, not of the tokenization: the BPE ranks and regex are the shipped ones, so
ids match what the remote code would produce. `verify_against_remote_code()` checks that
claim wherever a compatible transformers is available.

One subtlety worth stating: the reserved-token block starts at `len(mergeable_ranks)`,
and every unnamed slot gets `<|reserved_token_{i}|>`. Getting that offset wrong shifts
every special id without affecting ordinary text, which would look fine on
"The capital of France is" and break every chat-formatted prompt.
"""
from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

NUM_RESERVED_SPECIAL_TOKENS = 256

# Transcribed verbatim from tokenization_kimi.py:54-63.
PAT_STR = "|".join([
    r"""[\p{Han}]+""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
    r"""\p{N}{1,3}""",
    r""" ?[^\s\p{L}\p{N}]+[\r\n]*""",
    r"""\s*[\r\n]+""",
    r"""\s+(?!\S)""",
    r"""\s+""",
])


class K3Tokenizer:
    def __init__(self, model_dir: str | Path):
        self.dir = Path(model_dir)
        self.vocab_file = self.dir / "tiktoken.model"
        if not self.vocab_file.is_file():
            raise FileNotFoundError(f"missing {self.vocab_file}")
        cfg_path = self.dir / "tokenizer_config.json"
        self.config = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}

    @cached_property
    def encoding(self):
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        ranks = load_tiktoken_bpe(str(self.vocab_file))
        base = len(ranks)

        named = {int(i): v["content"]
                 for i, v in self.config.get("added_tokens_decoder", {}).items()}
        special = {named.get(i, f"<|reserved_token_{i}|>"): i
                   for i in range(base, base + NUM_RESERVED_SPECIAL_TOKENS)}

        return tiktoken.Encoding(
            name=self.vocab_file.name,
            pat_str=PAT_STR,
            mergeable_ranks=ranks,
            special_tokens=special,
        )

    # ------------------------------------------------------------------ interface
    @property
    def n_vocab(self) -> int:
        return self.encoding.n_vocab

    @property
    def special_tokens(self) -> dict[str, int]:
        return dict(self.encoding._special_tokens)

    def encode(self, text: str, allow_special: bool = False) -> list[int]:
        if allow_special:
            return self.encoding.encode(text, allowed_special="all")
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, ids) -> str:
        return self.encoding.decode(list(ids))

    def token_id(self, token: str) -> int:
        return self.special_tokens[token]

    # ------------------------------------------------------------------- checking
    def verify_against_remote_code(self) -> tuple[bool, str]:
        """Compare against the shipped tokenizer where transformers allows loading it."""
        probes = ["The capital of France is", "hello world", "  spaced   out\n\n",
                  "1234567890", "def f(x):\n    return x * 2\n", "北京是中国的首都",
                  "emoji: \U0001F600 \U0001F680", ""]
        try:
            from transformers import AutoTokenizer
            ref = AutoTokenizer.from_pretrained(str(self.dir), trust_remote_code=True)
        except Exception as e:
            return False, f"remote tokenizer unavailable ({type(e).__name__}: {e})"
        for t in probes:
            a = self.encode(t)
            b = ref(t, add_special_tokens=False)["input_ids"]
            if a != b:
                return False, f"mismatch on {t!r}: {a} vs {b}"
        return True, f"matches the shipped tokenizer on {len(probes)} probes"

    def self_check(self) -> list[tuple[str, bool, str]]:
        """Internal consistency checks that need no reference implementation."""
        out = []
        probes = ["The capital of France is", "hello world", "  spaced   out\n\n",
                  "1234567890", "def f(x):\n    return x * 2\n", "北京是中国的首都",
                  "emoji: \U0001F600 \U0001F680", "a" * 200, ""]
        bad = [t for t in probes if self.decode(self.encode(t)) != t]
        out.append(("encode/decode round-trips exactly", not bad,
                    f"{len(probes)-len(bad)}/{len(probes)} probes"))

        cfg_vocab = self.config.get("vocab_size")
        out.append(("special token block starts at len(mergeable_ranks)",
                    min(self.special_tokens.values()) ==
                    self.n_vocab - NUM_RESERVED_SPECIAL_TOKENS,
                    f"first special id {min(self.special_tokens.values())}"))

        named = {int(i): v["content"]
                 for i, v in self.config.get("added_tokens_decoder", {}).items()}
        wrong = [(i, n) for i, n in named.items()
                 if self.special_tokens.get(n) != i]
        out.append(("every named special token lands on its configured id",
                    not wrong, f"{len(named)} named, {len(wrong)} misplaced"))
        if cfg_vocab:
            out.append(("n_vocab matches tokenizer_config", self.n_vocab == cfg_vocab,
                        f"{self.n_vocab} vs {cfg_vocab}"))
        return out


def load(model_dir: str | Path) -> K3Tokenizer:
    return K3Tokenizer(model_dir)
