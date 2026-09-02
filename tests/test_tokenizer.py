"""Tokenizer shim gate.

The shim rebuilds the tiktoken Encoding from the checkpoint's own `tiktoken.model`,
`pat_str`, and `added_tokens_decoder`, avoiding trust_remote_code and any coupling to a
transformers version. What must be true:

  * ids are IDENTICAL to the shipped tokenizer (checked whenever it can be loaded)
  * encode/decode round-trips exactly, including CJK, emoji, whitespace runs
  * the 256-slot reserved block starts exactly at len(mergeable_ranks)

That last one is the trap: get the offset wrong and ordinary text still tokenizes
perfectly while every special token silently shifts.

    python tests/test_tokenizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import CKPT  # noqa: E402  -- honours K3_CKPT
from air_llm.models.kimi_k3.tokenizer import load  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main():
    if not (CKPT / "tiktoken.model").is_file():
        print("checkpoint not reachable; skipping")
        return 0

    tok = load(CKPT)
    print(f"tokenizer gate: n_vocab {tok.n_vocab}")
    print("=" * 72)

    for label, ok, detail in tok.self_check():
        check(label, ok, detail)

    ok, msg = tok.verify_against_remote_code()
    if "unavailable" in msg:
        print(f"  [SKIP] parity vs shipped tokenizer -- {msg}")
    else:
        check("ids identical to the shipped tokenizer", ok, msg)

    # config agreement
    import json
    cfg = json.loads((CKPT / "config.json").read_text())
    vocab = cfg["text_config"]["vocab_size"]
    check("n_vocab matches config.json vocab_size", tok.n_vocab == vocab,
          f"{tok.n_vocab} vs {vocab}")

    sp = tok.special_tokens
    for name, want in (("[BOS]", cfg["bos_token_id"]), ("[EOS]", 163585)):
        if name == "[BOS]":
            check(f"{name} id matches config bos_token_id", sp.get(name) == want,
                  f"{sp.get(name)} vs {want}")

    check("eos_token_id from config is a known special token",
          cfg["eos_token_id"] in set(sp.values()), f"{cfg['eos_token_id']}")
    check("pad_token_id from config is a known special token",
          cfg["pad_token_id"] in set(sp.values()), f"{cfg['pad_token_id']}")

    # special tokens must not appear from ordinary text unless explicitly allowed
    hostile = "[BOS] <|open|> not really special"
    ids = tok.encode(hostile)
    check("special-token strings in plain text are NOT encoded as specials",
          all(i < tok.n_vocab - 256 for i in ids), f"{len(ids)} tokens, max id {max(ids)}")
    ids_allowed = tok.encode(hostile, allow_special=True)
    check("allow_special=True does encode them as specials",
          any(i >= tok.n_vocab - 256 for i in ids_allowed))

    # determinism
    t = "The capital of France is"
    check("encoding is deterministic", tok.encode(t) == tok.encode(t),
          f"{tok.encode(t)}")

    print("=" * 72)
    print(f"TOKENIZER: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
