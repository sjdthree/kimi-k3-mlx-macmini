"""Resume gate for tools/run_chat_job.py, on the synthetic model (no checkpoint needed).

A multi-day run will be interrupted. What must hold: a run that stops and restarts
produces exactly the tokens of a run that never stopped, and a restart with settings
that change the sampled tokens is refused rather than silently mixed in.

The synthetic model has a 512-token vocabulary, so it gets a byte-level tokenizer:
256 single-byte ranks, then the 256 special slots with the chat-format tokens named.

    python tests/test_chat_job.py
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def sh(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *map(str, args)], capture_output=True, text=True)


def build(tmp: Path) -> Path:
    mini = tmp / "mini-k3"
    for r in (sh(ROOT / "tools/make_synthetic_k3.py", "--out", mini),
              sh(ROOT / "tools/k3_manifest.py", "build", "--ckpt", mini,
                 "--out", mini / "manifest.jsonl.gz")):
        if r.returncode:
            raise SystemExit(r.stdout + r.stderr)
    with (mini / "tiktoken.model").open("w") as f:
        for b in range(256):
            f.write(f"{base64.b64encode(bytes([b])).decode()} {b}\n")
    names = ["[BOS]", "[EOS]", "<|end_of_msg|>", "<|open|>", "<|close|>", "<|sep|>"]
    (mini / "tokenizer_config.json").write_text(json.dumps(
        {"added_tokens_decoder": {str(256 + i): {"content": n} for i, n in enumerate(names)}}))
    return mini


def main() -> int:
    print("chat job resume gate")
    print("=" * 72)
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        mini = build(tmp)
        msgs = tmp / "messages.json"
        msgs.write_text(json.dumps([{"role": "user", "content": "Summarise the seal log."}]))

        def job(out, max_tokens, seed=3):
            return sh(ROOT / "tools/run_chat_job.py", "--model", mini,
                      "--manifest", mini / "manifest.jsonl.gz", "--resident-gb", 0,
                      "--messages", msgs, "--out", tmp / out, "--save-every", 5,
                      "--seed", seed, "--max-tokens", max_tokens)

        def ids(out):
            return [json.loads(line)["id"] for line in (tmp / out / "tokens.jsonl").open()]

        r = job("straight", 30)
        check("uninterrupted run completes", r.returncode == 0, r.stderr[-300:])
        job("resumed", 12)
        r = job("resumed", 30)
        check("restarted run completes", r.returncode == 0, r.stderr[-300:])
        a, b = ids("straight"), ids("resumed")
        check("restarted run reproduces the uninterrupted tokens exactly",
              len(a) == len(b) and a == b, f"{len(a)} vs {len(b)} tokens")
        log = (tmp / "resumed" / "job.log").read_text()
        check("restart loaded the saved state instead of re-running the prefill",
              "state covers" in log and "prefill" not in log.split("state covers")[-1])
        check("both channels are written",
              all((tmp / "resumed" / n).is_file() for n in ("reasoning.md", "answer.md")))

        r = job("resumed", 40, seed=4)
        check("restart with a different seed is refused", r.returncode == 2,
              f"exit {r.returncode}")
        check("the refused restart left the tokens alone", ids("resumed") == b)
    print("=" * 72)
    print("PASS" if FAILS == 0 else f"FAIL ({FAILS})")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
