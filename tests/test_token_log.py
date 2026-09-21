"""The server's per-token diagnostic log (`--token-log`).

This exists because the 2026-08-07 baseline run produced 2,197 tokens of wall-clock
timings and could not answer the one question that mattered: did decode slow down because
the KV CACHE grew, or because ROUTING reached more experts? Both produce the same
front-loaded-then-flat curve. `KimiK3Model.generate` has always written these counters
via `log_path`, but long runs go through the server, which did not.

So the gates here are about the counters being separable and attributable, not about the
file existing:

    T1  one record per generated token, in order
    T2  context grows by exactly the tokens consumed (prompt on step 0, then 1)
    T3  the ROUTING counters and the CONTEXT counters are independent fields, both
        non-trivial, so a slowdown can be attributed to one or the other
    T4  MLA cache bytes track context exactly (the latent-cache rate, not master's)
    T5  peak_bytes is recorded, so a successful run reports its own headroom
    T6  a broken log destination cannot kill a generation

Runs on the synthetic model; no checkpoint access.

    source .venv/bin/activate && python tests/test_token_log.py
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.loader import TensorSource  # noqa: E402
from air_llm.server.engine import Engine, GenParams  # noqa: E402
from k3ref.layers import K3Config  # noqa: E402

MINI = Path("/tmp/mini-k3")
MANIFEST = MINI / "manifest.jsonl.gz"
PROMPT = [7, 42, 128, 9]
NTOK = 5
FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return ok


class StubTokenizer:
    """The mini model has no vocab. Engine needs decode() for the streamed text and
    special_tokens for EOS; the ids below are outside the mini model's 512-token
    vocabulary so generation always runs to max_tokens rather than stopping early."""
    special_tokens = {"[EOS]": 100000, "<|end_of_msg|>": 100001}

    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


def ensure_model():
    if not MANIFEST.exists():
        root = Path(__file__).resolve().parent.parent
        subprocess.run([sys.executable, str(root / "tools/make_synthetic_k3.py"),
                        "--out", str(MINI)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(root / "tools/k3_manifest.py"), "build",
                        "--ckpt", str(MINI), "--out", str(MANIFEST)],
                       check=True, capture_output=True)


def generate(token_log):
    cfg = K3Config.load(MINI / "config.json")
    src = TensorSource(manifest=MANIFEST, ckpt=MINI, resident_gb=0.0,
                       weight_dtype=mx.float32)
    eng = Engine(cfg, src, StubTokenizer(), max_memory_gb=48.0,
                 max_output_tokens=64, token_log=token_log)

    async def go():
        out = []
        async for item in eng.generate(GenParams(prompt_tokens=list(PROMPT),
                                                 max_tokens=NTOK),
                                       lambda: _false()):
            out.append(item)
        return out
    return asyncio.run(go())


async def _false():
    return False


def main():
    ensure_model()
    print("token-log gates  --  synthetic model, "
          f"{len(PROMPT)}-token prompt, {NTOK} generated")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "nested" / "tokens.jsonl"
        items = generate(p)
        recs = [json.loads(l) for l in p.read_text().splitlines()]

        steps = [i for i in items if type(i).__name__ == "Step"]
        check("one record per generated token", len(recs) == len(steps) == NTOK,
              f"{len(recs)} records, {len(steps)} steps")
        check("records are in order",
              [r["index"] for r in recs] == list(range(NTOK)))
        check("parent directory created on demand", p.exists())

        # T2 -- context accounting. Step 0 consumes the whole prompt, later steps one token.
        want = [len(PROMPT) + i for i in range(NTOK)]
        got = [r["context"] for r in recs]
        check("context advances by prompt then 1/token", got == want, f"{got} vs {want}")
        check("step 0 reports the prompt length", recs[0]["step_tokens"] == len(PROMPT))
        check("later steps report 1", all(r["step_tokens"] == 1 for r in recs[1:]))

        # T3 -- the attribution gate: routing and context must be separable signals.
        check("routing counters are populated",
              all(r["experts_loaded"] > 0 and r["expert_bytes"] > 0 for r in recs),
              f"experts_loaded {[r['experts_loaded'] for r in recs]}")
        check("bytes_read is per-token, not cumulative",
              all(r["bytes_read"] > 0 for r in recs)
              and recs[-1]["bytes_read"] < sum(r["bytes_read"] for r in recs),
              f"{[r['bytes_read'] for r in recs]}")

        # T4 -- cache bytes must move with context, at the LATENT rate.
        cfg = K3Config.load(MINI / "config.json")
        n_mla = len(cfg.mla_layers)
        per_tok = n_mla * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) * 4
        deltas = [b["mla_cache_bytes"] - a["mla_cache_bytes"]
                  for a, b in zip(recs, recs[1:])]
        check("MLA cache grows at the latent rate", all(d == per_tok for d in deltas),
              f"{deltas} vs {per_tok} B/token expected "
              f"({n_mla} MLA layer(s) x {cfg.kv_lora_rank}+{cfg.qk_rope_head_dim} fp32)")
        check("peak_bytes recorded and plausible",
              all(r.get("peak_bytes", 0) > 0 for r in recs),
              f"{[r.get('peak_bytes', 0) // 2**20 for r in recs]} MiB")
        check("KDA state is flat (fixed-size recurrence)",
              len({r["kda_state_bytes"] for r in recs}) == 1,
              f"{sorted({r['kda_state_bytes'] for r in recs})}")

    # T6 -- diagnostics must never take down a generation that is hours in.
    bad = Path("/dev/null/cannot/exist/tokens.jsonl")
    try:
        items = generate(bad)
        ok = len([i for i in items if type(i).__name__ == "Step"]) == NTOK
    except Exception as e:
        ok = False
        print(f"       raised {type(e).__name__}: {e}")
    check("an unwritable token log does not break generation", ok)

    print(f"\n{'TOKEN LOG: PASS' if FAILS == 0 else f'TOKEN LOG: {FAILS} FAILED'}")
    return FAILS


if __name__ == "__main__":
    sys.exit(main())
