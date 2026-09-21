"""Recompute the aggregates quoted in docs/evidence.md from the raw logs in this repo.

Reads only tracked files; needs no checkpoint, MLX or network. If a number here and a
number in the docs disagree, the logs win and the docs are wrong.

    python tools/summarize_evidence.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
SWEEP = ROOT / "docs" / "kv-cache" / "sweep"
RUNS = ROOT / "runs"
GIB = 2 ** 30


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def decode(recs: list[dict], first: int) -> list[dict]:
    """Single-token decode steps with index >= first (index 0 is prefill)."""
    return [r for r in recs if r["index"] >= first and r["step_tokens"] == 1]


def sweep() -> None:
    print("Resident-budget sweep  (docs/kv-cache/sweep/r*-tokens.jsonl)")
    print(f"  {'GiB':>4} {'n':>3} {'all-decode s/tok':>17} {'steady s/tok':>13} "
          f"{'steady read GiB/tok':>20} {'max MLX peak GiB':>17}")
    for gb in (8, 16, 24, 32, 40):
        recs = records(SWEEP / f"r{gb}-tokens.jsonl")
        all_dec, steady = decode(recs, 1), decode(recs, 2)
        peak = max((r.get("peak_bytes", 0) for r in recs), default=0) / GIB
        print(f"  {gb:>4} {len(all_dec):>3} {mean(r['elapsed_s'] for r in all_dec):>17.4f} "
              f"{mean(r['elapsed_s'] for r in steady):>13.4f} "
              f"{mean(r['bytes_read'] for r in steady) / GIB:>20.4f} {peak:>17.4f}")
    print("  bytes_read is logical bytes requested by the loader, including OS cache hits.")


def whole_run() -> None:
    print("\nWhole-run comparison  (runs/*/progress.json, includes prefill)")
    rates = {}
    for name in ("polymath", "final-validation"):
        p = json.loads((RUNS / name / "progress.json").read_text())
        rates[name] = p["elapsed_s"] / p["tokens_generated_total"]
        print(f"  {name:<17} {p['elapsed_s']:>10.1f} s / {p['tokens_generated_total']:>5} tokens"
              f" = {rates[name]:7.3f} s/token   TTFT {p['ttft_s']} s")
    print(f"  ratio {rates['polymath'] / rates['final-validation']:.3f}x  "
          "(different resident budgets and output lengths; not a cache-only A/B)")

    recs = records(RUNS / "final-validation" / "tokens.jsonl")
    dec = decode(recs, 2)
    print(f"  final-validation: {len(recs)} records, prompt {recs[0]['step_tokens']} tokens, "
          f"final context {recs[-1]['context']}, {len(dec)} records after index 1 "
          f"average {mean(r['elapsed_s'] for r in dec):.4f} s")


def long_context() -> None:
    recs = records(RUNS / "longctx-2k" / "tokens.jsonl")
    dec = decode(recs, 2)
    print("\nLong context  (runs/longctx-2k/tokens.jsonl)")
    print(f"  prefill {recs[0]['step_tokens']} tokens in {recs[0]['elapsed_s']:.0f} s; "
          f"{len(recs)} records; {len(dec)} steady decode records average "
          f"{mean(r['elapsed_s'] for r in dec):.3f} s")
    if not any("peak_bytes" in r for r in recs):
        print("  no peak-memory field in this log")


def mla_arithmetic() -> None:
    layers, heads, qk, v, latent, rope = 24, 96, 192, 128, 512, 64
    expanded = layers * heads * (qk + v) * 4
    compact = layers * (latent + rope) * 4
    print("\nMLA cache payload  (fp32, per context token, excludes KDA state and slack)")
    print(f"  expanded {expanded:,} B   latent {compact:,} B   ratio {expanded / compact:.3f}x")


if __name__ == "__main__":
    sweep()
    whole_run()
    long_context()
    mla_arithmetic()
