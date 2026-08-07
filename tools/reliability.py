"""M11 (spec section 16, "Reliability Test").

    5 prompts x 16 generated tokens x 3 independent process launches

Passing requirements:
  * 15 of 15 runs complete
  * deterministic token output
  * no memory leak exceeding 2% between equivalent token positions
  * no corrupted reads
  * no NaN or infinity in hidden states, recurrent states, routing weights, or logits

"Independent process launches" is load-bearing: an in-process loop would share MLX's
allocator, its compiled-kernel cache, and any accumulated state, which is exactly what a
leak test must not do. So each run is a real subprocess.

    python tools/reliability.py --model /Volumes/USB4_2TBB/Kimi-K3 --max-new-tokens 16
    python tools/reliability.py --model /tmp/mini-k3 \
        --manifest /tmp/mini-k3/manifest.jsonl.gz --max-new-tokens 6   # harness self-test
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Five real prompts, pre-tokenised so each subprocess need not load the tokenizer.
PROMPTS = [
    [1008, 10484, 318, 15383, 387],   # 'The capital of France is'
    [1166, 21490, 96445, 2145, 3118],   # 'def fibonacci(n):'
    [1008, 3499, 12524, 318, 20429, 138808, 554],   # 'The three laws of thermodynamics are'
    [13426, 7625, 261, 1169, 11, 306, 261, 3307, 17316],   # 'Once upon a time, in a small village'
    [48, 25, 5071, 387, 220, 1810, 4442, 220, 2466, 1820, 32, 25],   # 'Q: What is 17 times 23?\nA:'
]

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def run_one(a, prompt, launch, logdir: Path):
    log = logdir / f"p{PROMPTS.index(prompt)}_l{launch}.jsonl"
    cmd = [sys.executable, "-m", "air_llm.run",
           "--model", str(a.model),
           "--manifest", str(a.manifest),
           "--prompt-tokens", ",".join(map(str, prompt)),
           "--max-new-tokens", str(a.max_new_tokens),
           "--max-memory-gb", str(a.max_memory_gb),
           "--mode", a.mode,
           "--resident-gb", str(a.resident_gb),
           "--check-finite",
           "--validation-log", str(log)]
    if a.cache:
        cmd += ["--cache", str(a.cache)]
    if a.layers:
        cmd += ["--layers", str(a.layers)]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    dt = time.time() - t0

    recs = []
    if log.exists():
        recs = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    return {
        "prompt": prompt, "launch": launch, "rc": proc.returncode,
        "seconds": dt, "records": recs,
        "tokens": [r["token_id"] for r in recs],
        "stderr": proc.stderr[-2000:] if proc.returncode else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("manifest.jsonl.gz"))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--launches", type=int, default=3)
    ap.add_argument("--max-memory-gb", type=float, default=48.0)
    ap.add_argument("--mode", default="validation", choices=["validation", "runtime"])
    ap.add_argument("--resident-gb", type=float, default=0.0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--logdir", type=Path, default=Path("results/reliability"))
    a = ap.parse_args()

    a.logdir.mkdir(parents=True, exist_ok=True)
    n_expected = len(PROMPTS) * a.launches
    print(f"M11 reliability: {len(PROMPTS)} prompts x {a.max_new_tokens} tokens x "
          f"{a.launches} launches = {n_expected} runs")
    print(f"  model {a.model}  mode {a.mode}  resident {a.resident_gb} GiB")
    print("=" * 78)

    runs = []
    t0 = time.time()
    for li in range(a.launches):
        for p in PROMPTS:
            r = run_one(a, p, li, a.logdir)
            runs.append(r)
            status = "ok" if r["rc"] == 0 else f"rc={r['rc']}"
            print(f"  launch {li} prompt {PROMPTS.index(p)}: {status:>6}  "
                  f"{r['seconds']:7.1f}s  {len(r['tokens'])} tokens", flush=True)
            if r["rc"] != 0:
                print(f"      stderr tail: {r['stderr'][-400:]}")
    wall = time.time() - t0

    print("\n" + "=" * 78)

    # 1 -- completion
    done = [r for r in runs if r["rc"] == 0 and len(r["tokens"]) == a.max_new_tokens]
    check(f"{n_expected} of {n_expected} runs complete",
          len(done) == n_expected, f"{len(done)} completed")

    # 2 -- determinism across independent launches
    by_prompt = defaultdict(list)
    for r in done:
        by_prompt[tuple(r["prompt"])].append(r["tokens"])
    nondet = {p: v for p, v in by_prompt.items() if len({tuple(t) for t in v}) > 1}
    check("token output is deterministic across launches", not nondet,
          f"{len(nondet)} prompts differed" if nondet else
          f"{len(by_prompt)} prompts, {a.launches} launches each")
    for p, v in list(nondet.items())[:2]:
        print(f"      prompt {list(p)}: {[tuple(t) for t in v]}")

    # also: checksums must match, not just ids -- identical ids with differing
    # checksums would mean the logits drifted while the argmax happened to survive
    ck_bad = 0
    by_p_recs = defaultdict(list)
    for r in done:
        by_p_recs[tuple(r["prompt"])].append([x["logit_checksum"] for x in r["records"]])
    for p, v in by_p_recs.items():
        if len({tuple(x) for x in v}) > 1:
            ck_bad += 1
    check("logit checksums match across launches (no corrupted reads)", ck_bad == 0,
          f"{ck_bad} prompts with differing checksums")

    # 3 -- memory growth between equivalent token positions
    worst, worst_where = 0.0, ""
    peaks = defaultdict(list)        # (prompt, token_index) -> peaks across launches
    for r in done:
        for rec in r["records"]:
            peaks[(tuple(r["prompt"]), rec["token_index"])].append(
                rec["peak_unified_memory"])
    for key, vals in peaks.items():
        if len(vals) < 2 or min(vals) == 0:
            continue
        growth = (max(vals) - min(vals)) / min(vals)
        if growth > worst:
            worst, worst_where = growth, f"prompt {list(key[0])} token {key[1]}"
    check("no memory growth >2% between equivalent token positions (spec section 16)",
          worst <= 0.02, f"worst {worst*100:.2f}% at {worst_where}")

    # within a run: later tokens must not creep upward either
    creep, creep_where = 0.0, ""
    for r in done:
        pk = [x["peak_unified_memory"] for x in r["records"]]
        if len(pk) > 2 and min(pk[1:]) > 0:
            g = (pk[-1] - pk[1]) / pk[1]
            if g > creep:
                creep, creep_where = g, f"prompt {r['prompt']} launch {r['launch']}"
    check("no intra-run memory creep >2% from token 1 to the last token",
          creep <= 0.02, f"worst {creep*100:+.2f}% at {creep_where}")

    # 4 -- no NaN/inf anywhere
    nf = [(r["prompt"], x["token_index"], x.get("nonfinite_where", ""))
          for r in done for x in r["records"] if x.get("nonfinite", 0)]
    check("no NaN or infinity in any run", not nf,
          f"{len(nf)} occurrences" + (f", first at {nf[0]}" if nf else ""))

    # 5 -- read volume identical for the same prompt across launches
    rb = defaultdict(set)
    for r in done:
        rb[tuple(r["prompt"])].add(tuple(x["bytes_read"] for x in r["records"]))
    var = [p for p, v in rb.items() if len(v) > 1]
    check("bytes read per token is identical across launches", not var,
          f"{len(var)} prompts varied")

    toks = sum(len(r["tokens"]) for r in done)
    print(f"\n  {len(done)}/{n_expected} runs, {toks} tokens, {wall/60:.1f} min wall "
          f"({wall/max(toks,1):.1f}s/token)")
    print(f"  logs in {a.logdir}")
    print("=" * 78)
    print(f"M11: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
