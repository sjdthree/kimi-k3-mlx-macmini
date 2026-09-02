#!/usr/bin/env python3
"""Run every suite in tests/ as a subprocess and summarise.

Each suite is a standalone script that exits non-zero on failure. Suites needing the
1.42 TiB checkpoint skip themselves (exit 0, "SKIP:" on stdout) unless K3_CKPT points at
one, so a fresh clone runs green on the checkpoint-free half.

    python tests/run_all.py                       # everything reachable
    K3_CKPT=/path/to/Kimi-K3 python tests/run_all.py
    python tests/run_all.py --only test_mxfp4_mlx test_mla_cache
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# The full-depth parity run walks all 93 layers off disk and takes hours. It is the
# headline correctness result, not a routine gate -- opt in with --include-slow.
SLOW = {"test_full_parity"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="run just these suites (bare names)")
    ap.add_argument("--include-slow", action="store_true", help=f"also run {sorted(SLOW)}")
    a = ap.parse_args()

    names = sorted(p.stem for p in HERE.glob("test_*.py"))
    if a.only:
        names = [n for n in names if n in set(a.only)]
    if not a.include_slow:
        names = [n for n in names if n not in SLOW]

    rows, failed = [], 0
    for n in names:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / f"{n}.py")],
                           capture_output=True, text=True)
        dt = time.time() - t0
        out = r.stdout + r.stderr
        if r.returncode != 0:
            status, failed = "FAIL", failed + 1
        elif "SKIP:" in out or "skipping" in out:
            status = "SKIP"
        else:
            status = "PASS"
        rows.append((status, n, dt, out))

    width = max((len(n) for _, n, _, _ in rows), default=10)
    print("=" * (width + 24))
    for status, n, dt, _ in rows:
        print(f"  {status:4}  {n:<{width}}  {dt:6.1f}s")
    print("=" * (width + 24))

    for status, n, _, out in rows:
        if status == "FAIL":
            print(f"\n--- {n} ---\n{out.rstrip()[-2000:]}")

    tally = {s: sum(1 for r in rows if r[0] == s) for s in ("PASS", "SKIP", "FAIL")}
    print(f"\n{tally['PASS']} passed, {tally['SKIP']} skipped, {tally['FAIL']} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
