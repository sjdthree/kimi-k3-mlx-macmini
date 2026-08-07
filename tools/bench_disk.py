"""Comparable disk benchmark for the Kimi-K3 streaming workload.

The workload is NOT generic: per generated token the runtime reads ~128 GiB as a mix of
large dense-weight tensors (typically 100-350 MiB each) and ~4400 expert-projection reads
of exactly 5.25 MiB scattered inside 17 GiB shards. So the number that matters is
**random reads at ~5 MiB granularity**, not queue-depth-1 4K IOPS and not pure sequential.

Cache defeat: macOS F_NOCACHE alone is not sufficient -- an earlier run of this benchmark
reported random reads FASTER than sequential, which is impossible and revealed page-cache
hits. So the test file is sized larger than RAM, and the harness flags the contradiction
if it reappears.

    python tools/bench_disk.py --path /Volumes/USB4_2TBB/.bench --size-gib 80
"""
from __future__ import annotations

import argparse
import fcntl
import os
import random
import time

F_NOCACHE = 48
BLOCK = 8 << 20
EXPERT_READ = 5_505_024          # exact size of one K3 expert projection


def _open(path, flags):
    fd = os.open(path, flags)
    try:
        fcntl.fcntl(fd, F_NOCACHE, 1)
    except OSError:
        pass
    return fd


def write_file(path, size):
    fd = _open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    buf = os.urandom(BLOCK)
    t0 = time.time()
    try:
        left = size
        while left:
            n = os.write(fd, buf[:min(BLOCK, left)])
            left -= n
        os.fsync(fd)
    finally:
        os.close(fd)
    return size / (time.time() - t0)


def seq_read(path, size):
    fd = _open(path, os.O_RDONLY)
    t0, got = time.time(), 0
    try:
        while got < size:
            b = os.read(fd, BLOCK)
            if not b:
                break
            got += len(b)
    finally:
        os.close(fd)
    return got / (time.time() - t0)


def rand_read(path, size, n=400, sz=EXPERT_READ, seed=1):
    random.seed(seed)
    fd = _open(path, os.O_RDONLY)
    lat, tot = [], 0
    t0 = time.time()
    try:
        for _ in range(n):
            os.lseek(fd, random.randrange(0, max(size - sz, 1)), os.SEEK_SET)
            t1 = time.time()
            left = sz
            while left:
                b = os.read(fd, left)
                if not b:
                    break
                left -= len(b)
            lat.append(time.time() - t1)
            tot += sz - left
    finally:
        os.close(fd)
    lat.sort()
    return tot / (time.time() - t0), lat[len(lat) // 2], lat[int(len(lat) * 0.95)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="test file path on the target volume")
    ap.add_argument("--size-gib", type=int, default=80,
                    help="must exceed RAM to defeat the page cache")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    size = a.size_gib << 30

    print(f"target      {a.path}")
    print(f"file size   {a.size_gib} GiB (must exceed RAM to be trustworthy)")
    print(f"random read {EXPERT_READ/2**20:.2f} MiB  <- one K3 expert projection")
    print("-" * 66)

    w = write_file(a.path, size)
    print(f"  sequential write   {w/1e6:8.1f} MB/s")
    s = seq_read(a.path, size)
    print(f"  sequential read    {s/1e6:8.1f} MB/s")
    r, p50, p95 = rand_read(a.path, size)
    print(f"  random read 5.25MiB{r/1e6:8.1f} MB/s   p50 {p50*1e3:6.2f} ms  "
          f"p95 {p95*1e3:6.2f} ms")

    if not a.keep:
        os.unlink(a.path)

    if r > s * 1.05:
        print("\n  !! random faster than sequential -> page-cache contamination.")
        print("     Increase --size-gib above RAM and re-run; results are not usable.")
        return

    # --------------------------------------------------------- workload projection
    DENSE, EXPERTS = 103.5, 24.1        # GiB read per generated token
    print("\n" + "-" * 66)
    print("  Kimi-K3 per-token projection (128 GiB: 103.5 dense + 24.1 experts)")
    dense_s = DENSE * 2 ** 30 / r
    exp_s = EXPERTS * 2 ** 30 / r
    print(f"    dense   {dense_s:6.1f} s")
    print(f"    experts {exp_s:6.1f} s")
    print(f"    compute    2.0 s")
    print(f"    TOTAL   {dense_s + exp_s + 2.0:6.1f} s/token  "
          f"({1/(dense_s+exp_s+2.0):.3f} tok/s)")
    print(f"\n  note: measured end-to-end efficiency through the loader is ~79% of raw,")
    print(f"        so expect roughly {(dense_s+exp_s)/0.79 + 2.0:.0f} s/token in practice.")


if __name__ == "__main__":
    main()
