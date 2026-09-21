"""Copy the Kimi-K3 checkpoint to a local drive, resumably and verifiably.

macOS ships openrsync (protocol 29), which lacks --info=progress2 and friends, so this
does the job directly: file-at-a-time, skip-if-complete, per-file size verification.

    python tools/k3_copy.py --dest /Volumes/USB4_2TBB/Kimi-K3
    python tools/k3_copy.py --dest /Volumes/USB4_2TBB/Kimi-K3 --verify-only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import time
from pathlib import Path

SRC = Path("/Volumes/LLMModels/Kimi-K3")
BLOCK = 16 << 20


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def files(src: Path):
    out = []
    for p in sorted(src.iterdir()):
        if p.is_file() and p.suffix.lower() != ".pdf":
            out.append(p)
    return out


def copy_file(src: Path, dst: Path) -> int:
    tmp = dst.with_suffix(dst.suffix + ".part")
    n = 0
    with src.open("rb") as fi, tmp.open("wb") as fo:
        while True:
            b = fi.read(BLOCK)
            if not b:
                break
            fo.write(b)
            n += len(b)
        fo.flush()
        os.fsync(fo.fileno())
    tmp.rename(dst)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--sha256", action="store_true", help="Compare file content hashes; reads both copies in full")
    a = ap.parse_args()

    a.dest.mkdir(parents=True, exist_ok=True)
    todo = files(a.src)
    total = sum(p.stat().st_size for p in todo)
    print(f"{len(todo)} files, {total/2**40:.3f} TiB", flush=True)

    done_bytes = 0
    copied = 0
    t0 = time.time()
    bad = []

    for i, p in enumerate(todo, 1):
        dst = a.dest / p.name
        want = p.stat().st_size
        if dst.exists() and dst.stat().st_size == want and (not a.sha256 or digest(p) == digest(dst)):
            done_bytes += want
            continue
        if a.verify_only:
            bad.append((p.name, want, dst.stat().st_size if dst.exists() else None))
            continue
        t1 = time.time()
        got = copy_file(p, dst)
        done_bytes += got
        copied += 1
        if got != want or (a.sha256 and digest(p) != digest(dst)):
            bad.append((p.name, want, got))
        el = time.time() - t0
        rate = done_bytes / el
        eta = (total - done_bytes) / rate if rate else 0
        print(f"  [{i:>3}/{len(todo)}] {p.name}  {got/2**30:6.2f} GiB @ "
              f"{got/(time.time()-t1)/1e6:6.1f} MB/s | "
              f"{done_bytes/2**40:.3f}/{total/2**40:.3f} TiB  "
              f"avg {rate/1e6:6.1f} MB/s  ETA {eta/3600:4.2f} h", flush=True)

    el = time.time() - t0
    print(f"\n{'verified' if a.verify_only else 'copied'} {copied} files, "
          f"{done_bytes/2**40:.3f} TiB in {el/3600:.2f} h")
    if bad:
        print(f"MISMATCHES ({len(bad)}):")
        for n, w, g in bad[:20]:
            print(f"  {n}: want {w} got {g}")
    else:
        print("all files present and " + ("SHA-256 matched" if a.sha256 else "size-correct (contents not hashed)"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
