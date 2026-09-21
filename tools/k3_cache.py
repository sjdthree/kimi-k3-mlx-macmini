"""Local working-set cache for Kimi-K3.

Copies a selected subset of tensors off the NAS into a flat local blob + JSON index.
Byte-level copy: no dtype conversion, so the cache is lossless and fast.

The blob+index layout is deliberately the shape an expert/tensor streaming loader wants
(mmap the blob, seek to an offset), so this doubles as the M1 staging format. When the
USB4 NVMe arrives, point --dest at it and widen --layers.

Usage:
    python tools/k3_cache.py --dest ~/k3_cache --layers 0-12 --expert-layers 1
    python tools/k3_cache.py --dest ~/k3_cache --plan-only
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from collections import defaultdict
from pathlib import Path

CKPT = Path(os.environ.get("K3_CKPT", "Kimi-K3"))


def parse_range(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def load_manifest(path: Path):
    meta, recs = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            (recs.append(r) if not r.get("__meta__") else None)
            if r.get("__meta__"):
                meta = r
    return meta, recs


def select(recs, layers: list[int], expert_layers: list[int], globals_too: bool):
    """Dense tensors for `layers`; full expert sets for `expert_layers`; model-level tensors."""
    lset, eset = set(layers), set(expert_layers)
    picked = []
    for r in recs:
        if r["component"] != "text":
            continue
        if r["expert"] is not None:
            if r["layer"] in eset:
                picked.append(r)
        elif r["layer"] is not None:
            if r["layer"] in lset:
                picked.append(r)
        elif globals_too:
            picked.append(r)          # embed_tokens, lm_head, norm, output_attn_res_*
    return picked


def copy_out(picked, dest: Path, name: str):
    """Sequentialise by (shard, offset) so the NAS sees forward-only reads."""
    dest.mkdir(parents=True, exist_ok=True)
    blob_path, idx_path = dest / f"{name}.bin", dest / f"{name}.index.json"

    if blob_path.exists() or idx_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache {name}; choose a fresh destination")
    by_shard = defaultdict(list)
    for r in picked:
        by_shard[r["file"]].append(r)
    for v in by_shard.values():
        v.sort(key=lambda r: r["offset"])

    total = sum(r["nbytes"] for r in picked)
    index, written, t0 = {}, 0, time.time()

    with blob_path.open("wb") as out:
        for shard in sorted(by_shard):
            with (CKPT / shard).open("rb") as src:
                for r in by_shard[shard]:
                    src.seek(r["offset"])
                    off = out.tell()
                    left = r["nbytes"]
                    while left:
                        b = src.read(min(8 << 20, left))
                        if not b:
                            raise IOError(f"short read on {r['name']}")
                        out.write(b)
                        left -= len(b)
                    index[r["name"]] = {"offset": off, "nbytes": r["nbytes"],
                                        "shape": r["shape"], "dtype": r["dtype"]}
                    written += r["nbytes"]
            el = time.time() - t0
            print(f"  {shard}  {written/2**30:7.2f}/{total/2**30:.2f} GiB "
                  f"({written/el/1e6:6.1f} MB/s, {el/60:.1f} min)", flush=True)

    idx_path.write_text(json.dumps(index))
    print(f"  -> {blob_path} ({written/2**30:.2f} GiB, {len(index):,} tensors)")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("manifest.jsonl.gz"))
    ap.add_argument("--src", type=Path, default=None,
                    help="read from here instead of the manifest's checkpoint path "
                         "(e.g. the USB4 copy, which is far faster than the NAS)")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--layers", default="0-12", help="dense layers to cache")
    ap.add_argument("--expert-layers", default="1", help="layers to cache ALL 896 experts for")
    ap.add_argument("--no-globals", action="store_true")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args()

    meta, recs = load_manifest(a.manifest)
    CKPT = a.src or Path(os.environ.get("K3_CKPT", meta["checkpoint"]))
    layers, elayers = parse_range(a.layers), parse_range(a.expert_layers)
    picked = select(recs, layers, elayers, not a.no_globals)

    dense = [r for r in picked if r["expert"] is None]
    exps = [r for r in picked if r["expert"] is not None]
    dsz = sum(r["nbytes"] for r in dense)
    esz = sum(r["nbytes"] for r in exps)
    print(f"plan: dense layers {a.layers} -> {len(dense):,} tensors, {dsz/2**30:.2f} GiB")
    print(f"      experts for layers {a.expert_layers} -> {len(exps):,} tensors, {esz/2**30:.2f} GiB")
    print(f"      TOTAL {(dsz+esz)/2**30:.2f} GiB")
    free = os.statvfs(a.dest.expanduser().parent if not a.dest.expanduser().exists()
                      else a.dest.expanduser())
    print(f"      free at dest: {free.f_bavail*free.f_frsize/2**30:.0f} GiB")
    if a.plan_only:
        raise SystemExit(0)

    dest = a.dest.expanduser()
    t0 = time.time()
    if dense:
        print("\ncaching dense ...")
        copy_out(dense, dest, "dense")
    if exps:
        print("\ncaching experts ...")
        copy_out(exps, dest, "experts")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
