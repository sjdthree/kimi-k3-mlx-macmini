"""M1: Kimi-K3 checkpoint manifest builder + verifier.

Maps every tensor in the checkpoint to its shard, byte range, shape, dtype, and role,
without ever loading a full tensor (safetensors headers only).

Corrects K3_MLX_PLAN.md section 1: experts are `block_sparse_moe.experts.N.{w1,w2,w3}`
operating in a 3584-dim latent space, not `mlp.experts.N.{gate_up,down}` on hidden.

Usage:
    python tools/k3_manifest.py build   [--out manifest.jsonl.gz]
    python tools/k3_manifest.py verify  [--manifest manifest.jsonl.gz]
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

CKPT = Path("/Volumes/LLMModels/Kimi-K3")
DEFAULT_OUT = Path("manifest.jsonl.gz")


def set_ckpt(path):
    """Retarget the builder (used for the synthetic model and the local NVMe copy)."""
    global CKPT
    CKPT = Path(path)

LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(w[123])\.(weight_packed|weight_scale)$")

DTYPE_BYTES = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1, "I8": 1, "I32": 4, "I64": 8}


# --------------------------------------------------------------------------- roles

def classify(name: str) -> dict:
    """Assign component / layer / role / expert to a tensor name."""
    d: dict = {"component": "other", "layer": None, "expert": None, "role": None}

    if name.startswith("vision_tower."):
        d["component"] = "vision"
    elif name.startswith("mm_projector."):
        d["component"] = "projector"
    elif name.startswith("language_model."):
        d["component"] = "text"

    m = LAYER_RE.search(name)
    if m:
        d["layer"] = int(m.group(1))

    m = EXPERT_RE.search(name)
    if m:
        d["expert"] = int(m.group(1))
        d["role"] = f"routed_expert_{m.group(2)}"
        d["kind"] = m.group(3)
        return d

    tail = name.split(".")
    if "shared_experts" in name:
        d["role"] = "shared_expert_" + tail[-2]
    elif "routed_expert_" in name:
        d["role"] = tail[-2] if tail[-1] == "weight" else tail[-1]
    elif "block_sparse_moe.gate" in name:
        d["role"] = "router_" + tail[-1]
    elif ".self_attn." in name:
        d["role"] = "attn_" + (tail[-2] if tail[-1] == "weight" else tail[-1])
    elif ".mlp." in name:
        d["role"] = "dense_mlp_" + tail[-2]
    elif "res_proj" in name or "res_norm" in name:
        d["role"] = tail[-2]
    elif name.endswith("layernorm.weight") or name.endswith("norm.weight"):
        d["role"] = "norm"
    elif "embed_tokens" in name:
        d["role"] = "embed"
    elif "lm_head" in name:
        d["role"] = "lm_head"
    return d


def read_header(shard: Path) -> tuple[dict, int]:
    with shard.open("rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        return json.loads(fh.read(n)), 8 + n


# --------------------------------------------------------------------------- build

def build(out_path: Path) -> None:
    cfg = json.loads((CKPT / "config.json").read_text())
    tcfg = cfg["text_config"]
    gsize = tcfg["quantization_config"]["config_groups"]["group_0"]["weights"]["group_size"]

    shards = sorted(CKPT.glob("model-*.safetensors"))
    print(f"scanning {len(shards)} shards ...", file=sys.stderr)

    n = 0
    with gzip.open(out_path, "wt") as out:
        meta = {
            "__meta__": True,
            "checkpoint": str(CKPT),
            "num_shards": len(shards),
            "num_hidden_layers": tcfg["num_hidden_layers"],
            "num_experts": tcfg["num_experts"],
            "num_experts_per_token": tcfg["num_experts_per_token"],
            "num_shared_experts": tcfg["num_shared_experts"],
            "hidden_size": tcfg["hidden_size"],
            "routed_expert_hidden_size": tcfg["routed_expert_hidden_size"],
            "moe_intermediate_size": tcfg["moe_intermediate_size"],
            "group_size": gsize,
            # config lists are 1-INDEXED; convert once, here, so nothing downstream re-derives it
            "mla_layers": sorted(i - 1 for i in tcfg["linear_attn_config"]["full_attn_layers"]
                                 if i - 1 < tcfg["num_hidden_layers"]),
            "kda_layers": sorted(i - 1 for i in tcfg["linear_attn_config"]["kda_layers"]
                                 if i - 1 < tcfg["num_hidden_layers"]),
        }
        out.write(json.dumps(meta) + "\n")

        for shard in shards:
            hdr, base = read_header(shard)
            for name, t in hdr.items():
                if name == "__metadata__":
                    continue
                s, e = t["data_offsets"]
                rec = {
                    "name": name,
                    "file": shard.name,
                    "offset": base + s,
                    "nbytes": e - s,
                    "shape": t["shape"],
                    "dtype": t["dtype"],
                    **classify(name),
                }
                if rec.get("kind") == "weight_packed":
                    rec["group_size"] = gsize
                    rec["scale_tensor"] = name.replace("weight_packed", "weight_scale")
                    # packed along the last axis, 2 values per byte
                    rec["logical_shape"] = [rec["shape"][0], rec["shape"][1] * 2]
                out.write(json.dumps(rec) + "\n")
                n += 1

    print(f"wrote {n:,} tensor records -> {out_path} "
          f"({out_path.stat().st_size/2**20:.1f} MiB)", file=sys.stderr)


# -------------------------------------------------------------------------- verify

def load_manifest(path: Path):
    meta, recs = None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("__meta__"):
                meta = r
            else:
                recs.append(r)
    return meta, recs


def verify(path: Path) -> int:
    meta, recs = load_manifest(path)
    idx = json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = json.loads((CKPT / "config.json").read_text())["text_config"]
    fails = 0

    def check(label, ok, detail=""):
        nonlocal fails
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    by_name = {r["name"]: r for r in recs}

    check("no duplicate tensor names", len(by_name) == len(recs),
          f"{len(recs)-len(by_name)} dups")
    check("manifest count == index.json count", len(by_name) == len(idx),
          f"manifest {len(by_name):,} vs index {len(idx):,}")

    mismatched = [n for n, r in by_name.items() if idx.get(n) != r["file"]]
    check("every tensor resolves to its index.json shard", not mismatched,
          f"{len(mismatched)} mismatched")

    missing = set(idx) - set(by_name)
    check("no index.json tensor missing from shards", not missing, f"{len(missing)} missing")

    bad = [r["name"] for r in recs
           if r["nbytes"] != DTYPE_BYTES[r["dtype"]] * _prod(r["shape"])]
    check("byte range matches shape x dtype for every tensor", not bad, f"{len(bad)} bad")

    # ---- structural checks against config
    H = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    E = cfg["num_experts"]
    latent = cfg["routed_expert_hidden_size"]
    inter = cfg["moe_intermediate_size"]

    moe_layers = sorted({r["layer"] for r in recs if r["role"] == "router_weight"})
    check(f"MoE layers are 1..{L-1}", moe_layers == list(range(1, L)),
          f"got {len(moe_layers)} layers")

    per_layer = defaultdict(set)
    for r in recs:
        if r["expert"] is not None:
            per_layer[r["layer"]].add((r["expert"], r["role"], r["kind"]))
    incomplete = {l: len(v) for l, v in per_layer.items() if len(v) != E * 3 * 2}
    check(f"every MoE layer has {E} experts x 3 weights x 2 tensors = {E*3*2:,}",
          not incomplete, f"{len(incomplete)} incomplete: {dict(list(incomplete.items())[:3])}")

    orphans = [r["scale_tensor"] for r in recs
               if r.get("kind") == "weight_packed" and r["scale_tensor"] not in by_name]
    check("every weight_packed has its weight_scale", not orphans, f"{len(orphans)} orphans")

    # group_size consistency: logical cols / scale cols == 32
    gs_bad = []
    for r in recs:
        if r.get("kind") != "weight_packed":
            continue
        sc = by_name[r["scale_tensor"]]
        if r["logical_shape"][1] // sc["shape"][1] != meta["group_size"]:
            gs_bad.append(r["name"])
    check(f"scale grouping == {meta['group_size']} everywhere", not gs_bad, f"{len(gs_bad)} bad")

    # expert shapes: w1/w3 latent->inter, w2 inter->latent
    want = {"routed_expert_w1": [inter, latent], "routed_expert_w3": [inter, latent],
            "routed_expert_w2": [latent, inter]}
    shape_bad = [r["name"] for r in recs
                 if r.get("kind") == "weight_packed" and r["logical_shape"] != want[r["role"]]]
    check(f"expert logical shapes (w1/w3 {inter}x{latent}, w2 {latent}x{inter})",
          not shape_bad, f"{len(shape_bad)} bad")

    # attention typing must match the 0-indexed lists derived in build()
    mla = {r["layer"] for r in recs if r["role"] == "attn_kv_a_proj_with_mqa"}
    kda = {r["layer"] for r in recs if r["role"] == "attn_A_log"}
    check("MLA layers match config (converted from 1-indexed)",
          sorted(mla) == meta["mla_layers"], f"{sorted(mla)[:4]}... n={len(mla)}")
    check("KDA layers match config (converted from 1-indexed)",
          sorted(kda) == meta["kda_layers"], f"n={len(kda)}")
    check("MLA and KDA partition all layers",
          sorted(mla | kda) == list(range(L)) and not (mla & kda))

    # A_log padding: the P0-2 finding, asserted structurally. The invariant is that
    # A_log is stored at head_dim length while only the first num_heads entries carry
    # decay -- NOT the literal value 128, which is just this checkpoint's head_dim.
    lac = cfg["linear_attn_config"]
    hd, nheads = lac["head_dim"], lac["num_heads"]
    alog = [r for r in recs if r["role"] == "attn_A_log"]
    check(f"A_log is stored at head_dim={hd} on every KDA layer "
          f"(zero-padded from num_heads={nheads}; see P0_RESULTS.md)",
          all(r["shape"] == [hd] for r in alog),
          f"n={len(alog)}" + ("" if hd != nheads else "  [degenerate: hd == nheads]"))

    files = Counter(r["file"] for r in recs)
    check("all shards referenced", len(files) == meta["num_shards"],
          f"{len(files)}/{meta['num_shards']}")

    # size accounting
    tot = sum(r["nbytes"] for r in recs)
    dense = sum(r["nbytes"] for r in recs if r["expert"] is None and r["component"] == "text")
    exp = sum(r["nbytes"] for r in recs if r["expert"] is not None)
    vis = sum(r["nbytes"] for r in recs if r["component"] in ("vision", "projector"))
    print(f"\n  total {tot/2**40:.3f} TiB | text-dense {dense/2**30:.1f} GiB "
          f"| experts {exp/2**40:.3f} TiB | vision {vis/2**30:.2f} GiB")
    per_tok_exp = cfg["num_experts_per_token"] * len(moe_layers) * (exp / (len(moe_layers) * E))
    print(f"  per-token expert reads: {cfg['num_experts_per_token']} x {len(moe_layers)} layers "
          f"= {per_tok_exp/2**30:.1f} GiB")

    print(f"\nM1 verify: {'PASS' if fails == 0 else f'FAIL ({fails} checks)'}")
    return fails


def _prod(xs):
    p = 1
    for x in xs:
        p *= x
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "verify"])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint directory (defaults to the NAS path)")
    a = ap.parse_args()
    if a.ckpt:
        set_ckpt(a.ckpt)
    if a.cmd == "build":
        build(a.out)
    else:
        sys.exit(1 if verify(a.manifest) else 0)
