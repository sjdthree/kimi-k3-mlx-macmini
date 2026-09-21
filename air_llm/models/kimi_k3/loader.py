"""Weight loading + layer assembly for the MLX Kimi-K3 model.

Resolves tensors from (in order) a local blob cache, then the checkpoint shards named by
the manifest. Experts are handed to LatentMoE as a callable so a single expert's packed
bytes are materialised, used, and dropped -- never a whole layer.
"""
from __future__ import annotations

import gzip
import json
import os
from collections import OrderedDict
from pathlib import Path

import mlx.core as mx
import numpy as np

from .attention_kda import KDAAttention
from .attention_mla import MLAAttention
from .layer import DecoderLayer, DenseMLP
from .moe import LatentMoE

_NP = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16,
       "U8": np.uint8, "I32": np.int32, "I64": np.int64}

class _NotAdjacent(Exception):
    """The packed/scale pair is not contiguous; use two reads."""


MOE_KEYS = ("gate.weight", "gate.e_score_correction_bias",
            "routed_expert_down_proj.weight", "routed_expert_up_proj.weight",
            "routed_expert_norm.weight", "shared_experts.gate_proj.weight",
            "shared_experts.up_proj.weight", "shared_experts.down_proj.weight")


def _open_manifest(path):
    """gzip.open with an actionable error; the manifest is gitignored and rebuilt."""
    try:
        return gzip.open(path, "rt")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"manifest not found: {path}\n"
            f"It is machine-specific (it embeds an absolute checkpoint path) and so is\n"
            f"gitignored. Rebuild it in ~8 s:\n"
            f"    python tools/k3_manifest.py build --ckpt /path/to/Kimi-K3 --out {path}"
        ) from None


class TensorSource:
    """Tensor access with an optional bounded resident cache.

    Profiling showed the runtime is I/O-bound ~20x over compute: 129.8 GiB read per
    token, of which 105.7 GiB is DENSE weights that are byte-identical on every token.
    Only 24.1 GiB is expert data. So holding as much dense weight resident as memory
    allows removes proportional I/O for free.

    `resident_gb` is opt-in and defaults to 0 (strict streaming). Note the tension with
    spec section 16, which literally requires "only the current layer's non-expert
    weights are resident" -- that phrasing assumes re-reading is the only option. With a
    budget set, the streaming guarantee that still holds is the one that matters for
    memory safety: never more than `resident_gb` plus one layer plus one expert.

    Experts are deliberately NOT cached: 98% of them are unused per token, so they would
    evict dense weights that are certain to be reused.
    """

    def __init__(self, manifest: Path = Path(os.environ.get("K3_MANIFEST", "manifest.jsonl.gz")),
                 ckpt: Path | None = None, cache: Path | None = None,
                 resident_gb: float = 0.0, weight_dtype=mx.float32, fused: bool | None = None):
        self.recs: dict[str, dict] = {}
        self.meta: dict = {}
        with _open_manifest(manifest) as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("__meta__"):
                    self.meta = r
                else:
                    self.recs[r["name"]] = r
        self.ckpt = Path(ckpt or os.environ.get("K3_CKPT") or self.meta["checkpoint"])
        self.blobs = []
        if cache:
            for name in ("dense", "experts"):
                idx = Path(cache) / f"{name}.index.json"
                if idx.exists():
                    self.blobs.append((Path(cache) / f"{name}.bin",
                                       json.loads(idx.read_text())))
        self.bytes_read = 0
        self.expert_bytes = 0
        self.experts_loaded = 0
        # Spec section 13. VALIDATION mode keeps weights in fp32 so parity against the
        # k3ref oracle is exact to ~1e-5. RUNTIME mode keeps them in bf16 -- the dtype
        # they are stored in on disk, so no information is lost -- which halves unified
        # memory per tensor and therefore doubles the effective resident-cache capacity.
        self.weight_dtype = weight_dtype
        self.fused = weight_dtype != mx.float32 if fused is None else fused
        self.resident_budget = int(resident_gb * 2 ** 30)
        self._resident: OrderedDict[str, mx.array] = OrderedDict()
        self._resident_bytes = 0
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------ resident cache
    def resident_get(self, name: str):
        a = self._resident.get(name)
        if a is not None:
            self._resident.move_to_end(name)
            self.hits += 1
        return a

    def resident_put(self, name: str, a: mx.array) -> None:
        """FILL-AND-PIN, deliberately not LRU.

        A token walks layers 0..92 in order and the next token repeats the same walk.
        Against that cyclic scan, LRU is pathological: by the time layer 0 comes round
        again it has been evicted by layer 92, so the hit rate is ~0 no matter how big
        the cache is. Measured on the real model -- a 24 GiB LRU cache produced 127.58
        GiB of reads per token, identical to no cache at all.

        Filling once and never evicting turns the hit rate into
        cache_size / working_set, which is the best any policy can do here without
        knowing the cycle length. (MRU would also work; pinning is simpler and its
        residency is predictable, which matters for the memory ceiling.)
        """
        if self.resident_budget <= 0 or name in self._resident:
            return
        nb = a.nbytes
        if self._resident_bytes + nb > self.resident_budget:
            return                      # full: keep what we have, stream the rest
        self._resident[name] = a
        self._resident_bytes += nb

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def cache_stats(self) -> dict:
        tot = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": self.hits / tot if tot else 0.0,
                "resident_bytes": self._resident_bytes,
                "entries": len(self._resident)}

    def reset_counters(self) -> None:
        self.bytes_read = 0
        self.expert_bytes = 0
        self.experts_loaded = 0

    # --------------------------------------------------------------- row access
    def _locate(self, name: str) -> tuple[Path, int, dict]:
        """(file, base_offset, record) for a tensor, cache first then checkpoint."""
        for blob, ix in self.blobs:
            e = ix.get(name)
            if e is not None:
                return blob, e["offset"], e
        r = self.recs[name]
        return self.ckpt / r["file"], r["offset"], r

    def rows(self, name: str, indices) -> np.ndarray:
        """Read only the given rows of a 2-D tensor. For embed_tokens this is 14 KiB
        per token instead of the full 2.35 GiB table."""
        path, base, r = self._locate(name)
        cols = r["shape"][1]
        itemsize = np.dtype(_NP[r["dtype"]]).itemsize
        stride = cols * itemsize
        out = np.empty((len(indices), cols), dtype=_NP[r["dtype"]])
        with path.open("rb") as fh:
            for i, row in enumerate(indices):
                fh.seek(base + row * stride)
                out[i] = np.frombuffer(fh.read(stride), dtype=_NP[r["dtype"]])
        self.bytes_read += len(indices) * stride
        if r["dtype"] == "BF16":
            return (out.astype(np.uint32) << 16).view(np.float32)
        return out

    def row_block(self, name: str, start: int, n: int) -> np.ndarray:
        """A contiguous block of rows -- used to stream lm_head in bounded chunks."""
        path, base, r = self._locate(name)
        cols = r["shape"][1]
        itemsize = np.dtype(_NP[r["dtype"]]).itemsize
        stride = cols * itemsize
        with path.open("rb") as fh:
            fh.seek(base + start * stride)
            buf = fh.read(n * stride)
        self.bytes_read += len(buf)
        a = np.frombuffer(buf, dtype=_NP[r["dtype"]]).reshape(n, cols)
        if r["dtype"] == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a

    def raw(self, name: str) -> tuple[np.ndarray, str]:
        for blob, ix in self.blobs:
            e = ix.get(name)
            if e is not None:
                with blob.open("rb") as fh:
                    fh.seek(e["offset"])
                    b = fh.read(e["nbytes"])
                self.bytes_read += len(b)      # cache reads are still reads
                return np.frombuffer(b, dtype=_NP[e["dtype"]]).reshape(e["shape"]), e["dtype"]
        r = self.recs[name]
        with (self.ckpt / r["file"]).open("rb") as fh:
            fh.seek(r["offset"])
            b = fh.read(r["nbytes"])
        self.bytes_read += len(b)
        return np.frombuffer(b, dtype=_NP[r["dtype"]]).reshape(r["shape"]), r["dtype"]

    def mx(self, name: str) -> mx.array:
        hit = self.resident_get(name)
        if hit is not None:
            return hit
        self.misses += 1
        a, dt = self.raw(name)
        a = np.ascontiguousarray(a)
        if dt == "BF16":
            if self.weight_dtype == mx.bfloat16:
                # reinterpret the raw bf16 bits directly -- no widening round-trip
                arr = mx.array(a).view(mx.bfloat16)
            else:
                arr = mx.array((a.astype(np.uint32) << 16).view(np.float32))
        else:
            arr = mx.array(a)
        self.resident_put(name, arr)
        return arr

    def mx_u8(self, name: str) -> mx.array:
        a, _ = self.raw(name)
        return mx.array(np.ascontiguousarray(a))

    # ------------------------------------------------------------------ experts
    def expert_source(self, layer: int, expert: int, which: str):
        """Fetch one MXFP4 projection as (packed, scales).

        In every shard checked, `weight_scale` starts exactly where `weight_packed`
        ends (240/240 projections, zero gap), so the pair is fetched in ONE contiguous
        read and sliced. Two reads of 5.25 MiB + 344 KiB were measuring 2295 MB/s on
        the USB4 drive against 3378 MB/s raw -- the small second read was dragging the
        average down. Falls back to two reads if the layout is ever not adjacent.
        """
        base = (f"language_model.model.layers.{layer}.block_sparse_moe.experts."
                f"{expert}.{which}")
        self.experts_loaded += 1
        before = self.bytes_read
        try:
            out = self._read_pair(base + ".weight_packed", base + ".weight_scale")
        except _NotAdjacent:
            out = self.mx_u8(base + ".weight_packed"), self.mx_u8(base + ".weight_scale")
        from .mxfp4 import assert_scales_representable
        assert_scales_representable(out[1])
        self.expert_bytes += self.bytes_read - before
        return out

    def _read_pair(self, name_a: str, name_b: str):
        pa, oa, ra = self._locate(name_a)
        pb, ob, rb = self._locate(name_b)
        if pa != pb or oa + ra["nbytes"] != ob:
            raise _NotAdjacent
        n = ra["nbytes"] + rb["nbytes"]
        with pa.open("rb") as fh:
            fh.seek(oa)
            buf = fh.read(n)
        if len(buf) != n:
            raise _NotAdjacent
        self.bytes_read += n
        a = np.frombuffer(buf, dtype=np.uint8, count=ra["nbytes"]).reshape(ra["shape"])
        b = np.frombuffer(buf, dtype=np.uint8, offset=ra["nbytes"],
                          count=rb["nbytes"]).reshape(rb["shape"])
        return mx.array(a), mx.array(b)


def build_layer(cfg, layer: int, src: TensorSource) -> DecoderLayer:
    p = f"language_model.model.layers.{layer}."
    norms = {k: src.mx(p + k) for k in DecoderLayer.NORM_KEYS}

    if cfg.is_kda(layer):
        attn = KDAAttention(cfg, {k: src.mx(p + "self_attn." + k) for k in KDAAttention.KEYS})
    else:
        attn = MLAAttention(cfg, {k: src.mx(p + "self_attn." + k) for k in MLAAttention.KEYS})

    if layer < cfg.first_k_dense_replace:
        mlp = DenseMLP(cfg, {k: src.mx(p + "mlp." + k)
                             for k in ("gate_proj.weight", "up_proj.weight", "down_proj.weight")})
    else:
        w = {k: src.mx(p + "block_sparse_moe." + k) for k in MOE_KEYS}
        mlp = LatentMoE(cfg, layer, w, src.expert_source, fused=src.fused)

    return DecoderLayer(cfg, layer, norms, attn, mlp)
