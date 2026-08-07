"""Weight access for the k3ref oracle.

Reads from the local working-set cache (blob + index) when present, else falls back to
the NAS via the manifest. Returns fp32 torch tensors; MXFP4 experts are decoded on access.
"""
from __future__ import annotations

import gzip
import json
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

import os

# K3_CKPT / K3_CACHE let the oracle read from the fast local copy instead of the NAS.
CKPT = Path(os.environ.get("K3_CKPT", "/Volumes/LLMModels/Kimi-K3"))
CACHE = Path(os.environ.get("K3_CACHE", "~/k3_cache")).expanduser()

_NP = {"F32": np.float32, "BF16": np.uint16, "F16": np.float16,
       "U8": np.uint8, "I32": np.int32, "I64": np.int64}

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def decode_mxfp4(packed: np.ndarray, scales: np.ndarray, group_size: int = 32) -> np.ndarray:
    """See p0/p0_3_mxfp4.py - bit-exact vs compressed-tensors 0.17.1."""
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    mag = E2M1[codes & 0x07]
    vals = np.where((codes & 0x08).astype(bool), -mag, mag)
    scale = np.ldexp(np.ones(scales.shape, dtype=np.float32), scales.astype(np.int32) - 127)
    return vals * np.repeat(scale, group_size, axis=1)


def _bf16(a: np.ndarray) -> np.ndarray:
    return (a.astype(np.uint32) << 16).view(np.float32)


class WeightStore:
    def __init__(self, cache: Path = CACHE,
                 manifest: Path = Path(os.environ.get("K3_MANIFEST",
                                                      "manifest.jsonl.gz"))):
        self.blobs: dict[str, tuple[Path, dict]] = {}
        for name in ("dense", "experts"):
            idx = cache / f"{name}.index.json"
            if idx.exists():
                self.blobs[name] = (cache / f"{name}.bin", json.loads(idx.read_text()))
        self.local = {k: v for _, (_, ix) in self.blobs.items() for k, v in ix.items()}
        self._mani: dict | None = None
        self._manifest_path = manifest
        self._shard_hdr: dict = {}

    # ---------------------------------------------------------------- sources
    @property
    def manifest(self) -> dict:
        if self._mani is None:
            m = {}
            with gzip.open(self._manifest_path, "rt") as fh:
                for line in fh:
                    r = json.loads(line)
                    if not r.get("__meta__"):
                        m[r["name"]] = r
            self._mani = m
        return self._mani

    def _from_cache(self, name: str) -> np.ndarray:
        for blob, ix in self.blobs.values():
            if name in ix:
                e = ix[name]
                with blob.open("rb") as fh:
                    fh.seek(e["offset"])
                    raw = fh.read(e["nbytes"])
                a = np.frombuffer(raw, dtype=_NP[e["dtype"]])
                return a.reshape(e["shape"]), e["dtype"]
        raise KeyError(name)

    def _from_nas(self, name: str):
        r = self.manifest[name]
        with (CKPT / r["file"]).open("rb") as fh:
            fh.seek(r["offset"])
            raw = fh.read(r["nbytes"])
        return np.frombuffer(raw, dtype=_NP[r["dtype"]]).reshape(r["shape"]), r["dtype"]

    def raw(self, name: str):
        try:
            return self._from_cache(name)
        except KeyError:
            return self._from_nas(name)

    # ---------------------------------------------------------------- public
    def get(self, name: str) -> torch.Tensor:
        """Any non-quantised tensor, as fp32."""
        a, dt = self.raw(name)
        if dt == "BF16":
            a = _bf16(a)
        return torch.from_numpy(np.ascontiguousarray(a)).float()

    def expert(self, layer: int, expert: int, w: str) -> torch.Tensor:
        """Routed expert weight (w1|w2|w3), MXFP4-decoded to fp32."""
        base = f"language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{w}"
        p, _ = self.raw(base + ".weight_packed")
        s, _ = self.raw(base + ".weight_scale")
        return torch.from_numpy(decode_mxfp4(np.ascontiguousarray(p), np.ascontiguousarray(s)))

    def is_local(self, name: str) -> bool:
        return name in self.local


@lru_cache(maxsize=1)
def default_store() -> WeightStore:
    return WeightStore()
