"""Minimal safetensors reader for the K3 checkpoint (no torch, no full-shard load)."""
import json, struct, numpy as np

CKPT = "/Volumes/LLMModels/Kimi-K3/"
IDX = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]
_HDR = {}

_NP = {"F32": np.float32, "BF16": np.uint16, "U8": np.uint8,
       "F16": np.float16, "I64": np.int64, "I32": np.int32}


def header(shard):
    if shard not in _HDR:
        with open(CKPT + shard, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _HDR[shard] = (json.loads(fh.read(n)), 8 + n)
    return _HDR[shard]


def load(name, shard=None):
    shard = shard or IDX[name]
    hdr, base = header(shard)
    meta = hdr[name]
    s, e = meta["data_offsets"]
    with open(CKPT + shard, "rb") as fh:
        fh.seek(base + s)
        raw = fh.read(e - s)
    a = np.frombuffer(raw, dtype=_NP[meta["dtype"]])
    if meta["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a.reshape(meta["shape"])


def meta(name, shard=None):
    shard = shard or IDX[name]
    return header(shard)[0][name]
