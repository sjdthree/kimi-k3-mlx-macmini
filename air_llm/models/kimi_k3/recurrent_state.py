"""Spec section 12: explicit Kimi-K3 model state, with a backend-neutral wire format.

THE BACKEND TRAP THIS EXISTS TO CLOSE
-------------------------------------
The KDA short-convolution state has a different memory layout per backend:

    MLX   [B, k-1, C]   channels-LAST  (mx.conv1d is channels-last)
    torch [B, C, k-1]   channels-FIRST (F.conv1d is channels-first)

Same values, transposed. Nothing in either implementation notices, because each is
self-consistent -- the M5 parity test only caught it because it compared the two directly.
But state saved from one backend and restored into the other would be silently wrong: the
conv window would be scrambled, and generation would diverge a few tokens later with no
error anywhere.

So the on-disk format pins ONE canonical layout (channels-first, the checkpoint/torch
convention) and every backend converts on the way in and out. `CANONICAL_LAYOUT` is part
of the file header, and loading refuses a file it does not recognise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

# v2: MLA caches hold the 512-wide latent + 64-wide MQA rope slice instead of the
# expanded per-head key/value. A v1 file cannot be reinterpreted as v2 -- the expansion
# through kv_b is not invertible -- so `load` refuses it rather than guessing.
FORMAT_VERSION = 2
CANONICAL_CONV_LAYOUT = "channels_first"      # [B, C, k-1]


# --------------------------------------------------------------------- conversions

def conv_to_canonical(state: mx.array) -> np.ndarray:
    """MLX [B, k-1, C] -> canonical [B, C, k-1]."""
    return np.array(state, copy=False).transpose(0, 2, 1)


def conv_from_canonical(arr: np.ndarray) -> mx.array:
    """canonical [B, C, k-1] -> MLX [B, k-1, C]."""
    return mx.array(np.ascontiguousarray(arr.transpose(0, 2, 1)))


# -------------------------------------------------------------------------- state

@dataclass
class KimiK3State:
    """Everything that must persist between generated tokens."""
    token_position: int = 0
    kda_states: dict[int, mx.array] = field(default_factory=dict)      # [B,H,K,V]
    kda_conv: dict[int, tuple] = field(default_factory=dict)           # 3x [B,k-1,C]
    mla_caches: dict[int, object] = field(default_factory=dict)        # MLACache

    # NOT part of the carried state, despite spec section 12 listing
    # `attention_residuals`. block_residual is allocated fresh inside every forward pass
    # (modeling_kimi_linear.py:1188), accumulates across layers, and is discarded. It is
    # sized to batch*seqlen, so persisting it also breaks the moment prefill (N=prompt)
    # gives way to decode (N=1). Kept here only so callers can stash one for debugging.
    block_residual: mx.array | None = None                            # [N,B,H]

    # ------------------------------------------------------------------ lifecycle
    def advance(self, n: int = 1) -> None:
        self.token_position += n

    def memory_report(self) -> dict[str, int]:
        """Bytes by component (spec section 12: 'report state memory by component')."""
        def nb(a):
            return 0 if a is None else a.nbytes

        kda = sum(nb(v) for v in self.kda_states.values())
        conv = sum(nb(c) for t in self.kda_conv.values() for c in t)
        # nbytes_used, not the buffer size: the cache carries chunked slack, and reporting
        # capacity would make the number jump by a chunk every CACHE_STEP tokens.
        mla = sum(c.nbytes_used() for c in self.mla_caches.values())
        br = nb(self.block_residual)
        return {"kda_recurrent": kda, "kda_conv": conv, "mla_cache": mla,
                "block_residual": br, "total": kda + conv + mla + br}

    # ----------------------------------------------------------------- (de)serialise
    def save(self, path: str | Path) -> None:
        path = Path(path)
        arrays: dict[str, np.ndarray] = {}
        meta = {
            "format_version": FORMAT_VERSION,
            "conv_layout": CANONICAL_CONV_LAYOUT,
            "token_position": self.token_position,
            "kda_layers": sorted(self.kda_states),
            "mla_layers": sorted(self.mla_caches),
            "has_block_residual": self.block_residual is not None,
        }
        for L, s in self.kda_states.items():
            arrays[f"kda_state.{L}"] = np.array(s, copy=False)
        for L, (q, k, v) in self.kda_conv.items():
            # convert OUT of the MLX layout on the way to disk
            arrays[f"kda_conv.{L}.q"] = conv_to_canonical(q)
            arrays[f"kda_conv.{L}.k"] = conv_to_canonical(k)
            arrays[f"kda_conv.{L}.v"] = conv_to_canonical(v)
        meta["mla_quant"] = {}
        for L, cache in self.mla_caches.items():
            n = cache.length
            # trim to the valid prefix -- the buffer tail is uninitialised slack
            arrays[f"mla.{L}.rot"] = np.array(cache.rot[:, :n], copy=False)
            if cache.quantised:
                # keep the codes; dequantising to fp32 on the way out would inflate the
                # file ~7x and re-quantising on load would round a second time
                q = cache.latent
                meta["mla_quant"][str(L)] = {"bits": q.bits, "group_size": q.gs, "r": q.r}
                arrays[f"mla.{L}.codes"] = np.array(q.codes[:, :n], copy=False)
                arrays[f"mla.{L}.scales"] = np.array(q.scales[:, :n], copy=False)
                arrays[f"mla.{L}.biases"] = np.array(q.biases[:, :n], copy=False)
            else:
                arrays[f"mla.{L}.latent"] = np.array(cache.latent[:, :n], copy=False)
        if self.block_residual is not None:
            arrays["block_residual"] = np.array(self.block_residual, copy=False)

        np.savez(path, __meta__=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                 **arrays)

    @staticmethod
    def load(path: str | Path) -> "KimiK3State":
        z = np.load(Path(path), allow_pickle=False)
        meta = json.loads(bytes(z["__meta__"]).decode())
        if meta["format_version"] != FORMAT_VERSION:
            extra = ""
            if meta["format_version"] == 1:
                extra = (" v1 stored the MLA cache expanded through kv_b, which is not "
                         "invertible back to the latent, so it cannot be upgraded -- "
                         "re-run the prefill instead.")
            raise ValueError(
                f"unsupported state format {meta['format_version']}, expected "
                f"{FORMAT_VERSION}.{extra}")
        if meta["conv_layout"] != CANONICAL_CONV_LAYOUT:
            raise ValueError(
                f"conv state layout is {meta['conv_layout']!r}, expected "
                f"{CANONICAL_CONV_LAYOUT!r}. Refusing to load: a mismatched conv window "
                f"produces silently wrong continuations, not an error.")

        st = KimiK3State(token_position=meta["token_position"])
        for L in meta["kda_layers"]:
            st.kda_states[L] = mx.array(z[f"kda_state.{L}"])
            st.kda_conv[L] = tuple(
                conv_from_canonical(z[f"kda_conv.{L}.{w}"]) for w in ("q", "k", "v"))
        for L in meta["mla_layers"]:
            from .attention_mla import MLACache
            rot = mx.array(z[f"mla.{L}.rot"])
            qmeta = meta.get("mla_quant", {}).get(str(L))
            if qmeta is None:
                lat = mx.array(z[f"mla.{L}.latent"])
            else:
                from .mla_quant import QuantLatent
                lat = QuantLatent(rot.shape[0], rot.shape[1], qmeta["r"],
                                  qmeta["bits"], qmeta["group_size"])
                lat.codes = mx.array(z[f"mla.{L}.codes"])
                lat.scales = mx.array(z[f"mla.{L}.scales"])
                lat.biases = mx.array(z[f"mla.{L}.biases"])
            st.mla_caches[L] = MLACache(lat, rot, rot.shape[1])
        if meta["has_block_residual"]:
            st.block_residual = mx.array(z["block_residual"])
        return st


def fmt_bytes(n: int) -> str:
    for u in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or u == "GiB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} GiB"
