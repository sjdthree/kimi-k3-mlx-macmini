"""Optional quantised storage for the MLA latent cache. OFF by default.

WHY THIS IS OFF BY DEFAULT
--------------------------
Latent caching (see attention_mla.py) already removed 53.3x losslessly, which took the
MLA cache from 6.45 GiB to 0.12 GiB at the 2.3k context this model actually runs at.
Quantising on top of that saves ~90 MiB -- nothing -- and it is the first change in this
file's history that is not exactly reversible. It earns its place only in a regime we
cannot currently reach:

    context     fp32 latent    4-bit latent
      8 k          0.42 GiB       0.07 GiB
     32 k          1.69 GiB       0.26 GiB
    128 k          6.75 GiB       1.03 GiB

So: implemented, gated, measured, and left off. `mla_cache_bits` on the config turns it
on when a long-context run needs it.

WHY THE MLA LATENT IS A HARDER TARGET THAN A NORMAL KV CACHE
------------------------------------------------------------
The usual argument for aggressive KV quantisation is that attention only needs an
approximate inner product, and per-head key error averages out under softmax. That
argument is weaker here, twice over:

  * every one of the 96 heads reads the SAME latent, through kv_b. Error in c is not
    independent across heads, so it does not average -- it is a common-mode perturbation
    applied 96 times.
  * kv_b produces both the key half and the VALUE half. Value error does not pass through
    a softmax at all; it lands directly on the output.

That is why the gates in tests/test_mla_quant.py score end-to-end token agreement rather
than reconstruction cosine alone. A latent can reconstruct at cos 0.995 and still move
greedy tokens.

TRANSIENT COST
--------------
Reading dequantises the valid prefix of ONE layer. Layers are streamed serially
(model.py forward_token), so the fp32 temporary is one layer's latent -- 64 MiB at 32k,
256 MiB at 128k -- not the whole 24-layer cache. The storage saving is what persists.
"""
from __future__ import annotations

import mlx.core as mx

# MLX's quantised matmul kernels accept these; anything else raises rather than silently
# reinterpreting the buffer.
VALID_BITS = (2, 3, 4, 5, 6, 8)
VALID_GROUPS = (32, 64, 128)


def pick_group_size(r: int, preferred: int = 64) -> int:
    """Largest supported group size that divides the latent width."""
    for g in sorted(VALID_GROUPS, reverse=True):
        if g <= preferred and r % g == 0:
            return g
    for g in sorted(VALID_GROUPS):
        if r % g == 0:
            return g
    raise ValueError(
        f"kv_lora_rank {r} is not divisible by any supported group size {VALID_GROUPS}; "
        f"quantised MLA caching cannot be used for this checkpoint.")


class QuantLatent:
    """Grouped affine quantised [B, capacity, r] store with in-place appends.

    Mirrors the plain-array buffer's interface (`write`, `grow`, `read`, `bytes_per_token`)
    so MLACache does not need to branch on which one it holds.
    """

    def __init__(self, B: int, cap: int, r: int, bits: int, group_size: int):
        if bits not in VALID_BITS:
            raise ValueError(f"mla_cache_bits must be one of {VALID_BITS}, got {bits}")
        self.B, self.r, self.bits, self.gs = B, r, bits, group_size
        self.codes = mx.zeros((B, cap, r * bits // 32), dtype=mx.uint32)
        self.scales = mx.zeros((B, cap, r // group_size), dtype=mx.float32)
        self.biases = mx.zeros((B, cap, r // group_size), dtype=mx.float32)

    @property
    def capacity(self) -> int:
        return self.codes.shape[1]

    def grow(self, cap: int, keep: int) -> None:
        for name in ("codes", "scales", "biases"):
            old = getattr(self, name)
            new = mx.zeros((old.shape[0], cap, old.shape[2]), dtype=old.dtype)
            if keep:
                new[:, :keep] = old[:, :keep]
            setattr(self, name, new)

    def write(self, start: int, x: mx.array) -> None:
        """Quantise T new rows and place them at [start, start+T)."""
        B, T, r = x.shape
        q, s, b = mx.quantize(x.reshape(B * T, r).astype(mx.float32),
                              group_size=self.gs, bits=self.bits)
        end = start + T
        self.codes[:, start:end] = q.reshape(B, T, -1)
        self.scales[:, start:end] = s.reshape(B, T, -1)
        self.biases[:, start:end] = b.reshape(B, T, -1)

    def read(self, length: int) -> mx.array:
        """Dequantise the valid prefix to fp32 [B, length, r]."""
        B = self.B
        d = mx.dequantize(self.codes[:, :length].reshape(B * length, -1),
                          self.scales[:, :length].reshape(B * length, -1),
                          self.biases[:, :length].reshape(B * length, -1),
                          group_size=self.gs, bits=self.bits)
        return d.reshape(B, length, self.r).astype(mx.float32)

    def bytes_per_token(self) -> float:
        ngroups = self.r // self.gs
        return (self.r * self.bits / 8) + ngroups * (self.scales.itemsize
                                                     + self.biases.itemsize)

    def nbytes_capacity(self) -> int:
        return self.codes.nbytes + self.scales.nbytes + self.biases.nbytes
