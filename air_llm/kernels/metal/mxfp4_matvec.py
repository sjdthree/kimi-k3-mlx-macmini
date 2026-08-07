"""M10: fused MXFP4 matvec as a JIT-compiled Metal kernel.

Why this matters more than the plan assumed
-------------------------------------------
The unfused path (decode_mxfp4_mlx -> matmul) measured 3.17 Gval/s, i.e. ~15.3 s/token of
pure decode across 16 experts x 92 layers x 3 projections -- about 2x the expert I/O cost
on the USB4 drive. It expands all 11M values of a projection into a full bf16 matrix when
a single decoded token only needs a matvec.

The fused kernel never materialises the matrix. It reads the packed bytes once (5.25 MiB
per projection instead of 21 MiB of expanded bf16), unpacks in-register, multiplies, and
accumulates in fp32. Memory traffic drops ~4x and the intermediate allocation disappears.

Numerics: unpack and scale are exact (integers and powers of two), and accumulation is
fp32 -- strictly better than the unfused bf16-weight path, so parity is against the
NumPy oracle, not against the unfused kernel.
"""
from __future__ import annotations

import mlx.core as mx

_HEADER = """
constant float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
"""

# One thread per output element out[m, n]. Each thread walks one packed row of W,
# which is also the minimum possible weight traffic: every byte is read exactly once
# per token-batch.
_SOURCE = """
    uint n = thread_position_in_grid.x;      // output feature
    uint m = thread_position_in_grid.y;      // token
    uint N = packed_shape[0];
    uint KB = packed_shape[1];               // K/2 bytes per row
    uint KS = scales_shape[1];               // K/32 scales per row
    uint K  = KB * 2;
    uint M  = x_shape[0];
    if (n >= N || m >= M) return;

    uint prow = n * KB;
    uint srow = n * KS;
    uint xrow = m * K;

    float acc = 0.0f;
    for (uint kb = 0; kb < KB; ++kb) {
        uint8_t b = packed[prow + kb];
        uint k0 = kb << 1;

        // E8M0: a float32 with exponent field e and zero mantissa is exactly 2^(e-127).
        // Both nibbles of a byte share a group (group_size 32 is even, k0 is even).
        float scale = as_type<float>((uint)(scales[srow + (k0 >> 5)]) << 23);

        uint8_t lo = b & 0x0F;
        uint8_t hi = (b >> 4) & 0x0F;
        float v0 = kE2M1[lo & 7] * scale;
        float v1 = kE2M1[hi & 7] * scale;
        if (lo & 8) { v0 = -v0; }
        if (hi & 8) { v1 = -v1; }

        acc = fma(x[xrow + k0],     v0, acc);
        acc = fma(x[xrow + k0 + 1], v1, acc);
    }
    out[m * N + n] = acc;
"""

_kernel = mx.fast.metal_kernel(
    name="mxfp4_matvec",
    input_names=["x", "packed", "scales"],
    output_names=["out"],
    header=_HEADER,
    source=_SOURCE,
    ensure_row_contiguous=True,
)


def mxfp4_matvec_fused(x: mx.array, packed: mx.array, scales: mx.array) -> mx.array:
    """y = x @ W.T, W described by (packed, scales). x [M, K] fp32 -> out [M, N] fp32.

    `packed` is [N, K/2] uint8 and `scales` is [N, K/32] uint8, exactly as stored in the
    checkpoint -- nothing is pre-expanded.
    """
    if x.dtype != mx.float32:
        x = x.astype(mx.float32)
    M = x.shape[0]
    N = packed.shape[0]
    if packed.shape[1] * 2 != x.shape[1]:
        raise ValueError(f"K mismatch: x has {x.shape[1]}, packed implies {packed.shape[1]*2}")

    tg = 64 if N >= 64 else 1
    (out,) = _kernel(
        inputs=[x, packed, scales],
        output_shapes=[(M, N)],
        output_dtypes=[mx.float32],
        grid=(N, M, 1),
        threadgroup=(tg, 1, 1),
    )
    return out


def expert_projection(x: mx.array, packed: mx.array, scales: mx.array) -> mx.array:
    """Convenience wrapper matching the shape convention used by LatentMoE."""
    squeeze = x.ndim == 1
    if squeeze:
        x = x.reshape(1, -1)
    y = mxfp4_matvec_fused(x, packed, scales)
    return y.reshape(-1) if squeeze else y
