"""M3: MXFP4 decoding for Kimi-K3 experts, MLX implementation.

Format (established bit-exactly in P0-3, see P0_RESULTS.md):
    nibble order   low nibble = even element (2i), high nibble = odd (2i+1)
    code           bit3 = sign (1 => negative), bits 0-2 = index into
                   [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    scale          E8M0, value = 2**(scale_byte - 127)
    grouping       group_size 32 along the INPUT axis (last axis of the packed tensor)

Every decoded value is exactly representable in bf16 (proven in P0-3), so `out_dtype`
may be bfloat16 with zero loss.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

_E2M1_MX = mx.array(E2M1, dtype=mx.float32)
_E2M1_NP = np.array(E2M1, dtype=np.float32)


# --------------------------------------------------------------------- reference

def decode_mxfp4_reference(packed: np.ndarray, scales: np.ndarray,
                           group_size: int = 32) -> np.ndarray:
    """Authoritative NumPy oracle. Bit-exact vs compressed-tensors 0.17.1."""
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    mag = _E2M1_NP[codes & 0x07]
    vals = np.where((codes & 0x08).astype(bool), -mag, mag)
    scale = np.ldexp(np.ones(scales.shape, dtype=np.float32),
                     scales.astype(np.int32) - 127)
    return vals * np.repeat(scale, group_size, axis=1)


# --------------------------------------------------------------------------- MLX

def decode_scale_mlx(scales: mx.array) -> mx.array:
    """E8M0 byte -> float32 power of two, without a pow() call.

    A float32 with exponent field `e` and zero mantissa is exactly 2**(e-127), so the
    decode is a bit shift. Exact for e in 1..254.

    Byte 0 (2**-127) is SUBNORMAL in fp32 (min normal is 2**-126) and **Metal flushes
    denormals to zero** -- on the GPU it decodes to 0.0 where NumPy gives 5.877e-39.
    That divergence is unavoidable on this backend, and it is unreachable in practice:
    a scan of 132.1M real scale bytes found the range 110..122 only, with zero
    occurrences of byte 0 or byte 255. `assert_scales_representable` turns that from an
    assumption into a check. Byte 255 (E8M0 NaN) maps to +inf here and likewise never
    occurs.
    """
    return (scales.astype(mx.uint32) << 23).view(mx.float32)


def assert_scales_representable(scales) -> None:
    """Guard the one case Metal cannot represent. Cheap: two reductions per tensor."""
    lo = int(scales.min())
    if lo == 0:
        raise ValueError(
            "MXFP4 scale byte 0 (2**-127) is subnormal in fp32 and is flushed to zero "
            "by Metal. This checkpoint was expected never to emit it.")


def decode_mxfp4_mlx(packed: mx.array, scales: mx.array, group_size: int = 32,
                     out_dtype: mx.Dtype = mx.float32) -> mx.array:
    """packed [R, C/2] uint8, scales [R, C/32] uint8 -> [R, C] out_dtype."""
    R, Ch = packed.shape
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    # interleave so element 2i comes from the low nibble, 2i+1 from the high nibble
    codes = mx.stack([lo, hi], axis=-1).reshape(R, Ch * 2)

    mag = _E2M1_MX[codes & 0x07]
    vals = mx.where((codes & 0x08) != 0, -mag, mag)

    scale = mx.repeat(decode_scale_mlx(scales), group_size, axis=1)
    return (vals * scale).astype(out_dtype)


def mxfp4_matvec(x: mx.array, packed: mx.array, scales: mx.array,
                 group_size: int = 32, accum_dtype: mx.Dtype = mx.float32) -> mx.array:
    """y = x @ W.T where W is the MXFP4 tensor described by (packed, scales).

    Decodes to bf16 (lossless, per P0-3) and accumulates in fp32. This is the unfused
    path; the fused Metal kernel (M10) must match it.
    """
    w = decode_mxfp4_mlx(packed, scales, group_size, out_dtype=mx.bfloat16)
    return (x.astype(mx.bfloat16) @ w.T).astype(accum_dtype)
