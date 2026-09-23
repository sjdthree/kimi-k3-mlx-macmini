"""P0-3: NumPy MXFP4 reference decoder, validated bit-exactly against compressed-tensors
on real Kimi-K3 expert tensors.

Contract established from compressed_tensors 0.17.1:
  compressors/nvfp4/helpers.py  -> nibble order + E2M1 code table
  quantization/utils/mxfp_utils.py:64 -> scale is E8M0: 2**(scale_byte - 127)
"""
import json
import numpy as np
import torch
from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
from k3io import load, meta, CKPT

# ---------------------------------------------------------------- reference decoder

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def decode_mxfp4_reference(packed: np.ndarray, scales: np.ndarray,
                           group_size: int = 32) -> np.ndarray:
    """packed [R, C/2] uint8, scales [R, C/group_size] uint8 (E8M0) -> [R, C] float32.

    Nibble order: low nibble is the EVEN element, high nibble the ODD one.
    Code layout: bit3 = sign (1 => negative), bits 0-2 = index into E2M1.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi

    mag = E2M1[codes & 0x07]
    vals = np.where((codes & 0x08).astype(bool), -mag, mag)

    # E8M0: 2**(byte - 127). byte==0 -> 2**-127, representable in float32 (subnormal-free).
    exp = scales.astype(np.int32) - 127
    scale = np.ldexp(np.ones_like(exp, dtype=np.float32), exp)
    scale = np.repeat(scale, group_size, axis=1)
    return (vals * scale).astype(np.float32)


# ---------------------------------------------------------------- tests

def t_exhaustive_codes():
    """All 16 FP4 codes decode exactly as compressed-tensors does."""
    packed = np.arange(256, dtype=np.uint8).reshape(16, 16)
    scales = np.full((16, 1), 127, dtype=np.uint8)  # 2**0 == 1
    mine = decode_mxfp4_reference(packed, scales, group_size=32)
    theirs = unpack_fp4_from_uint8(torch.from_numpy(packed), 16, 32,
                                   dtype=torch.float32).numpy()
    ok = np.array_equal(mine, theirs)
    print(f"  [1] all 256 byte values / 16 FP4 codes  : {'PASS' if ok else 'FAIL'}")
    if not ok:
        bad = np.argwhere(mine != theirs)[:5]
        print("      first mismatches:", bad, mine[tuple(bad.T)], theirs[tuple(bad.T)])
    return ok


def t_exhaustive_scales():
    """All 256 E8M0 scale bytes produce the right power of two."""
    packed = np.full((256, 16), 0x22, dtype=np.uint8)   # code 2 == +1.0 in both nibbles
    scales = np.arange(256, dtype=np.uint8).reshape(256, 1)
    mine = decode_mxfp4_reference(packed, scales, group_size=32)
    want = np.ldexp(np.ones(256, dtype=np.float32), np.arange(256) - 127)[:, None]
    want = np.repeat(want, 32, axis=1)
    ok = np.array_equal(mine, want)
    print(f"  [2] all 256 E8M0 scale bytes            : {'PASS' if ok else 'FAIL'}")
    if not ok:
        d = np.argwhere(mine != want)[:5]
        print("      first mismatches:", d)
    return ok


def t_real_tensors(n=6):
    """Real expert tensors: decoder vs compressed-tensors, bit-exact."""
    allok = True
    picks = [(1, 0, "w1"), (1, 0, "w2"), (1, 0, "w3"),
             (1, 417, "w1"), (40, 895, "w2"), (92, 123, "w3")][:n]
    for L, E, w in picks:
        base = f"language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w}"
        p = load(base + ".weight_packed")
        s = load(base + ".weight_scale")
        R, Chalf = p.shape
        C = Chalf * 2
        gs = C // s.shape[1]

        mine = decode_mxfp4_reference(p, s, group_size=gs)
        ref_codes = unpack_fp4_from_uint8(torch.from_numpy(p.copy()), R, C,
                                          dtype=torch.float32).numpy()
        ref_scale = np.repeat(np.ldexp(np.ones_like(s, dtype=np.float32),
                                       s.astype(np.int32) - 127), gs, axis=1)
        theirs = ref_codes * ref_scale

        exact = np.array_equal(mine, theirs)
        allok &= exact
        nz = float((mine != 0).mean())
        print(f"  [3] L{L:>2} e{E:<3} {w}  packed{list(p.shape)} scale{list(s.shape)} "
              f"gs={gs:<3} -> {list(mine.shape)}  {'EXACT' if exact else 'MISMATCH'}"
              f"   absmax {np.abs(mine).max():.4g}  nonzero {nz:.1%}")
    return allok


def t_scale_sanity():
    """Are the decoded magnitudes plausible weights? (sanity, not correctness)"""
    base = "language_model.model.layers.1.block_sparse_moe.experts.0.w1"
    p, s = load(base + ".weight_packed"), load(base + ".weight_scale")
    d = decode_mxfp4_reference(p, s)
    print(f"  [4] decoded w1: mean|w| {np.abs(d).mean():.5f}  std {d.std():.5f}  "
          f"max {np.abs(d).max():.4f}  exact-zeros {float((d==0).mean()):.1%}")
    print(f"      scale bytes: min {s.min()} max {s.max()} "
          f"=> 2^[{int(s.min())-127}, {int(s.max())-127}]")
    return bool(np.isfinite(d).all() and np.abs(d).max() < 100)


if __name__ == "__main__":
    q = json.load(open(CKPT + "config.json"))["text_config"]["quantization_config"]
    print("quantization_config.config_groups.group_0.weights:")
    print("  ", json.dumps(q["config_groups"]["group_0"]["weights"]))
    print("   format:", q["config_groups"]["group_0"]["format"],
          "| ignore entries:", len(q.get("ignore", [])))
    print()
    r = [t_exhaustive_codes(), t_exhaustive_scales(), t_real_tensors(), t_scale_sanity()]
    print(f"\nP0-3 {'PASS' if all(r) else 'FAIL'}")
