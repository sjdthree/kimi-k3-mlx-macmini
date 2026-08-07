"""Spec section 15: build a miniature Kimi-K3 checkpoint with the SAME operations.

    hidden size: reduced      KDA layers: 2      MLA layers: 1
    routed experts: 8         active experts: 2  shared experts: 1
    vocabulary: reduced       MXFP4 group size: 32

Every structural feature of the real model is preserved -- latent MoE, sigmoid router with
correction bias, situ activation, block attention residuals, NoPE MLA, KDA with A_log
zero-padding, MXFP4-packed experts -- at a size that runs in milliseconds. That makes it
possible to run all six execution paths and compare them, which the 1.4 TB checkpoint
cannot support inside a test loop.

Writing this required an MXFP4 *encoder*, which the project did not previously have; it is
verified here against the decoder that M3 proved bit-exact.

    python tools/make_synthetic_k3.py --out /tmp/mini-k3
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
GROUP = 32

CFG = {
    "hidden_size": 256,
    "num_hidden_layers": 3,
    "intermediate_size": 512,
    "rms_norm_eps": 1e-5,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 64,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "mla_use_nope": True,
    "mla_use_output_gate": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "moe_intermediate_size": 64,
    "routed_expert_hidden_size": 96,
    "routed_scaling_factor": 1.0,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc",
    "num_expert_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "attn_res_block_size": 2,
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "vocab_size": 512,
    "latent_moe_use_norm": True,
    "hidden_act": "situ",
    "dtype": "bfloat16",
    "model_type": "kimi_linear",
    "linear_attn_config": {
        "num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4,
        "use_full_rank_gate": True, "gate_lower_bound": -5.0,
        # 1-INDEXED, exactly like the real checkpoint (see K3_MLX_PLAN.md section 1)
        "full_attn_layers": [3], "kda_layers": [1, 2],
    },
    "quantization_config": {
        "config_groups": {"group_0": {
            "format": "mxfp4-pack-quantized",
            "targets": ["Linear"],
            "weights": {"num_bits": 4, "type": "float", "symmetric": True,
                        "strategy": "group", "group_size": GROUP,
                        "scale_dtype": "torch.uint8", "observer": "minmax",
                        "dynamic": False, "actorder": None, "block_structure": None,
                        "observer_kwargs": {}, "zp_dtype": None},
            "input_activations": None, "output_activations": None}},
        "ignore": ["lm_head"],
    },
}


# ------------------------------------------------------------------- MXFP4 encode

def encode_mxfp4(x: np.ndarray, group: int = GROUP):
    """float -> (packed uint8 [R, C/2], scales uint8 [R, C/group]).

    Scale selection mirrors compressed-tensors `generate_mx_scales`:
    e = 127 + floor(log2(amax)) - floor(log2(6)), i.e. the group's max lands in [4, 8)
    before clamping to the E2M1 grid.
    """
    R, C = x.shape
    assert C % group == 0
    g = x.reshape(R, C // group, group).astype(np.float32)

    amax = np.abs(g).max(axis=-1)
    with np.errstate(divide="ignore"):
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0))).astype(np.int32) - 2
    byte = np.clip(e + 127, 1, 254).astype(np.uint8)          # keep byte 0 unreachable
    scale = np.ldexp(np.ones_like(byte, dtype=np.float32), byte.astype(np.int32) - 127)

    q = g / scale[..., None]
    sign = np.signbit(q)
    idx = np.abs(np.abs(q)[..., None] - E2M1).argmin(axis=-1).astype(np.uint8)
    codes = (idx | (sign.astype(np.uint8) << 3)).reshape(R, C)

    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    return packed, byte.astype(np.uint8)


def decode_mxfp4(packed, scales, group: int = GROUP):
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = lo, hi
    mag = E2M1[codes & 0x07]
    vals = np.where((codes & 0x08).astype(bool), -mag, mag)
    sc = np.ldexp(np.ones(scales.shape, np.float32), scales.astype(np.int32) - 127)
    return vals * np.repeat(sc, group, axis=1)


# ----------------------------------------------------------------- safetensors io

def bf16(a: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit pattern (round-to-nearest-even)."""
    u = a.astype(np.float32).view(np.uint32)
    rounded = ((u >> 16) & 1) + 0x7FFF
    return ((u + rounded) >> 16).astype(np.uint16)


def write_safetensors(path: Path, tensors: dict[str, tuple[np.ndarray, str]]):
    header, offset, blobs = {}, 0, []
    for name, (arr, dtype) in tensors.items():
        b = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(arr.shape),
                        "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    hj = json.dumps(header).encode()
    hj += b" " * ((8 - len(hj) % 8) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)
    return header


# ---------------------------------------------------------------------- generator

def build(out: Path, seed: int = 0):
    rng = np.random.default_rng(seed)
    c = CFG
    H = c["hidden_size"]
    lac = c["linear_attn_config"]
    nh, hd = lac["num_heads"], lac["head_dim"]
    proj = nh * hd
    latent = c["routed_expert_hidden_size"]
    inter = c["moe_intermediate_size"]
    mla_layers = {i - 1 for i in lac["full_attn_layers"]}

    def W(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    out.mkdir(parents=True, exist_ok=True)
    weight_map, all_headers = {}, {}

    for L in range(c["num_hidden_layers"]):
        t: dict[str, tuple[np.ndarray, str]] = {}
        p = f"language_model.model.layers.{L}."

        for n in ("input_layernorm", "post_attention_layernorm",
                  "self_attention_res_norm", "mlp_res_norm"):
            t[p + n + ".weight"] = (bf16(np.ones(H, np.float32) + W(H, s=0.01)), "BF16")
        for n in ("self_attention_res_proj", "mlp_res_proj"):
            t[p + n + ".weight"] = (bf16(W(1, H)), "BF16")

        if L in mla_layers:
            qhd = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]
            t[p + "self_attn.q_a_proj.weight"] = (bf16(W(c["q_lora_rank"], H)), "BF16")
            t[p + "self_attn.q_a_layernorm.weight"] = (bf16(np.ones(c["q_lora_rank"], np.float32)), "BF16")
            t[p + "self_attn.q_b_proj.weight"] = (bf16(W(c["num_attention_heads"] * qhd, c["q_lora_rank"])), "BF16")
            t[p + "self_attn.kv_a_proj_with_mqa.weight"] = (bf16(W(c["kv_lora_rank"] + c["qk_rope_head_dim"], H)), "BF16")
            t[p + "self_attn.kv_a_layernorm.weight"] = (bf16(np.ones(c["kv_lora_rank"], np.float32)), "BF16")
            t[p + "self_attn.kv_b_proj.weight"] = (bf16(W(c["num_attention_heads"] * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"])), "BF16")
            t[p + "self_attn.g_proj.weight"] = (bf16(W(c["num_attention_heads"] * c["v_head_dim"], H)), "BF16")
            t[p + "self_attn.o_proj.weight"] = (bf16(W(H, c["num_attention_heads"] * c["v_head_dim"])), "BF16")
        else:
            for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
                t[p + f"self_attn.{n}.weight"] = (bf16(W(proj, H)), "BF16")
            t[p + "self_attn.o_proj.weight"] = (bf16(W(H, proj)), "BF16")
            for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
                t[p + f"self_attn.{n}.weight"] = (W(proj, 1, lac["short_conv_kernel_size"], s=0.3), "F32")
            t[p + "self_attn.f_a_proj.weight"] = (bf16(W(hd, H)), "BF16")
            t[p + "self_attn.f_b_proj.weight"] = (bf16(W(proj, hd)), "BF16")
            t[p + "self_attn.b_proj.weight"] = (bf16(W(nh, H)), "BF16")
            t[p + "self_attn.o_norm.weight"] = (np.ones(hd, np.float32), "F32")
            # A_log: real length is head_dim with only the first num_heads non-zero
            a = np.zeros(hd, np.float32)
            a[:nh] = np.log(rng.uniform(1, 16, nh)).astype(np.float32)
            t[p + "self_attn.A_log"] = (a, "F32")
            t[p + "self_attn.dt_bias"] = (W(proj, s=1.0).ravel().astype(np.float32), "F32")

        if L < c["first_k_dense_replace"]:
            for n in ("gate_proj", "up_proj"):
                t[p + f"mlp.{n}.weight"] = (bf16(W(c["intermediate_size"], H)), "BF16")
            t[p + "mlp.down_proj.weight"] = (bf16(W(H, c["intermediate_size"])), "BF16")
        else:
            m = p + "block_sparse_moe."
            t[m + "gate.weight"] = (bf16(W(c["num_experts"], H)), "BF16")
            t[m + "gate.e_score_correction_bias"] = (W(c["num_experts"], s=0.1).astype(np.float32), "F32")
            t[m + "routed_expert_down_proj.weight"] = (bf16(W(latent, H)), "BF16")
            t[m + "routed_expert_up_proj.weight"] = (bf16(W(H, latent)), "BF16")
            t[m + "routed_expert_norm.weight"] = (bf16(np.ones(latent, np.float32)), "BF16")
            sh = inter * c["num_shared_experts"]
            for n in ("gate_proj", "up_proj"):
                t[m + f"shared_experts.{n}.weight"] = (bf16(W(sh, H)), "BF16")
            t[m + "shared_experts.down_proj.weight"] = (bf16(W(H, sh)), "BF16")
            for e in range(c["num_experts"]):
                for w, shape in (("w1", (inter, latent)), ("w3", (inter, latent)),
                                 ("w2", (latent, inter))):
                    pk, sc = encode_mxfp4(W(*shape, s=0.05))
                    t[f"{m}experts.{e}.{w}.weight_packed"] = (pk, "U8")
                    t[f"{m}experts.{e}.{w}.weight_scale"] = (sc, "U8")

        fn = f"model-{L+1:05d}-of-{c['num_hidden_layers']+1:05d}.safetensors"
        all_headers[fn] = write_safetensors(out / fn, t)
        for k in t:
            weight_map[k] = fn

    # model-level tensors
    t = {
        "language_model.model.embed_tokens.weight": (bf16(W(c["vocab_size"], H, s=0.05)), "BF16"),
        "language_model.lm_head.weight": (bf16(W(c["vocab_size"], H, s=0.05)), "BF16"),
        "language_model.model.norm.weight": (bf16(np.ones(H, np.float32)), "BF16"),
        "language_model.model.output_attn_res_norm.weight": (bf16(np.ones(H, np.float32)), "BF16"),
        "language_model.model.output_attn_res_proj.weight": (bf16(W(1, H)), "BF16"),
    }
    fn = f"model-{c['num_hidden_layers']+1:05d}-of-{c['num_hidden_layers']+1:05d}.safetensors"
    all_headers[fn] = write_safetensors(out / fn, t)
    for k in t:
        weight_map[k] = fn

    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map}))
    (out / "config.json").write_text(json.dumps(
        {"model_type": "kimi_k3", "architectures": ["KimiK3ForConditionalGeneration"],
         "dtype": "bfloat16", "text_config": CFG}, indent=1))

    print(f"wrote {len(weight_map)} tensors across {len(all_headers)} shards -> {out}")
    return weight_map


def selftest():
    """The encoder must round-trip through the M3-verified decoder."""
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((64, 96)) * 0.05).astype(np.float32)
    pk, sc = encode_mxfp4(x)
    back = decode_mxfp4(pk, sc)
    rel = np.abs(back - x).max() / np.abs(x).max()
    # 4-bit with a shared exponent per 32 values: ~15% worst-case element error is normal
    assert rel < 0.30, rel
    assert sc.min() >= 1 and sc.max() <= 254
    # re-encoding a decoded tensor must be a fixed point
    pk2, sc2 = encode_mxfp4(back)
    assert np.array_equal(pk, pk2) and np.array_equal(sc, sc2), "encode is not idempotent"
    print(f"  encoder self-test OK (max rel err {rel:.3f}, idempotent)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("/tmp/mini-k3"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    selftest()
    build(a.out, a.seed)
