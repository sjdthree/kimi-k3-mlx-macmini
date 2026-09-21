"""k3ref: Kimi-K3 reference implementation, torch/CPU, fp32.

Derivative work of `modeling_kimi_linear.py` as shipped in the Kimi-K3 checkpoint,
Copyright (c) 2026 Moonshot AI, used under the Kimi K3 License. The full license text
is in third_party/LICENSE.kimi-k3; see NOTICE for the attribution trail.

This is NOT a reimplementation. It is the shipped `modeling_kimi_linear.py` semantics with
the two CUDA-only kernels replaced by their pure-torch equivalents:
  flash_attn_varlen_func -> eager_attention_forward (already in the shipped file)
  fla.ops.kda (Triton)   -> fla/ops/kda/naive.py::naive_recurrent_kda (upstream, unmodified)
and MXFP4 decoded by the P0-3 decoder (bit-exact vs compressed-tensors).

It is the parity oracle for the MLX implementation (K3_MLX_PLAN.md section 3).

Source anchors (checkpoint's own modeling_kimi_linear.py):
  KimiRMSNorm:232  SituAndMul:75  KimiBlockSparseMLP:262  KimiMLP:294
  eager_attention_forward:311  KimiMLAAttention:405  KimiDeltaAttention:543
  KimiMoEGate:703  KimiSparseMoeBlock:815  moe_infer:841
  KimiDecoderLayer._forward_attn_residual:973  _apply_attn_res:1075
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .weights import CKPT, WeightStore, default_store

# upstream naive KDA, imported by path (fla/__init__ needs triton, which Metal cannot have)
_FLA_NAIVE = (Path(__file__).resolve().parent.parent / "third_party" / "kda_naive.py")
_spec = importlib.util.spec_from_file_location("kda_naive", _FLA_NAIVE)
kda_naive = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kda_naive)


# --------------------------------------------------------------------------- config

from air_llm.models.kimi_k3.config import K3Config


# ------------------------------------------------------------------------------ ops

def rmsnorm(x, w, eps):
    """KimiRMSNorm:232"""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps))


def situ(gate_up, beta, linear_beta):
    """SituAndMul:75 -- beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)"""
    d = gate_up.shape[-1] // 2
    g, u = gate_up[..., :d].float(), gate_up[..., d:].float()
    a = beta * torch.tanh(g / beta) * torch.sigmoid(g)
    if linear_beta is not None:
        u = linear_beta * torch.tanh(u / linear_beta)
    return a * u


def l2norm(x, eps=1e-6):
    """fla/modules/l2norm.py:105 -- x / sqrt(sum(x^2) + eps)"""
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def gated_rmsnorm(x, w, g, eps):
    """fla/modules/fused_norm_gate.py:104 -- y = x*rstd*w*sigmoid(g)"""
    return (x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)) * w * torch.sigmoid(g)


def short_conv(x, w, k, state=None):
    """Causal depthwise conv (kernel k, no bias) then silu. x [B,T,C], w [C,1,k]."""
    B, T, C = x.shape
    xt = x.transpose(1, 2)
    pad = state if state is not None else xt.new_zeros(B, C, k - 1)
    xp = torch.cat([pad, xt], dim=-1)
    y = F.conv1d(xp, w, groups=C)
    return F.silu(y.transpose(1, 2)), xp[..., -(k - 1):]


def apply_attn_res(prefix_sum, block_residual, proj_w, norm_w, eps):
    """_apply_attn_res:1075. Softmax attention over block checkpoints + current state.

    prefix_sum     [N, H]      block_residual [N, B, H]      proj_w [1, H]      norm_w [H]
    """
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1).float()
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    score_w = norm_w.float() * proj_w.squeeze(0).float()
    probs = (k * score_w).sum(-1).softmax(-1).unsqueeze(1)
    return torch.matmul(probs, v).squeeze(1)


# ------------------------------------------------------------------------ attention

class KDAAttention:
    """KimiDeltaAttention:543. Semantics resolved in P0-2, see P0_RESULTS.md."""

    def __init__(self, cfg: K3Config, layer: int, ws: WeightStore):
        self.cfg, self.L = cfg, layer
        p = f"language_model.model.layers.{layer}.self_attn."
        g = ws.get
        self.q_w, self.k_w, self.v_w = g(p + "q_proj.weight"), g(p + "k_proj.weight"), g(p + "v_proj.weight")
        self.g_w, self.o_w = g(p + "g_proj.weight"), g(p + "o_proj.weight")
        self.qc, self.kc, self.vc = g(p + "q_conv1d.weight"), g(p + "k_conv1d.weight"), g(p + "v_conv1d.weight")
        self.fa, self.fb, self.b_w = g(p + "f_a_proj.weight"), g(p + "f_b_proj.weight"), g(p + "b_proj.weight")
        self.o_norm_w = g(p + "o_norm.weight")
        self.dt_bias = g(p + "dt_bias")
        A = g(p + "A_log")
        H = cfg.kda_num_heads
        assert A.shape == (cfg.kda_head_dim,) and A[H:].abs().max() == 0, \
            f"A_log layout changed at layer {layer}: {tuple(A.shape)}"
        self.A_log = A[:H]                      # zero-padded [96] per value head (P0-2)

    def __call__(self, x, state=None, conv=None):
        c, B, T = self.cfg, x.shape[0], x.shape[1]
        H, D, K = c.kda_num_heads, c.kda_head_dim, c.kda_conv_size
        q, cq = short_conv(x @ self.q_w.T, self.qc, K, None if conv is None else conv[0])
        k, ck = short_conv(x @ self.k_w.T, self.kc, K, None if conv is None else conv[1])
        v, cv = short_conv(x @ self.v_w.T, self.vc, K, None if conv is None else conv[2])

        q, k = l2norm(q.view(B, T, H, D)), l2norm(k.view(B, T, H, D))
        v = v.view(B, T, H, D)

        g_raw = ((x @ self.fa.T) @ self.fb.T).view(B, T, H, D) + self.dt_bias.view(H, D)
        g = c.kda_gate_lower_bound * torch.sigmoid(self.A_log.view(H, 1).exp() * g_raw)
        beta = torch.sigmoid(x @ self.b_w.T)

        o, new_state = kda_naive.naive_recurrent_kda(
            q=q, k=k, v=v, g=g, beta=beta, scale=D ** -0.5,
            initial_state=state, output_final_state=True)

        og = (x @ self.g_w.T).view(B, T, H, D)
        o = gated_rmsnorm(o, self.o_norm_w, og, c.rms_norm_eps)
        return (o.reshape(B, T, H * D) @ self.o_w.T), new_state, (cq, ck, cv)


class MLAAttention:
    """KimiMLAAttention:405. NoPE: rotary_emb is None and no rotation is applied."""

    def __init__(self, cfg: K3Config, layer: int, ws: WeightStore):
        self.cfg, self.L = cfg, layer
        p = f"language_model.model.layers.{layer}.self_attn."
        g = ws.get
        self.q_a, self.q_b = g(p + "q_a_proj.weight"), g(p + "q_b_proj.weight")
        self.q_an = g(p + "q_a_layernorm.weight")
        self.kv_a, self.kv_b = g(p + "kv_a_proj_with_mqa.weight"), g(p + "kv_b_proj.weight")
        self.kv_an = g(p + "kv_a_layernorm.weight")
        self.g_w, self.o_w = g(p + "g_proj.weight"), g(p + "o_proj.weight")

    def __call__(self, x, cache=None):
        c = self.cfg
        B, T, _ = x.shape
        nh, qhd, vhd = c.num_attention_heads, c.q_head_dim, c.v_head_dim

        q = (rmsnorm(x @ self.q_a.T, self.q_an, c.rms_norm_eps) @ self.q_b.T)
        q = q.view(B, T, nh, qhd).transpose(1, 2)
        q_pass, q_rot = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)

        ckv = x @ self.kv_a.T
        k_pass, k_rot = ckv.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        k_pass = (rmsnorm(k_pass, self.kv_an, c.rms_norm_eps) @ self.kv_b.T)
        k_pass = k_pass.view(B, T, nh, c.qk_nope_head_dim + vhd).transpose(1, 2)
        k_pass, v = k_pass.split([c.qk_nope_head_dim, vhd], dim=-1)
        k_rot = k_rot.view(B, 1, T, c.qk_rope_head_dim).expand(*k_pass.shape[:-1], -1)

        key = torch.cat((k_pass, k_rot), dim=-1)
        query = torch.cat((q_pass, q_rot), dim=-1)

        if cache is not None and cache[0] is not None:
            key = torch.cat([cache[0], key], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (key, v)

        scores = torch.einsum("bhqd,bhkd->bhqk", query, key) * (qhd ** -0.5)
        kl = key.shape[2]
        causal = torch.full((T, kl), float("-inf")).triu(kl - T + 1)
        scores = scores + causal
        probs = F.softmax(scores, dim=-1, dtype=torch.float32)
        o = torch.einsum("bhqk,bhkd->bhqd", probs, v).transpose(1, 2).reshape(B, T, nh * vhd)

        if c.mla_use_output_gate:
            o = o * torch.sigmoid(x @ self.g_w.T)
        return (o @ self.o_w.T), new_cache


# ------------------------------------------------------------------------------ MoE

@dataclass
class RouterResult:
    expert_indices: torch.Tensor      # [N, top_k]
    expert_weights: torch.Tensor      # [N, top_k]
    raw_scores: torch.Tensor          # [N, num_experts]  sigmoid(logits)
    corrected_scores: torch.Tensor    # raw + e_score_correction_bias


class LatentMoE:
    """KimiSparseMoeBlock:815 with the latent wrapper (K3_MLX_PLAN.md section 1 row 2)."""

    def __init__(self, cfg: K3Config, layer: int, ws: WeightStore):
        self.cfg, self.L, self.ws = cfg, layer, ws
        p = f"language_model.model.layers.{layer}.block_sparse_moe."
        g = ws.get
        self.gate_w = g(p + "gate.weight")
        self.gate_b = g(p + "gate.e_score_correction_bias")
        self.down = g(p + "routed_expert_down_proj.weight")
        self.up = g(p + "routed_expert_up_proj.weight")
        self.norm_w = g(p + "routed_expert_norm.weight")
        self.sh_gate = g(p + "shared_experts.gate_proj.weight")
        self.sh_up = g(p + "shared_experts.up_proj.weight")
        self.sh_down = g(p + "shared_experts.down_proj.weight")

    def route(self, x) -> RouterResult:
        """KimiMoEGate:703. num_expert_group=1 => no grouping branch."""
        c = self.cfg
        flat = x.reshape(-1, x.shape[-1]).float()
        scores = torch.sigmoid(flat @ self.gate_w.float().T)
        corrected = scores + self.gate_b.unsqueeze(0)
        idx = torch.topk(corrected, k=c.num_experts_per_token, dim=-1, sorted=False)[1]
        w = scores.gather(1, idx)
        if c.num_experts_per_token > 1 and c.moe_renormalize:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return RouterResult(idx, w * c.routed_scaling_factor, scores, corrected)

    def expert_forward(self, h, e: int):
        """One routed expert in latent space. Streams its MXFP4 weights, then drops them."""
        w1 = self.ws.expert(self.L, e, "w1")
        w3 = self.ws.expert(self.L, e, "w3")
        act = situ(torch.cat([h @ w1.T, h @ w3.T], dim=-1),
                   self.cfg.situ_beta, self.cfg.situ_linear_beta)
        del w1, w3
        w2 = self.ws.expert(self.L, e, "w2")
        return act @ w2.T

    def shared_forward(self, x):
        """KimiMLP:294 on the ORIGINAL hidden, not the latent."""
        gu = torch.cat([x @ self.sh_gate.T, x @ self.sh_up.T], dim=-1)
        return situ(gu, self.cfg.situ_beta, self.cfg.situ_linear_beta) @ self.sh_down.T

    def __call__(self, x, return_router=False):
        c = self.cfg
        identity = x
        orig = x.shape
        r = self.route(x)
        flat = x.reshape(-1, orig[-1])

        h = flat @ self.down.T                                  # hidden -> latent
        y = torch.zeros(flat.shape[0], c.routed_expert_hidden_size, dtype=torch.float32)

        # serial expert streaming, accumulating in top-k order (matches moe_infer's sum)
        for e in sorted(set(r.expert_indices.reshape(-1).tolist())):
            mask = (r.expert_indices == e)
            tok, slot = mask.nonzero(as_tuple=True)
            out = self.expert_forward(h[tok], e)
            y.index_add_(0, tok, out * r.expert_weights[tok, slot].unsqueeze(-1))

        y = rmsnorm(y, self.norm_w, c.rms_norm_eps)             # latent_moe_use_norm
        y = (y @ self.up.T).view(*orig)                         # latent -> hidden
        y = y + self.shared_forward(identity)
        return (y, r) if return_router else y


class DenseMLP:
    """Layer 0 only (first_k_dense_replace=1)."""

    def __init__(self, cfg: K3Config, layer: int, ws: WeightStore):
        self.cfg = cfg
        p = f"language_model.model.layers.{layer}.mlp."
        self.gate, self.up, self.down = ws.get(p + "gate_proj.weight"), ws.get(p + "up_proj.weight"), ws.get(p + "down_proj.weight")

    def __call__(self, x, return_router=False):
        gu = torch.cat([x @ self.gate.T, x @ self.up.T], dim=-1)
        y = situ(gu, self.cfg.situ_beta, self.cfg.situ_linear_beta) @ self.down.T
        return (y, None) if return_router else y


# ---------------------------------------------------------------------- full layer

class DecoderLayer:
    """KimiDecoderLayer._forward_attn_residual:973"""

    def __init__(self, cfg: K3Config, layer: int, ws: WeightStore):
        self.cfg, self.L = cfg, layer
        p = f"language_model.model.layers.{layer}."
        g = ws.get
        self.attn = KDAAttention(cfg, layer, ws) if cfg.is_kda(layer) else MLAAttention(cfg, layer, ws)
        self.is_kda = cfg.is_kda(layer)
        self.mlp = (DenseMLP(cfg, layer, ws) if layer < cfg.first_k_dense_replace
                    else LatentMoE(cfg, layer, ws))
        self.in_norm = g(p + "input_layernorm.weight")
        self.post_norm = g(p + "post_attention_layernorm.weight")
        self.attn_res_norm = g(p + "self_attention_res_norm.weight")
        self.attn_res_proj = g(p + "self_attention_res_proj.weight")
        self.mlp_res_norm = g(p + "mlp_res_norm.weight")
        self.mlp_res_proj = g(p + "mlp_res_proj.weight")

    def __call__(self, x, block_residual, state=None, conv=None, cache=None, diag=None):
        c = self.cfg
        B, T, H = x.shape
        eps = c.rms_norm_eps
        prefix_sum = x

        if block_residual is not None and block_residual.shape[1] > 0:
            x = apply_attn_res(prefix_sum.view(-1, H), block_residual,
                               self.attn_res_proj, self.attn_res_norm, eps).view(B, T, H)

        if self.L % c.attn_res_block_size == 0:
            block_residual = torch.cat(
                [block_residual, prefix_sum.view(-1, H).unsqueeze(1)], dim=1)
            prefix_sum = None

        x = rmsnorm(x, self.in_norm, eps)
        if self.is_kda:
            x, state, conv = self.attn(x, state=state, conv=conv)
        else:
            x, cache = self.attn(x, cache=cache)

        prefix_sum = x if prefix_sum is None else prefix_sum + x

        x = apply_attn_res(prefix_sum.view(-1, H), block_residual,
                           self.mlp_res_proj, self.mlp_res_norm, eps).view(B, T, H)
        x = rmsnorm(x, self.post_norm, eps)
        x, router = self.mlp(x, return_router=True)

        prefix_sum = prefix_sum + x
        if diag is not None:
            diag.update(router=router, attn_out_shape=tuple(x.shape))
        return prefix_sum, block_residual, state, conv, cache
