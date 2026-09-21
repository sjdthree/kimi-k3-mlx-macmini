"""Checkpoint configuration; no MLX or PyTorch imports.

Extracted from the Moonshot-derived reference adapter; see NOTICE and the
Kimi K3 license in third_party/LICENSE.kimi-k3.
"""
import json
import os
from pathlib import Path
from dataclasses import dataclass, field

@dataclass
class K3Config:
    hidden_size: int
    num_hidden_layers: int
    rms_norm_eps: float
    num_attention_heads: int
    num_key_value_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    mla_use_output_gate: bool
    num_experts: int
    num_experts_per_token: int
    num_shared_experts: int
    moe_intermediate_size: int
    routed_expert_hidden_size: int
    routed_scaling_factor: float
    moe_renormalize: bool
    first_k_dense_replace: int
    intermediate_size: int
    attn_res_block_size: int
    situ_beta: float
    situ_linear_beta: float
    kda_num_heads: int
    kda_head_dim: int
    kda_conv_size: int
    kda_gate_lower_bound: float
    mla_layers: list[int] = field(default_factory=list)
    kda_layers: list[int] = field(default_factory=list)

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def is_kda(self, i: int) -> bool:
        return i in self.kda_layers

    @staticmethod
    def load(path: Path | None = None) -> "K3Config":
        path = Path(path) if path is not None else Path(os.environ.get("K3_CKPT", "Kimi-K3")) / "config.json"
        t = json.loads(path.read_text())["text_config"]
        lac = t["linear_attn_config"]
        n = t["num_hidden_layers"]
        # config layer lists are 1-INDEXED (see K3_MLX_PLAN.md section 1 row 3)
        mla = sorted(i - 1 for i in lac["full_attn_layers"] if i - 1 < n)
        return K3Config(
            hidden_size=t["hidden_size"], num_hidden_layers=n,
            rms_norm_eps=t["rms_norm_eps"],
            num_attention_heads=t["num_attention_heads"],
            num_key_value_heads=t["num_key_value_heads"],
            q_lora_rank=t["q_lora_rank"], kv_lora_rank=t["kv_lora_rank"],
            qk_nope_head_dim=t["qk_nope_head_dim"], qk_rope_head_dim=t["qk_rope_head_dim"],
            v_head_dim=t["v_head_dim"], mla_use_output_gate=t["mla_use_output_gate"],
            num_experts=t["num_experts"], num_experts_per_token=t["num_experts_per_token"],
            num_shared_experts=t["num_shared_experts"],
            moe_intermediate_size=t["moe_intermediate_size"],
            routed_expert_hidden_size=t["routed_expert_hidden_size"],
            routed_scaling_factor=t["routed_scaling_factor"],
            moe_renormalize=t["moe_renormalize"],
            first_k_dense_replace=t["first_k_dense_replace"],
            intermediate_size=t["intermediate_size"],
            attn_res_block_size=t["attn_res_block_size"],
            situ_beta=t["activation_situ_beta"],
            situ_linear_beta=t["activation_situ_linear_beta"],
            kda_num_heads=lac["num_heads"], kda_head_dim=lac["head_dim"],
            kda_conv_size=lac["short_conv_kernel_size"],
            kda_gate_lower_bound=lac["gate_lower_bound"],
            mla_layers=mla, kda_layers=[i for i in range(n) if i not in set(mla)],
        )


