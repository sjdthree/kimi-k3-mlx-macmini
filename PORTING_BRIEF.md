> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Engineering Brief: Port AirLLM's CUDA-only Kimi-K3 path to Apple Silicon (Metal/MLX) and AMD (ROCm)

## Objective
Make **Kimi-K3 (2.8T MoE, 1.56 TB MXFP4 checkpoint)** run under **AirLLM's per-expert streaming** on
non-NVIDIA hardware — specifically **Apple Silicon (Metal/MLX)** and **AMD GPUs** — matching CUDA
output within numerical tolerance. Speed is secondary; K3 streaming is disk-bound (~minutes/token) by
design. The deliverable is *correctness + fitting in a low-memory streaming budget*, not throughput.

## Background / why it doesn't run today
K3 is CUDA-locked in **its own remote modeling code** (not just in AirLLM). Three independent hard
dependencies, all verified against the checkpoint at `/Volumes/LLMModels/Kimi-K3`:

1. **Flash-Attention 2 is force-enabled.** `modeling_kimi_linear.py:1110-1119` overrides any requested
   attention impl to `flash_attention_2` and imports `from flash_attn import flash_attn_varlen_func`
   (`modeling_kimi_k3.py:52-54`). `flash-attn` is NVIDIA/CUDA-only. Used by the **24 full-attention
   (MLA) layers**.
2. **KDA linear attention needs Triton.** `modeling_kimi_linear.py:48` imports
   `from fla.ops.kda import chunk_kda, fused_recurrent_kda` (the `flash-linear-attention` lib). These
   are **Triton** kernels (CUDA; no Metal backend, limited/again-CUDA ROCm). Used by the **69 KDA
   linear-attention layers**.
3. **MXFP4 on-GPU dequant.** Experts ship as `weight_packed` (uint8, 4-bit packed) + `weight_scale`
   (uint8), `group_size=32`, decoded on the GPU under CUDA assumptions
   (`compressed-tensors` / `mxfp4-pack-quantized`, see `config.json` quantization_config).

Also: AirLLM's macOS dispatch (`airllm/auto_model.py`) unconditionally returns `AirLLMLlamaMlx` on
Darwin — it never reaches the `AirLLMKimiK3` class, and that class's forward pass runs the CUDA remote
code anyway. So there is **no Mac code path to K3 today**, patched or not.

## Verified K3 architecture (what a port must reproduce)
- **93 decoder layers.** Layer 0 dense (`first_k_dense_replace=1`); layers 1–92 are MoE.
- **Attention is mixed** (from `config.json text_config.linear_attn_config`):
  - **24 MLA full-attention layers** at indices `[4,8,12,...,92,93]` — DeepSeek-style **Multi-head
    Latent Attention** with LoRA projections: `q_lora_rank=1536`, `kv_lora_rank=512`,
    `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`, `mla_use_nope=true`,
    `mla_use_output_gate=true`. Tensors: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`,
    `q_a_layernorm`, `kv_a_layernorm`.
  - **69 KDA linear-attention layers** — **Kimi Delta Attention** (gated delta-rule / linear attn):
    `num_heads=96`, `head_dim=128`, `short_conv_kernel_size=4`, `use_full_rank_gate=true`,
    `gate_lower_bound=-5.0`. Tensors: `f_a_proj`, `f_b_proj`, `A_log`, `dt_bias`, `b_proj`,
    `{q,k,v}_conv1d` (short conv), `o_norm`, plus `g_proj`/`o_proj`.
- **MoE:** `num_experts=896`, `num_experts_per_token=16`, `num_shared_experts=2`,
  `moe_intermediate_size=3072`, `routed_expert_hidden_size=3584`, router `sigmoid` +
  `noaux_tc` grouped top-k, `routed_scaling_factor=1.0`. Experts (`w1/w2/w3`) are **MXFP4**;
  **shared experts, attention, gate, lm_head are NOT quantized** (bf16) — see
  `quantization_config.ignore` in `config.json`.
- **hidden_size=7168, vocab=163840, max_position=1048576**, bf16, `rms_norm_eps=1e-5`,
  `hidden_act="situ"` (custom activation — verify its definition in the remote code).
- **Multimodal:** MoonViT vision tower (27 blocks) + `mm_projector`; the LM decoder lives under
  `language_model.model.*` (see `airllm/airllm_kimi_k3.py` layer_names_dict). Text-only inference can
  keep vision resident/unused, but the projector/tower tensors are present.
- Extra per-layer residual modules outside the usual block: `self_attention_res_{proj,norm}`,
  `mlp_res_{proj,norm}`, and top-level `output_attn_res_{proj,norm}` — must be implemented, not dropped.
- **Requires `transformers` 4.56.x**; remote code fails on 5.x.

## Work streams
Deliver **correctness-first** (a slow, numerically-correct reference) before any kernel optimization.

### WS0 — Device/backend abstraction in AirLLM
- Remove hardcoded `.cuda()` / `device="cuda"` assumptions; add a device layer (`mps`/`cpu` for Apple,
  `cuda`(ROCm-HIP) for AMD).
- Fix macOS dispatch so `AirLLMKimiK3` (or a new `AirLLMKimiK3Mlx`) is reachable, not force-routed to
  `AirLLMLlamaMlx`.
- Preserve the **per-expert streaming hook** design (a hook per expert; load-compute-evict; the 880
  unrouted experts never read) on the new backend.

### WS1 — MLA attention without flash-attn (the 24 full-attn layers)
- Provide a non-flash attention path numerically equivalent to `flash_attn_varlen_func` for MLA
  (with NoPE + partial-RoPE split `qk_nope_head_dim=128` / `qk_rope_head_dim=64`, output gate).
- Apple: `mlx` scaled-dot-product attention or a custom Metal kernel; **or** run these layers eager on
  MPS. AMD: ROCm SDPA / CK-based attention, or eager.
- Must defeat the forced `flash_attention_2` override in `modeling_kimi_linear.py:1110-1119`
  (monkeypatch/fork the remote code to register an alternate attention interface).

### WS2 — KDA linear-attention kernels (the 69 layers) — HARDEST
- Replace `fla.ops.kda.{chunk_kda, fused_recurrent_kda}` (Triton) with equivalent kernels:
  - **Reference:** a pure-PyTorch/MLX recurrent implementation of the gated delta rule (correct, slow).
  - **Optimized:** custom **Metal** kernel (Apple) and **HIP/Triton-ROCm** kernel (AMD) for the chunked
    scan; include the short depthwise conv (`kernel_size=4`) and full-rank gating (`gate_lower_bound`,
    `A_log`, `dt_bias`).
- This is the main novel-kernel effort; budget the most time and the most numerical-parity testing here.

### WS3 — MXFP4 dequant on device (keep packed until on-GPU)
- Implement `mxfp4-pack-quantized` unpacking (4-bit, `group_size=32`, uint8 scale, symmetric float)
  in MLX/Metal and ROCm. Critically preserve AirLLM's memory trick: **weights cross the bus packed and
  expand one expert at a time on-device** — never materialize a full expanded layer (~55 GB).
- Validate the unpack numerically vs `compressed-tensors` CPU reference on a few expert tensors.

### WS4 — Router / MoE glue
- Reproduce `sigmoid` router + `noaux_tc` grouped top-k(16 of 896) + `e_score_correction_bias` +
  `routed_scaling_factor`, plus the **2 shared experts** (bf16) always active. Verify expert-index
  selection matches CUDA exactly (routing divergence => different experts => different output).

### WS5 — AMD-specific platform decision (RAISE EARLY)
- **AMD on Linux + ROCm** is the realistic target: ROCm PyTorch exists; Triton has partial ROCm support;
  HIP kernels are viable. **AMD on Windows 11 is effectively unsupported** (no ROCm PyTorch/Triton
  wheels) — flag whether Windows is in scope; if so, the only routes are DirectML/Vulkan-compute
  reimplementation (very large effort) or WSL2+ROCm (limited GPU support). **Recommend: scope AMD to
  Linux+ROCm; declare Windows-AMD out of scope unless separately funded.**

## Decision points for the team
1. **Port vs fork the remote modeling code?** The `modeling_kimi_*.py` files hardcode CUDA — decide
   between runtime monkeypatching (fragile across model updates) vs maintaining a forked, backend-neutral
   modeling module.
2. **Two very different backends** (Metal/MLX vs ROCm/HIP). Share a numpy/torch reference; keep kernels
   per-backend. Sequence Apple first (cleaner MLX story, our repro env is Apple) or AMD first?
3. **Numerical precision:** MXFP4 + bf16 accumulation — pin dtypes to match CUDA to keep logits in
   tolerance.

## Acceptance criteria (definition of done)
- **Numerical parity:** for a fixed prompt, first-N next-token logits match a CUDA K3 reference within
  an agreed tolerance (e.g. top-1 token identical for 20 tokens; logit MAE below threshold).
- **Sanity generation:** greedy decode of `"The capital of France is"` yields a coherent, correct
  continuation on each target backend.
- **Memory:** peak device memory stays within the streaming budget (never expands a full 55 GB layer);
  document the achieved ceiling.
- **Runs the full 1.56 TB checkpoint** (not a subset), from local NVMe.

## Suggested phasing
1. **Spike/feasibility (Apple):** get MLA + one KDA layer correct in a pure reference on MPS; unpack MXFP4
   correctly; prove parity on a few tokens through a handful of layers. Go/no-go gate.
2. **Full Apple reference path** (all 93 layers, streaming) — correctness, slow.
3. **Optimized Apple kernels** (Metal KDA + attention + MXFP4).
4. **AMD/ROCm** repeat of 2–3.

## Known-good reference environment (already built — use for parity baseline)
- Repo `~/kimi-k3-mlx`: `FINDINGS.md` (full checkpoint inspection),
  `p0/validate_mlx.py` (AirLLM MLX path on Llama-arch models), `p0/run_qwen3_mlx.py` (example of monkeypatching
  AirLLM's MLX backend to add a new architecture — QK-norm, head_dim, rope_theta — reusing on-disk
  reshard; a template for the K3 patch approach).
- Env: miniforge conda env `airllm`, native arm64 Python 3.12, `airllm 3.1.0` + `mlx` + `torch`; MLX
  confirmed on the Metal GPU. AirLLM's MLX layer-streaming validated end-to-end (TinyLlama-1.1B and a
  patched Qwen3-32B at ~18 s/token).
- Upstream: AirLLM (Apache License 2.0 — this brief originally said MIT, which was wrong)
  https://github.com/lyogavin/airllm ; `flash-linear-attention` (KDA);
  `compressed-tensors` (MXFP4). A CUDA box with a working K3 run is required to generate the parity
  reference.

## Risk callouts
- **WS2 (KDA kernels) is the schedule risk** — a novel gated-delta-rule kernel on two non-CUDA backends.
- **Windows-AMD is likely infeasible** without a major (DirectML/Vulkan) reimplementation.
- **Remote-code churn:** K3 modeling code updates could break monkeypatches; prefer a maintained fork.
- **Numerical drift** from MXFP4/bf16 differences can silently change routing → wrong experts → garbage;
  parity-test routing explicitly.
