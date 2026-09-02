# Kimi-K3 + AirLLM — Local Inspection Findings (Plan B1)

Date: 2026-08-06 · Host: Apple M4 Pro (64 GB unified, 20-core GPU/Metal) · Checkpoint: `/Volumes/LLMModels/Kimi-K3` (SMB NAS)

## TL;DR
- The checkpoint is exactly as the AirLLM article describes: **2.8T-param MoE, 1.56 TB, MXFP4-packed experts, 896 experts × 93 layers.**
- **K3 cannot run on this Mac (or on any AMD/Windows box).** The model's own code hard-locks to **CUDA**, in two independent ways — flash-attn *and* Triton KDA kernels. Neither has a Metal or Windows-AMD backend.
- AirLLM's Mac (MLX) path is real but does **not** cover K3; it's the general layer-streaming path for supported models (e.g. Qwen3-32B).

## Checkpoint facts (verified from the files)
| Item | Value |
|---|---|
| Total tensors | 497,220 |
| Total size | 1,560,860,324,864 B ≈ **1.56 TB** (matches doc) |
| Layers | 93 (index 0–92) |
| Experts / layer | 896 (index 0–895), top-16 routed, 2 shared |
| MoE layers | layers 1–92 (layer 0 is dense — `first_k_dense_replace: 1`) |
| Routed-expert weights | 82,432 each of w1/w2/w3 (= 896 × 92) |
| Expert storage | **MXFP4**: `weight_packed` (U8) + `weight_scale` (U8), group_size 32 |
| Sample expert w1 | `weight_packed U8 [3072,1792]`, `weight_scale U8 [3072,112]` (1792=3584/2, 112=3584/32) |
| Attention (mixed) | **69 KDA linear-attn layers** + **24 full MLA layers** = 93 |
| KDA tensors | f_a_proj, f_b_proj, A_log, {q,k,v}_conv1d, dt_bias, b_proj, o_norm |
| MLA tensors | q_a_proj, q_b_proj, kv_a_proj_with_mqa, kv_b_proj, {q,kv}_a_layernorm |
| Vision | MoonViT, 27 encoder blocks + mm_projector (VLM) |
| hidden_size | 7168 · vocab 163,840 · max_position 1,048,576 |
| Tokenizer | TikTokenTokenizer (`tiktoken.model`) — CPU-only, works on Mac |

## Why K3 is CUDA-locked (in the model code, not just AirLLM)
1. **flash-attn is forced.** `modeling_kimi_linear.py:1110-1119` overrides any requested attention impl:
   > "Ignoring the provided attention implementation … Using flash_attention_2 backend instead."
   and imports `from flash_attn import flash_attn_varlen_func`. flash-attn = NVIDIA CUDA only (no Metal build).
2. **KDA linear attention needs Triton.** `modeling_kimi_linear.py:48` imports `from fla.ops.kda import chunk_kda, fused_recurrent_kda` — the `flash-linear-attention` library's **Triton** kernels used by 69 of 93 layers. Triton has no Metal backend and no Windows-AMD path.

Either one alone blocks Mac/MPS and AMD/Windows. Together they make a native port a deep kernel-porting project, not a config change.

## Platform verdict
| Platform | Runs K3 via AirLLM? | Reason |
|---|---|---|
| Apple M4 Pro (this Mac) | ❌ | No CUDA; no flash-attn/Triton on Metal |
| AMD 128 GB / 96 GB-GPU, **Win11** | ❌ | No CUDA; ROCm PyTorch + ROCm flash-attn are **Linux-only**; model code hardcodes CUDA |
| NVIDIA CUDA GPU (even 8–16 GB) | ✅ | Expert streaming needs tiny VRAM; flash-attn + Triton available |

Note: for K3, **VRAM size is irrelevant** — AirLLM streams experts (headline: 3.72 GB). A cheap NVIDIA card runs it; a 96 GB AMD card does not.

## Requirements for a real run (when on CUDA)
- `pip install airllm compressed-tensors flash-attn`
- CUDA **12** build of torch (no flash-attn wheel for CUDA 13 yet)
- `transformers` **4.56.x** (K3 remote code fails on 5.x)
- Python **3.11/3.12** (torch/transformers have no 3.14 wheels)
- ~1.6 TB **fast local** NVMe (SMB NAS will be far slower than the doc's ~5 min/token)

## B2 result — AirLLM MLX path VALIDATED on this Mac (2026-08-06)
Env: miniforge conda env `airllm`, native **arm64 Python 3.12.13**. Installed: airllm 3.1.0,
mlx 0.32.0, mlx-lm 0.31.3, torch 2.13.0, transformers 5.12.1.

- MLX runs on the Metal GPU: `mx.default_device() -> Device(gpu, 0)`.
- Ran `p0/validate_mlx.py` with `TinyLlama/TinyLlama-1.1B-Chat-v1.0`:
  - AirLLM re-sharded to per-layer `.mlx` files under the HF cache `splitted_model/`.
  - Streamed all 22 layers per token on the GPU (~16 layers/s), then produced:
    **"The capital of France is Paris."** ✅
  - Setup 12.9 s (download + reshard), generation ~1.5 s/token for 1.1B.
- Confirmed dispatch: on macOS `AutoModel.from_pretrained` **always** returns `AirLLMLlamaMlx`
  (auto_model.py) — it never routes to `AirLLMKimiK3`. The K3 class exists only for the CUDA path,
  and its forward pass needs flash-attn + Triton KDA, absent on Metal. Fourth confirmation K3≠Mac.

## B2b — Qwen3-32B (~65 GB) on the Mac via a patch (2026-08-06)
Goal: run a big model end-to-end on the Mac. Qwen3-32B was the target (article's example).

**Problem:** airllm 3.1.0's MLX backend (`AirLLMLlamaMlx`) only models **Llama-2/3 attention**.
Qwen3 broke it three ways: (1) per-head QK-norm `q_norm`/`k_norm` not modeled
(`ValueError: Module does not have parameter named "q_norm"`); (2) head_dim inferred as
hidden/n_heads = 5120/64 = 80, but Qwen3's real head_dim is **128** (q_proj is 8192×5120);
(3) rope_theta defaulted to 1e4, Qwen3 uses **1e6**. The README's "Qwen3-32B on Mac" example
does NOT work with airllm 3.1.0 as shipped.

**Fix:** `p0/run_qwen3_mlx.py` — a self-contained monkeypatch (no site-packages edits) providing a
Qwen3-correct `Attention` (adds q_norm/k_norm RMSNorm over head_dim, applied per head before RoPE),
`TransformerBlock`, and model-args (head_dim=128, rope_theta=1e6). Reuses the already re-sharded
weights on disk — no re-download.

**Result (RTX-less, pure Metal/MLX streaming):**
- model_args: head_dim=128 n_heads=64 n_kv=8 rope_theta=1e6 n_layers=64 (setup 0.3 s, reused reshard)
- Output: **"The capital of France is Paris. What is the capital of Germany? The capital of"** ✅ coherent
- Speed: 12 tokens in 215.4 s = **17.9 s/token** (streams all 64 layers ≈ 65 GB/token off local NVMe)
- Disk: 122 GB (orig 65 GB + per-layer `.mlx` reshard)

Takeaway: AirLLM's Mac path works for big models, but only Llama-family arch out of the box; other
architectures (Qwen3 QK-norm, Qwen2 QKV-bias, Gemma) need a small per-arch patch like the above.

## Status
- **B1 (local inspection): complete.**
- **B2 (validate AirLLM Mac/MLX path): complete — works** (TinyLlama-1.1B).
- **B2b (big model, Qwen3-32B ~65 GB): complete — works via `p0/run_qwen3_mlx.py` patch.**
- Repro (Llama-arch): `conda activate airllm && python p0/validate_mlx.py [hf_repo_id]`
- Repro (Qwen3-32B): `conda activate airllm && python p0/run_qwen3_mlx.py ["prompt"] [max_new_tokens]`
- Next (optional) **A**: run K3 for real on an NVIDIA CUDA box (see requirements above).
