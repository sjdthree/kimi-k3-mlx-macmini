> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](docs/evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Implementation Plan — Kimi-K3 on Apple Silicon (MLX)

Plan for executing `apple_si_kimik3.md`. Written 2026-08-06 after re-verifying the spec against the
actual checkpoint at `/Volumes/LLMModels/Kimi-K3`, the shipped remote modeling code, the `fla`
upstream source, and this host.

Host: Apple M4 Pro (`Mac16,11`), **64 GiB** unified memory, internal NVMe **1.2 TiB free**,
checkpoint on **SMB NAS** (`//nas/LLMModels`, 4.6 TiB free).

---

## 0. Verdict

The spec is directionally right and the project is feasible, but **it was written without the
checkpoint in hand**. Six of its concrete claims are wrong against the real files, and two of its
acceptance gates are unreachable on this hardware as written. Section 1 below lists the corrections;
the rest of the plan is built on the verified facts, not the spec's.

The single biggest de-risking finding: **`fla/ops/kda/naive.py` exists upstream** (a 166-line pure
PyTorch `naive_recurrent_kda` / `naive_chunk_kda`). KDA — called out in `PORTING_BRIEF.md` as the
schedule risk — has a readable executable reference. It is not a from-scratch kernel derivation.

The second: **the dense (non-expert) weights are only ~114 GB.** They fit on local NVMe with room to
spare. Only the MXFP4 experts (~26 GB/token of selective reads) need to come off the NAS. This turns
the I/O story from "12 min/token over SMB" into "~1–4 min/token", and it is not in the spec at all.

---

## 1. Corrections to the spec (verified against the checkpoint)

| # | Spec says | Actually |
|---|---|---|
| 1 | Experts are `mlp.experts.N.gate_up_proj` / `down_proj` operating on hidden (§5, §11) | `block_sparse_moe.experts.N.{w1,w2,w3}.weight_packed`, and they operate in a **3584-dim latent space**, not hidden (7168) |
| 2 | MoE = router → experts → accumulate (§11) | There is a **latent MoE wrapper**: `routed_expert_down_proj [3584,7168]` → `routed_expert_norm [3584]` → experts → `routed_expert_up_proj [7168,3584]`. `latent_moe_use_norm=true`. Missing this = wrong output, silently |
| 3 | MLA layers at `[4,8,...,92,93]` (`PORTING_BRIEF.md`) | `config.linear_attn_config` is **1-indexed**. Real 0-indexed MLA layers: `[3,7,11,...,87,91,92]`. Note 91 and 92 are **adjacent**. KDA: the other 69, including **layer 0** |
| 4 | 1240 safetensors shards (§5 example) | **96 shards**, ~17 GB each, and provably **one layer per shard** (0 mixed-layer shards). Ideal for streaming; AirLLM's reshard step has nothing to do |
| 5 | `--max-memory-gb 112` (§19) | 64 GiB total unified on this box. Ceiling must be ~48 GB |
| 6 | 128 GB unified preferred, "internal or directly attached NVMe" for 1.56 TB | We have 64 GiB and 1.2 TiB free local — **cannot hold the full checkpoint locally**. Hence the dense/expert split (§4) |

Two acceptance gates are unreachable here:

- **§15 CUDA Reference Tests / §16 criterion 5–6 / M9.** There is no NVIDIA box. See §3 for the
  substitute oracle and when CUDA is actually needed.
- **§16 "loads the complete 1.56 TB checkpoint"** is satisfiable, but only from the NAS for the
  expert portion; "from local NVMe" (`PORTING_BRIEF.md`) is not.

Additional findings the spec does not mention, all of which must be implemented:

- **`situ` activation** (`modeling_kimi_linear.py:64-82`): `beta*tanh(gate/beta)*sigmoid(gate)*up`,
  with `up` pre-transformed by `linear_beta*tanh(up/linear_beta)`. `beta=4.0`, `linear_beta=25.0`,
  computed in fp32. Not SiLU. Not gated-SiLU.
- **Attention residuals are block-structured**: `attn_res_block_size=12`, per-layer
  `self_attention_res_proj [1,7168]` / `mlp_res_proj [1,7168]` (output dim **1** — these emit a
  per-token scalar), plus a model-level `output_attn_res_{proj,norm}`. See `_apply_attn_res`
  (`modeling_kimi_linear.py:1075`) and `_forward_attn_residual` (`:973`).
- **`A_log` shape mismatch.** Checkpoint ships `A_log F32 [128]`; the shipped modeling code declares
  it as `[num_heads]` = 96. `head_dim`=128, so K3's decay is **per-channel within a head**, not
  per-head. The shipped `modeling_kimi_linear.py` appears to be the Kimi-Linear-48B variant. Resolve
  against `fla/ops/kda/gate.py` before writing anything. **This is the first thing to nail down.**
- **KDA semantics live in `fla`, not in the modeling file.** The call passes
  `use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True, use_beta_sigmoid_in_kernel=True,
  safe_gate=True, lower_bound=-5.0, transpose_state_layout=True` — every one of those is a piece of
  math hidden inside the Triton kernel. Read `fla/ops/kda/{naive,gate,fused_recurrent}.py`.
- **PyPI's `flash-linear-attention` wheel does not ship `fla/ops`.** Must clone from GitHub
  (`fla-org/flash-linear-attention`) to get the KDA source. Already cloned to scratchpad.
- **Vision tower is skippable.** 168 of 173 non-layer tensors are MoonViT/`mm_projector`. Text-only
  inference never touches them.
- **`transformers` 5.12.1 is installed; remote code needs 4.56.x.** Irrelevant for an MLX-native
  reimplementation *except* for the tokenizer (`tokenization_kimi.py`, TikToken). Budget a shim.

---

## 2. Verified architecture (the thing to reproduce)

93 layers. Layer 0 dense MLP (`intermediate_size=33792`); layers 1–92 latent MoE.
`hidden_size=7168`, `vocab=163840`, bf16, `rms_norm_eps=1e-5`.

**KDA layer (69 of them, incl. layer 0)** — per-layer tensors and shapes:
```
q_proj/k_proj/v_proj/g_proj  BF16 [12288, 7168]     (96 heads x 128 head_dim)
q_conv1d/k_conv1d/v_conv1d   F32  [12288, 1, 4]     depthwise, silu, kernel 4
f_a_proj [128,7168] -> f_b_proj [12288,128]         low-rank gate g
b_proj   BF16 [96, 7168]                            beta, one per head
A_log    F32 [128]   dt_bias F32 [12288]  o_norm F32 [128]
o_proj   BF16 [7168, 12288]
```
`o_norm` is `FusedRMSNormGated(activation='sigmoid')` — gated RMSNorm, not plain RMSNorm.

**MLA layer (24)**:
```
q_a_proj [1536,7168] -> q_a_layernorm [1536] -> q_b_proj [18432,1536]
kv_a_proj_with_mqa [576,7168] -> kv_a_layernorm [512] -> kv_b_proj [24576,512]
g_proj [12288,7168]   (output gate; mla_use_output_gate=true)
o_proj [7168,12288]
qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128, mla_use_nope=true
```

**Latent MoE block (layers 1–92)**:
```
gate.weight BF16 [896,7168]  +  gate.e_score_correction_bias F32 [896]
  router: sigmoid, topk_method=noaux_tc, top-16 of 896, num_expert_group=1,
          topk_group=1, moe_renormalize=true, routed_scaling_factor=1.0
routed_expert_down_proj BF16 [3584,7168]   hidden -> latent
routed_expert_norm      BF16 [3584]
  expert i:  w1 U8 [3072,1792] + scale U8 [3072,112]   latent(3584) -> 3072   (gate)
             w3 U8 [3072,1792] + scale U8 [3072,112]   latent(3584) -> 3072   (up)
             w2 U8 [3584,1536] + scale U8 [3584,96]    3072 -> latent(3584)   (down)
routed_expert_up_proj   BF16 [7168,3584]   latent -> hidden
shared_experts.{gate,up}_proj BF16 [6144,7168], down_proj BF16 [7168,6144]  (2 shared, bf16, unquantized)
```
Per-expert packed footprint: **17,547,264 B ≈ 16.7 MiB**. 16 routed experts = **281 MB per layer per token**.

---

## 3. The oracle problem (no CUDA box)

The spec's entire correctness story rests on CUDA reference artifacts. We don't have CUDA. Plan:

**Primary oracle — torch/CPU fp32, same semantics.** Build `k3ref/`, a PyTorch reference that runs
one layer at a time on CPU in fp32, using:
- `fla.ops.kda.naive.naive_recurrent_kda` / `naive_chunk_kda` for KDA (upstream code, unmodified),
- eager attention (`eager_attention_forward`, already in `modeling_kimi_linear.py:311`) for MLA,
- `compressed-tensors` CPU dequant for MXFP4,
- the shipped `SituAndMul`, `KimiMoEGate`, `KimiSparseMoeBlock`, `_apply_attn_res` verbatim.

This is not a reimplementation — it is the shipped model with the two CUDA-only kernels swapped for
their pure-torch equivalents. It is the test oracle for **every** MLX component and for whole layers
on real weights. It runs on this Mac.

**Secondary — CUDA, once, at the end.** The torch/CPU oracle validates MLX against *the shipped
code's semantics*. It cannot catch a misreading shared by both. One rented NVIDIA box (48 GB card +
1.6 TB NVMe, the AirLLM article's exact config) generating the §15 reference artifacts for a fixed
prompt closes that. Do it **once, at M9**, not up front. Cost: a few hundred dollars + a 1.56 TB
HF download on the box. Decide at M8; the project is not blocked on it before then.

Consequence: M9 as written ("CUDA parity for 16 greedy tokens") becomes **"torch-CPU parity for 16
greedy tokens, CUDA confirmation optional"**.

---

## 4. Storage and I/O plan (not in the spec)

Measured/computed per-token read volume:

| Component | Bytes/token | Where |
|---|---:|---|
| Dense weights, 69 KDA layers (~1.27 GB each) | ~88 GB | **local NVMe** |
| Dense weights, 24 MLA layers (~0.84 GB each) | ~20 GB | **local NVMe** |
| Layer 0 dense MLP | ~1.5 GB | local NVMe |
| `lm_head` | 2.35 GB | local NVMe |
| `embed_tokens` (one row per token) | ~14 KB | local NVMe |
| **Routed experts** (16 × 92 × 16.7 MiB) | **~26 GB** | **NAS over SMB** |

**Dense total ≈ 114 GB** — copy once to local NVMe (1.2 TiB free). Experts stay on the NAS and are
read selectively; 98% of them are never touched for a given token.

Projected: dense at ~5 GB/s local ≈ 23 s, experts at SMB throughput T. At 200 MB/s → ~130 s; at
110 MB/s (gigabit) → ~240 s. **~1–4 min/token**, vs ~12 min if everything stays on SMB. Measuring T
is task P0-1.

Resident memory is comfortable: one layer's dense weights (~1.3 GB) + one expanded expert (~30 MB
fp32) + state. KDA recurrent state is `96×128×128` fp32 = 6.3 MB/layer × 69 = **434 MB**; MLA KV
cache is `576 × 2 B × 24` = **27.6 KB per token** (latent cache — tiny). A 48 GB ceiling is generous.

---

## 5. Structure

Keep the spec's §3 layout so the deliverable matches, but **do not build on AirLLM's MLX Llama
path** — it models Llama-2/3 attention only (see `FINDINGS.md` B2b) and shares nothing with K3.
AirLLM contributes its reshard/streaming scaffolding and the `AutoModel` entry point; given the
shards are already one-layer-per-file, even the reshard is a no-op (hardlink). Recommendation:
implement `air_llm/models/kimi_k3/` as a self-contained MLX model with an AirLLM-shaped dispatch
entry, in a fork. Do not monkeypatch the remote modeling code — fork it for the reference
(`PORTING_BRIEF.md` decision point 1, resolved: **fork**).

```
air_llm/models/kimi_k3/     config, model, attention_mla, attention_kda, router,
                            moe_latent, mxfp4, recurrent_state, weight_manifest
air_llm/streaming/          tensor_loader, expert_loader, cache, lifecycle
air_llm/kernels/metal/      mxfp4_matvec, kda_step, normalization
k3ref/                      torch-CPU fp32 oracle (see §3)
tools/                      build_manifest, verify_checkpoint, stage_dense_local,
                            compare_reference, inspect_memory
tests/                      unit, integration, parity, full_model
```

---

## 6. Milestones

Rescoped from the spec's M1–M11. Estimates are engineer-weeks for one experienced engineer.

| # | Deliverable | Gate | Est. |
|---|---|---|---|
| **P0** | **Feasibility spike** (§7) | see §7 | **1** |
| M1 | Manifest builder + checkpoint verifier + `stage_dense_local` tool; dispatch to `AirLLMKimiK3Mlx` | Every one of 497,220 tensors resolves; dense subset staged locally and byte-verified | 1 |
| M2 | `k3ref/` torch-CPU oracle running whole real layers, layer-by-layer streaming | Layer 1 (KDA+MoE) and layer 3 (MLA+MoE) run on real weights, fp32, output checksums recorded | 1.5 |
| M3 | MXFP4: NumPy reference decoder + MLX decoder | Exhaustive 16 FP4 codes + 256 scale bytes; matches `compressed-tensors` on 20 real expert tensors, exact before float cast | 1 |
| M4 | MLX MLA (24 layers) | Matches `k3ref` MLA on real layer-3 weights, seqlen 1/2/short-prefill/cached-decode, atol 1e-5 fp32 | 1.5 |
| M5 | MLX KDA (69 layers), reference path | Matches `naive_recurrent_kda` **state-per-token**, not just output. Includes short conv, full-rank gate, gated o_norm, `lower_bound` clamp | 2.5 |
| M6 | Router + latent MoE + shared experts + serial expert streaming | Exact top-16 index match vs `k3ref` on real layer-1 weights, weights atol 1e-5. Latent down/norm/up path verified | 1.5 |
| M7 | Full transformer layer incl. block attention residuals (`attn_res_block_size=12`) | 12 consecutive real layers match `k3ref` end-to-end; residual block boundary exercised | 1.5 |
| M8 | Full 93-layer one-token forward + `KimiK3State` save/restore | One token, real checkpoint, no NaN, memory under ceiling, logits checksum stable across two runs | 1.5 |
| M9 | 16 greedy tokens vs `k3ref`; optional CUDA confirmation | Top-1 identical for 16 tokens; router selection identical at all 92 MoE layers | 1.5 |
| M10 | Fused Metal MXFP4 matvec + optional fused KDA step | Matches unfused path; measurable wall-clock win. **Cut this if schedule slips** | 2 |
| M11 | Reliability: 5 prompts × 16 tokens × 3 launches; memory/leak report | 15/15 deterministic, <2% drift, no NaN | 1 |

**~17 engineer-weeks** to M11; **~15** if M10 is cut. M5 and M2 are where the risk lives.

Sequencing note: M3 and M4 are independent of M5 and can run in parallel with a second engineer.
M5 is on the critical path from P0 onward.

---

## 7. P0 — feasibility spike (do this first, ~1 week)

Five tasks, each with an explicit go/no-go. Nothing downstream is worth starting until these pass.

1. **Measure SMB throughput.** Sequential and 5.5 MB-random-read from the NAS. Feeds the token-time
   projection and decides whether an expert read-ahead cache is needed at all.
   *Go:* ≥100 MB/s sustained random-read.
2. **Resolve `A_log [128]` vs `num_heads=96`.** Read `fla/ops/kda/gate.py` and `naive.py`; determine
   whether decay is per-head or per-channel, and confirm `dt_bias [12288]` layout is
   `(head, channel)` row-major. *Go:* an unambiguous answer backed by the kernel source.
   **This is the highest-information task in the plan — do it first.**
3. **Decode one real MXFP4 group** from `layers.1.block_sparse_moe.experts.0.w1` with a hand-written
   NumPy decoder, and match `compressed-tensors` bit-exactly. Settles nibble order, scale
   interpretation (E8M0 exponent byte), and matrix orientation. *Go:* exact match.
4. **Run one real KDA layer** (layer 1) through `k3ref` on CPU in fp32 — projections, short conv,
   gates, `naive_recurrent_kda`, gated `o_norm`, `o_proj` — and record output + state checksums.
   *Go:* runs, finite, and stable across two invocations.
5. **Stage the dense subset locally.** Extract the ~114 GB of non-expert tensors from the 96 NAS
   shards into per-layer local files. *Go:* completes, byte-verified, ≤130 GB on disk.

**Overall P0 gate:** all five green → proceed to M1. Task 2 or 3 red → the KDA/MXFP4 semantics are
not recoverable from available source, and a CUDA reference run must be pulled forward to M0 before
anything else is built.

---

## 8. Risks

| Risk | Impact | Mitigation |
|---|---|---|
| KDA semantics not fully recoverable from `fla` source (the `*_in_kernel=True` flags hide math) | Blocks M5, the critical path | P0-2 and P0-4 test exactly this in week 1. Fallback: pull the CUDA reference run forward |
| No CUDA oracle → both MLX and `k3ref` share a misreading | Silently wrong output that passes every test | The CUDA run at M9 exists for precisely this. Do not skip it before declaring release-candidate |
| Numerical drift changes router top-16 → different experts → garbage | Catastrophic and non-obvious | Assert exact router index match at every MoE layer, every token (M6, M9). Never a tolerance |
| SMB latency/throughput worse than projected | Token time → 10+ min; iteration on M8–M11 becomes painful | P0-1 measures it. If poor: stage a *subset* of hot experts locally, or attach a USB4 NVMe |
| Latent-MoE / attention-residual structure misread | Wrong output, passes component tests | M7 gates on 12 consecutive real layers, which is the residual block period |
| 64 GiB unified, not the spec's 128 | Prefill length limits | Ceiling 48 GB; short prompts for parity work; state is small (§4) so this is manageable |
| Tokenizer needs `transformers` 4.56 API | Cosmetic; blocks end-to-end runs only | Pin a second env, or shim TikToken directly (`tiktoken.model` is present) |
| Remote-code churn upstream | Reference drifts | Fork and pin; record the checkpoint revision in the reference manifest (spec §15) |

---

## 9. Spec amendments to sign off before starting

1. §19 `--max-memory-gb 112` → **48** on this host.
2. §16 criteria 5–6 and M9: **CUDA parity → torch-CPU parity**, CUDA confirmation deferred and
   explicitly budgeted (§3).
3. §5 manifest schema: expert roles are `w1`/`w2`/`w3` in latent space, not `gate_up`/`down` in
   hidden space (§1 row 1–2).
4. §11 MoE: add the `routed_expert_down_proj` → `routed_expert_norm` → experts →
   `routed_expert_up_proj` latent wrapper.
5. §7/§12: add the `attn_res_block_size=12` block-residual structure and the model-level
   `output_attn_res_*` path.
6. Add: dense-weight local staging (§4). Add: `situ` activation. Add: MLA/KDA layer index lists are
   **1-indexed in config** and must be converted.
7. Storage prerequisite in §2 ("internal NVMe for 1.56 TB") is **not met**; §4 is the replacement.

---

## 10. Immediate next action

Run P0-2 (`A_log` semantics from `fla/ops/kda/gate.py`) and P0-3 (real MXFP4 group decode). Both are
a few hours, need no infrastructure, and between them they determine whether the two hardest pieces
of this project are well-posed. `fla` is already cloned to the session scratchpad.
