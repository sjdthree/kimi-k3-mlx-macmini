> **Historical engineering record — not current release guidance.**
> This document preserves earlier hypotheses, commands, and measurements. Some were
> superseded or overstated. See [the corrected evidence ledger](../../evidence.md) and
> the root README for current installation, claim scope, and release limitations.

# Full suite on `mla-latent-cache` — 2026-08-10 06:10 → 07:07

All 19 test files, run with the checkpoint volume free.

| suite | result |
|---|---|
| test_mla_cache | PASS |
| test_mla_quant | PASS |
| test_token_log | PASS |
| test_kda_prefill_memory | PASS *(new)* |
| test_openai_conformance | PASS |
| test_numerical_modes | PASS |
| test_resident_cache | PASS |
| test_streaming_memory | PASS |
| test_synthetic_integration | PASS |
| test_tokenizer | PASS |
| test_mxfp4_mlx | PASS |
| test_mxfp4_kernel | PASS |
| test_state_roundtrip | PASS |
| test_moe_mlx | PASS |
| test_attention_mlx | PASS |
| test_k3ref_layers | PASS |
| test_layer_mlx | PASS |
| test_model_mlx | PASS |
| test_full_parity | PASS *(52 min)* |

**19 passed, 0 failed.**

Run twice on this branch — once before the KDA prefill fix and once after. Both green,
and `test_full_parity` returned identical numbers each time (token 17374, worst per-layer
cosine 0.99999988 at layer 22, 0 router divergence across all 92 MoE layers), confirming
the periodic-eval change is a scheduling change only.

Note: the runner script exits 1 even on full success — its last statement is
`[ -n "$FAILED" ] && echo ...`, which returns 1 when there are no failures. Read the
`FULL SUITE:` line, not the exit code.
