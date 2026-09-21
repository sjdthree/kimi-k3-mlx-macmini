# Kimi-K3 on Apple Silicon / MLX

[![tests](https://github.com/sjdthree/kimi-k3-mlx-macmini/actions/workflows/tests.yml/badge.svg)](https://github.com/sjdthree/kimi-k3-mlx-macmini/actions/workflows/tests.yml)

> *"RAM is expensive! But my SSD can hold a 1.5 TB model — can't I run it off that?"*
>
> **Yes!**

Run the complete **2.8-trillion-parameter, ~1.42 TiB Kimi-K3 checkpoint locally
on a 64 GB Mac mini**: all **93 language-model layers**, using SSD-backed
per-expert streaming and native MLX/Metal execution.

The recorded machine was an **M4 Pro Mac mini with 64 GiB unified memory**. The best
measured configuration decoded at **~32.4 seconds/token**, with **~15 GiB peak MLX
allocator memory**. That memory number excludes the OS file cache and is **not total
application footprint or total system memory**. This is useful for patient local
experiments; a long answer can take many hours.

**That's seconds per token, not tokens per second.** Yes, it's slow. What did you expect?

This project was **inspired by and extends [AirLLM](https://github.com/lyogavin/airllm)
and Gavin Li’s Kimi-K3 streaming work**. SSD, layer, and expert streaming are prior
work. The contributions here are the Apple Silicon implementation, K3 architecture
port, fused Metal MXFP4 kernel, latent MLA cache implementation, and measurement and
validation tooling. [Attribution and licenses](docs/licensing.md).

The full upstream MXFP4 checkpoint is used, without pruning experts or reducing
language-model depth. “Complete” describes the checkpoint and text model; the included
vision tower is **not executed**. The CUDA/FlashAttention/Triton paths in the reference
implementation do not run on Metal, so the text architecture is implemented with MLX
operations and a Metal kernel instead.

## Evidence and limits

| Result | What the repository supports |
|---|---|
| ~32.4 s/token | 39 decode records at 8 GiB dense residency; excludes prefill |
| ~15 GiB memory | MLX allocator high-water mark, not RSS or system memory |
| 53.3× smaller MLA cache | Logical fp32 payload: 2,949,120 → 55,296 bytes/context token across 24 MLA layers; excludes KDA state and capacity slack |
| ~2.35× historical improvement | Whole-run seconds/output-token, including prefill; different output lengths and resident budgets, not an isolated cache-only A/B |
| 0.99999976 worst-layer cosine | Full-depth fp32 run against the derived PyTorch/CPU oracle, rerun at the release commit on 2026-09-21; one generated token, 92/92 routers matched (historical run: 0.99999988) |
| 9.5× fused kernel microbenchmark | One MXFP4 projection benchmark; not an end-to-end speedup |

The [evidence ledger](docs/evidence.md) defines sample selection, configuration,
units, numerical scope, and corrections to older reports. Raw logs remain intact.
Full-checkpoint parity, the smoke test and the numerical-mode tests were rerun at the
release commit on 2026-09-21 and passed. The smoke test's decode time and bytes read per
token matched the historical 8 GiB runs. The timing, memory and speedup figures are still
**historical results** from the earlier sweep and runs. They are not performance
promises for the current checkout. No CUDA end-to-end comparison or million-token-context test is claimed.

## Try it without the checkpoint

On an Apple Silicon Mac with GPU access, macOS 15+, and Python 3.11 or 3.12:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-test.txt -e '.[dev]'
python tests/test_synthetic_integration.py
python tests/run_all.py
```

The synthetic test creates a tiny random checkpoint and compares eager, streaming,
fused/unfused, and saved-state paths. It tests execution, **not language quality**.
It typically takes seconds to minutes. Real-checkpoint tests print `SKIP` when the
checkpoint is absent; skipped tests are not evidence that the full model passed.

For real inference, follow [INSTALL.md](INSTALL.md) for checkpoint acquisition,
storage, manifest verification, dense staging, and the first full-depth smoke test.
[QUICKSTART.md](QUICKSTART.md) provides the shortest command sequence.

## Implementation

- Native MLX text architecture: MLA, KDA recurrence, attention residuals, latent MoE,
  shared experts, and top-16 routing over 896 experts in each MoE layer.
- Bounded per-expert weight loading; optional fill-and-pin dense residency. The best
  recorded budget on the 64 GiB machine was 8 GiB, not the largest budget tested.
- Fused Metal MXFP4 projection without materializing a full dequantized expert matrix.
- Latent MLA KV storage and chunked capacity growth. This implements the architecture’s
  latent-attention algebra; it does not invent MLA or guarantee bitwise identity.
- CLI and a local server implementing a documented subset of the Chat Completions
  API, including streaming. [API limitations](docs/API.md).

## Reproduce, contribute, release

[Tests and contribution guide](CONTRIBUTING.md) ·
[Measurement definitions](docs/evidence.md) ·
[Historical records](docs/history.md) ·
[Security reporting](SECURITY.md) ·
[Release checklist](docs/release.md)

First-party work is MIT where indicated. Moonshot-derived reference/configuration
material and the copied tokenizer pattern are separately covered by the **Kimi K3
License**; the vendored FLA reference kernel is MIT with its upstream notice. The
model weights are separately licensed and are not distributed here. Read
[NOTICE](NOTICE) and [the provenance inventory](docs/licensing.md); the distribution
must not be represented as uniformly MIT.

The Kimi K3 License is permissive, but its sections 2 and 3 add conditions for large
commercial deployments. A model-as-a-service business with more than US$20 million
revenue over 12 months needs a separate agreement with Moonshot AI. A product with more
than 100 million monthly active users or US$20 million monthly revenue must display
"Kimi K3" prominently. Read [the license](licenses/LICENSE.kimi-k3) before commercial use.
This summary is not legal advice.
