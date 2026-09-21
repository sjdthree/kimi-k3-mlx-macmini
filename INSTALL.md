# Installation and full-checkpoint smoke test

## Prerequisites

Use an Apple Silicon Mac with a working Metal GPU and Python 3.11 or 3.12.
The documented dependency pair requires macOS 15 or newer. The historical full-model
measurements used an M4 Pro Mac mini, 64 GiB unified memory, and local USB4 NVMe
storage on a rear Thunderbolt port. A 32 GiB full-model configuration is unvalidated.
CI/synthetic success on another Mac does not validate its full-model performance.

Connect a USB4/Thunderbolt NVMe enclosure to a **rear** Thunderbolt port. On the M4 Pro
Mac mini the front USB-C ports are 10 Gbps USB. There the enclosure falls back to USB
mass storage, and it measured about 1.06 GB/s sequential versus 3.88 GB/s on a rear
port. Confirm with `ioreg -rc IONVMeController`: the drive should be listed with
`"Physical Interconnect Location" = "External"`, not only under `IOUSBHostDevice`.

Allow **~1.42 TiB for the checkpoint**, plus download/cache headroom. Dense staging is
optional and needs approximately **106 GiB extra**. A disk microbenchmark can need
another **80 GiB or more**; do not run it during inference measurements. A nominal 2 TB
drive provides about 1.82 TiB before filesystem overhead. Check free space on every
volume; do not accidentally keep a second full download in a package cache.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-test.txt -e '.[dev]'
python tests/test_synthetic_integration.py
```

`pip install -e .` installs the runtime only. `[server]` adds the HTTP server;
`[oracle]` adds PyTorch/einops for the CPU reference; `[dev]` includes the test clients.
The runtime configuration loader does not require PyTorch. Test constraints identify
a bounded environment; they do not establish checkpoint parity on this checkout.

## Acquire the upstream checkpoint

Review [Moonshot’s model page and license](https://huggingface.co/moonshotai/Kimi-K3)
first. Choose and record a **full immutable repository commit hash** from its history;
there is no verified original benchmark snapshot hash in this repository. Do not use
an unrecorded moving `main` revision for a reproducibility claim.

Install the download tool separately:

```sh
python -m pip install huggingface_hub
export K3_CKPT=/path/to/fast-ssd/Kimi-K3
export K3_REVISION=REPLACE_WITH_FULL_UPSTREAM_COMMIT_HASH
hf download moonshotai/Kimi-K3 --revision "$K3_REVISION" --local-dir "$K3_CKPT"
```

Download the whole snapshot, including `config.json`, `model.safetensors.index.json`,
all indexed shards, `tiktoken.model`, and tokenizer configuration. The implementation
does not execute downloaded Python code for ordinary inference. If authentication is
required by the host, use its login flow and keep tokens outside the repository.
Download time depends on the network and can be many hours or days.

## Build and verify the manifest

Run these from the repository root:

```sh
export K3_MANIFEST="$PWD/manifest.jsonl.gz"
python tools/k3_manifest.py build --ckpt "$K3_CKPT" --out "$K3_MANIFEST"
python tools/k3_manifest.py verify --ckpt "$K3_CKPT" --manifest "$K3_MANIFEST"
```

This reads headers, checks index coverage, shapes, layer structure, file bounds, and
config/index hashes. It does **not** cryptographically verify all model payloads.
The manifest records `K3_REVISION` as supplied by you, not as an attestation.
Keep snapshot identity and download verification records. The manifest embeds a local
path and is gitignored. Rebuild after moving or changing the checkpoint.

To copy an existing snapshot between drives with full content verification:

```sh
python tools/k3_copy.py --src /path/to/source/Kimi-K3 --dest "$K3_CKPT" --sha256
python tools/k3_copy.py --src /path/to/source/Kimi-K3 --dest "$K3_CKPT" --verify-only --sha256
```

Copy resumes at file boundaries. Without `--sha256`, verification is size-only.
Hash comparison reads both copies completely and can take hours; it proves equality
between copies, not authenticity of the source.

## Optional dense staging

```sh
export K3_CACHE=/path/to/internal-ssd/k3_dense
python tools/k3_cache.py --src "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --dest "$K3_CACHE" --layers 0-92 --expert-layers '' --plan-only
python tools/k3_cache.py --src "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --dest "$K3_CACHE" --layers 0-92 --expert-layers ''
```

Use a fresh cache destination. Staging preserves stored bytes and is optional; it is
separate from `--resident-gb`, which pins some dense tensors in MLX memory. Omit the
`--cache` argument below if you did not stage. Checkpoint-derived caches must stay
with the matching checkpoint; the current cache format does not attest that identity.

## First full-depth run

```sh
python -m air_llm.run --model "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --cache "$K3_CACHE" --mode runtime --resident-gb 8 --max-memory-gb 48 \
  --prompt 'The capital of France is' --max-new-tokens 4 \
  --check-finite --validation-log smoke.jsonl
```

Do not pass `--layers` for a full-model claim. Expect 93 layers in the startup output,
per-token diagnostics, and generated text. A short run may take several minutes;
prefill, cold reads, numerical checks, and machine load affect timing. Historical
~32.4 s/token is warm decode and excludes prefill. Do not compare a four-token smoke
test directly with that mean. `--max-memory-gb` is an after-layer MLX allocator check,
not a hard OS allocation cap; leave system headroom and monitor memory pressure/swap.

For the numerical release gate (expensive, potentially hours):

```sh
python tests/test_full_parity.py --ckpt "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --cache "$K3_CACHE" --tokens 1
python tests/test_numerical_modes.py
```

The mode test covers short two-layer runs, not full-depth runtime-mode equivalence.
Collect full-depth runtime-versus-validation comparisons over multiple prompts before
making broader equivalence claims. See [release gates](docs/release.md).

## Server

```sh
export AIR_LLM_API_KEYS="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export OPENAI_API_KEY="$AIR_LLM_API_KEYS"
python -m air_llm.serve --model "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --cache "$K3_CACHE" --mode runtime --resident-gb 8 --port 8080 \
  --max-output-tokens 256 --token-log server-tokens.jsonl
```

Keep the key out of shell transcripts and tracked files. The default bind address is
loopback. In another terminal, use the same key and the example in [docs/API.md](docs/API.md).
No public hosting, TLS, or production-hardening claim is made. A 1,959-token historical
prefill took about 25.7 minutes; use a sufficiently long client read timeout.
