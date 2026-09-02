# Installation

Complete setup for running Kimi-K3 on Apple Silicon. For the fast path, see
[QUICKSTART.md](QUICKSTART.md).

---

## 1. Hardware requirements

| | minimum | what was used |
|---|---|---|
| Machine | Apple Silicon Mac | Mac mini M4 Pro (`Mac16,11`) |
| Unified memory | 32 GiB | 64 GiB |
| Storage for checkpoint | **1.42 TiB** free | 2 TB USB4 NVMe (APFS) |
| Storage for dense stage | 106 GiB free (optional, ~1.25×) | internal SSD |
| macOS | any recent | Darwin 25.5 |

**Memory.** 64 GiB comfortably runs the model with 32 GiB of weights pinned (measured
peak 36 GiB). At 32 GiB total, run with `--resident-gb 0` and `--max-memory-gb 24`;
the streaming path itself only needs ~6 GiB, so it will work, just without the pinning
speedup.

**Storage is the whole ballgame.** ~96% of a token is disk I/O. The metric that matters
is **random read at ~5 MiB granularity** — one expert projection is exactly 5,505,024
bytes and a decode token issues 4,416 of them. Sequential throughput and 4K IOPS are both
misleading here.

Measured on this hardware:

| | sequential read | random 5.25 MiB | s/token |
|---|---:|---:|---:|
| NAS over SMB (1 GbE) | 102 MB/s | 45 MB/s | ~600 |
| USB4 NVMe, **front port** (10 Gbps) | 1061 MB/s | 975 MB/s | ~100 |
| USB4 NVMe, **rear TB port** | 3875 MB/s | 3378 MB/s | 54 |
| Mac internal SSD | 5348 MB/s | 4670 MB/s | ~44 |

**Network storage is not viable.** The 45 MB/s figure is not a slow link — it's what SMB
does to 5 MiB random reads, and it does not improve with concurrency (measured flat from
4 to 32 threads). Copy the checkpoint to local storage.

Benchmark your own drive before committing to it:

```bash
python tools/bench_disk.py --path /Volumes/YourDrive/.bench --size-gib 80
```

The file must exceed RAM or the page cache invalidates the result; the tool flags the
contradiction (random faster than sequential) if that happens.

## 2. External drive setup

If using an external drive, format it **APFS, GUID Partition Map, unencrypted**. In Disk
Utility: View → Show All Devices, select the *physical drive*, Erase.

Not exFAT (no sparse files, poor large-file throughput on macOS — shards are 17 GB each).
Not "APFS (Case-sensitive)" — no benefit, and it trips some tooling.

Then, immediately:

```bash
sudo mdutil -i off /Volumes/<NAME>     # stop Spotlight indexing 1.4 TB of safetensors
```

Spotlight will otherwise churn for hours competing with exactly the reads you care about.
(`tmutil addexclusion` on a volume root fails with "Invalid argument" and is unnecessary —
Time Machine doesn't back up external volumes unless you explicitly add them.)

### The port trap

**A USB4/Thunderbolt enclosure in the wrong port is 3.6× slower, silently.** The Mac
mini's *front* USB-C ports are 10 Gbps only; the *rear* ports are Thunderbolt. In a front
port, a USB4 enclosure falls back to USB mass storage mode and you get ~1 GB/s instead of
~3.9 GB/s.

Diagnose:

```bash
# In USB4/PCIe mode -- correct:
ioreg -rc IONVMeController | grep "Model Number"        # -> "PCIe SSD"

# In USB fallback -- wrong:
ioreg -rc IOUSBHostDevice | grep -A1 "SSD"              # -> "USB3.2 SSD Drive Enclosure"
```

If it shows up as a USB device, move it to a **rear** port, plugged in **directly** (no
hub, no dock, no monitor passthrough), using the cable that shipped with the enclosure.
A 10 Gbps or charge-only cable produces exactly this symptom.

## 3. Python environment

Native **arm64** Python 3.11 or 3.12. Verify with `python -c "import platform;
print(platform.machine())"` → `arm64`. An x86_64 Python under Rosetta will not get MLX
GPU acceleration.

```bash
conda create -n airllm python=3.12 -y
conda activate airllm
```

### Dependencies, by what actually needs them

The model has **no torch dependency**. Torch is used only by the correctness oracle.

```bash
# Runtime -- everything needed to generate tokens
pip install -e .        # mlx, numpy, tiktoken

# Oracle + tests -- only if validating
pip install -e '.[oracle]'   # torch, einops
```

| package | needed by | why |
|---|---|---|
| `mlx` | model, tools | the backend |
| `numpy` | model, tools, oracle | tensor loading, host-side staging |
| `tiktoken` | tokenizer | K3's BPE; skip it and use `--prompt-tokens` |
| `torch` | `k3ref/` only | the fp32 parity oracle |
| `einops` | `third_party/kda_naive.py` | upstream `fla` KDA reference uses it |
| `compressed-tensors` | `tests/test_mxfp4_mlx.py` | independent MXFP4 decode oracle |

Versions verified working: mlx 0.32.0, numpy 2.5.1, tiktoken 0.13.0, torch 2.13.0,
einops 0.8.2, compressed-tensors 0.17.1, Python 3.12.13.

**`transformers` is not required.** The tokenizer shim builds the `tiktoken.Encoding`
directly from the checkpoint, so there's no `trust_remote_code` and no transformers
version coupling. (Contrary to some reports, K3's shipped `tokenization_kimi.py` *does*
load under transformers 5.x — the "remote code fails on 5.x" issue applies to the
*modeling* code, which this project doesn't use at all.)

Verify:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"   # -> Device(gpu, 0)
```

## 4. Get the checkpoint

1.42 TiB across 96 safetensors shards plus config and tokenizer files.

From Hugging Face (`moonshotai/Kimi-K3`), or if you already have it on network storage,
copy it locally with the included tool — resumable, verifies size per file:

```bash
python tools/k3_copy.py --src /Volumes/NAS/Kimi-K3 --dest /Volumes/Fast/Kimi-K3
python tools/k3_copy.py --dest /Volumes/Fast/Kimi-K3 --verify-only
```

Copying from a 1 GbE NAS takes ~4.5 hours (network-bound at ~100 MB/s). It skips files
already complete, so it survives interruption.

The `.pdf` files some mirrors include are excluded automatically. The vision tower
(`vision_tower.*`, `mm_projector.*`, 0.83 GiB) is present but unused for text inference.

## 5. Build the manifest

Required. Maps all 497,220 tensors to shard, byte offset, shape, dtype, and role.

```bash
python tools/k3_manifest.py build  --ckpt /path/to/Kimi-K3 --out manifest.jsonl.gz
python tools/k3_manifest.py verify --ckpt /path/to/Kimi-K3
```

`verify` runs 15 structural checks — every tensor resolves to exactly one shard, byte
ranges match shape×dtype, all 92 MoE layers have their full 896×3×2 expert tensors, every
`weight_packed` has its `weight_scale`, MLA/KDA layer sets partition the model. **If any
check fails, stop.** A silently incomplete checkpoint produces garbage, not an error.

The manifest embeds an absolute checkpoint path, so **rebuild it if you move the
checkpoint**. It's gitignored for that reason.

## 6. Optional: stage dense weights (~1.25×)

Dense (non-expert) weights are 105.7 GiB and are read *in full on every token*. Experts
are only 24.1 GiB per token. Putting dense on your fastest disk is the single best
storage move:

```bash
python tools/k3_cache.py --src /path/to/Kimi-K3 --dest ~/k3_dense \
    --layers 0-92 --expert-layers ""
```

~35 s at 3.5 GB/s. Then pass `--cache ~/k3_dense`. Experts keep streaming from the main
checkpoint, so this is a two-tier setup, not a duplicate.

## 7. Run

```bash
python -m air_llm.run \
  --model /path/to/Kimi-K3 \
  --cache ~/k3_dense \
  --mode runtime \
  --resident-gb 8 \
  --prompt "The capital of France is" \
  --max-new-tokens 16 \
  --validation-log results/run.jsonl
```

---

## Troubleshooting

**`Device(cpu, 0)` instead of `Device(gpu, 0)`**
Python is running under Rosetta. Check `platform.machine()` — must be `arm64`. Recreate
the conda env with an arm64 miniforge.

**Slower than ~60 s/token**
Work through §1–2: port, then Spotlight, then whether the checkpoint is on network
storage. Measure with `python tools/bench_disk.py`. A useful sanity number: the loader
achieves ~92% of raw device throughput on dense reads, so if the drive benchmarks at
3.4 GB/s you should see ~32–35 s/token with `--cache` and `--resident-gb 8`.

**`--resident-gb` doesn't help — or makes things worse**
Expected above ~24 GiB. Pinning competes with the OS page cache for the same unified
memory, and the page cache is what serves the 105.7 GiB working set; a 5-point sweep
measured 8 GiB at 32.4 s/token and 40 GiB at 69.0, monotonically worse at every step.
Start at 8. Below that it only pays from token 2 onward — token 1 populates it — so
confirm `peak` is actually rising; if the budget exceeds free memory the allocator
refuses and residency stays at 0. Full measurements: [docs/kv-cache/](docs/kv-cache/README.md).

**`MemoryError: peak unified memory exceeded the configured ceiling`**
Lower `--resident-gb`, or raise `--max-memory-gb` if you have headroom. Working set
beyond the pinned budget is ~7 GiB.

**`ModuleNotFoundError: einops`**
Only the oracle needs it: `pip install einops`. The model doesn't.

**Tokenizer fails to load**
Use `--prompt-tokens 1,2,3` to bypass it entirely. To debug:
`python tests/test_tokenizer.py`.

**Manifest verify fails after moving the checkpoint**
Rebuild it — the path is embedded. `python tools/k3_manifest.py build --ckpt <new path>`.

**Output is coherent but drifts into code/diff formatting**
Expected for a base model given a bare prompt. K3 ships an XTML chat format
(`encoding_k3.py`) that this project does not currently apply.

## Validating the install

```bash
python tests/test_synthetic_integration.py   # ~30 s, no checkpoint needed
python tests/test_mxfp4_mlx.py               # MXFP4 decode, bit-exact vs oracle
python tests/test_tokenizer.py               # ids identical to the shipped tokenizer
python tests/test_attention_mlx.py           # MLA + KDA vs oracle on real weights
python tests/test_layer_mlx.py --layers 13 --cache ~/k3_dense --ckpt /path/to/Kimi-K3
```

The synthetic test is the best first check — it builds a 2.7 MB miniature K3 and asserts
all six execution paths agree, without touching the real checkpoint.

Full reliability run (5 prompts × 16 tokens × 3 process launches, ~2.5 h):

```bash
python tools/reliability.py --model /path/to/Kimi-K3 --cache ~/k3_dense \
    --mode runtime --resident-gb 8 --max-new-tokens 16
```

Results from the reference run: [VALIDATION_REPORT.md](VALIDATION_REPORT.md).
