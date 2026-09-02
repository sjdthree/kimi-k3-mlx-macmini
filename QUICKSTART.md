# Quick Start

From a checkout to a generated token. Assumes the Kimi-K3 checkpoint is already on
local storage. If not, or if anything below fails, see [INSTALL.md](INSTALL.md).

**Time:** ~5 minutes of setup, then ~35 s per generated token.

---

## 1. Environment (2 min)

The model itself needs only **MLX and NumPy**. Torch is required solely for the
correctness oracle.

```bash
conda create -n airllm python=3.12 -y && conda activate airllm
pip install -e .        # mlx, numpy, tiktoken
```

Verify MLX sees the GPU:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
# -> Device(gpu, 0)
```

## 2. Index the checkpoint (~8 s)

```bash
python tools/k3_manifest.py build  --ckpt /path/to/Kimi-K3 --out manifest.jsonl.gz
python tools/k3_manifest.py verify --ckpt /path/to/Kimi-K3
```

Expect `M1 verify: PASS` and 15 green checks over 497,220 tensors. If any fail, the
checkpoint is incomplete or corrupt — stop here, don't run the model.

## 3. Generate

```bash
python -m air_llm.run \
  --model /path/to/Kimi-K3 \
  --mode runtime \
  --prompt "The capital of France is" \
  --max-new-tokens 4
```

```
93 layers (24 MLA, 69 KDA), 896 experts top-16
[ 0] id=17374   72.9s  read 196.46 GiB  peak 35.49 GiB  experts 17061
[ 1] id=20829   34.8s  read  95.57 GiB  peak 36.21 GiB  experts  4416
full: 'The capital of France is Paris.'
```

The first token is slower — that's prefill, where every prompt token routes
independently (17,061 expert projections vs 4,416 for a decode step).

---

## Make it ~1.6× faster (5 min, worth it)

~96% of a token is disk I/O, and 81% of that is **dense weights, identical on every
token**. Two changes attack exactly that.

### Stage dense weights on your fastest disk (105.7 GiB)

```bash
python tools/k3_cache.py --src /path/to/Kimi-K3 --dest ~/k3_dense \
    --layers 0-92 --expert-layers ""
```

Takes ~35 s at 3.5 GB/s. Then pass `--cache ~/k3_dense`. If your checkpoint is on an
external drive and you have an internal SSD with 106 GiB free, this alone is ~1.25×.

### Pin weights in RAM

```bash
python -m air_llm.run --model /path/to/Kimi-K3 --cache ~/k3_dense \
  --mode runtime --resident-gb 32 \
  --prompt "The capital of France is" --max-new-tokens 16
```

`--resident-gb` holds that many GiB of dense weights in unified memory permanently. On a
64 GiB machine, 32 is safe (measured peak 36 GiB against the 48 GiB ceiling). Raise it if
you have more RAM; watch the `peak` column.

| configuration | s/token |
|---|---:|
| checkpoint on external NVMe | 54 |
| `+ --cache` (dense on internal SSD) | 44 |
| `+ --resident-gb 32` | **34** |

---

## Useful flags

| flag | why |
|---|---|
| `--mode validation` | fp32 weights, unfused kernels — for parity work. Same greedy tokens, ~2× slower. |
| `--prompt-tokens 1,2,3` | skip the tokenizer entirely |
| `--layers N` | run only the first N layers (fast smoke test) |
| `--check-finite` | assert no NaN/inf in states, routing weights, logits (costs a sync per layer) |
| `--validation-log run.jsonl` | per-token JSON: timing, bytes read, peak memory, experts, checksum |
| `--max-memory-gb N` | abort if peak exceeds N (default 48) |

## Verify your install is correct

```bash
pip install -e '.[oracle]'                     # oracle-only dependencies (torch, einops)
python tests/test_synthetic_integration.py      # ~30 s, needs no checkpoint
python tests/test_mxfp4_mlx.py                  # MXFP4 decode vs oracle
```

The synthetic test builds a 2.7 MB miniature K3 and runs all six execution paths
(eager, layer streaming, expert streaming, standard kernels, fused kernels, saved-state
continuation), asserting they produce identical output. It's the fastest end-to-end
confidence check and needs no checkpoint at all.

## If it's slow

Check the disk, in this order:

1. **External enclosure on the wrong port.** A USB4/Thunderbolt enclosure in a
   10 Gbps USB-C port runs at ~1 GB/s instead of ~3.9 GB/s — **3.6× slower**, silently.
   Verify: `ioreg -rc IONVMeController | grep "Model Number"` should show your drive.
   If it instead appears under `ioreg -rc IOUSBHostDevice`, it's in USB fallback mode —
   move it to a rear Thunderbolt port and use the cable that shipped with it.
2. **Network storage will not work.** Measured 45 MB/s on the 5 MiB random reads this
   workload depends on → ~10 min/token. Copy the checkpoint locally
   (`tools/k3_copy.py`).
3. **Spotlight indexing the checkpoint:** `sudo mdutil -i off /Volumes/<drive>`.

Full troubleshooting in [INSTALL.md](INSTALL.md#troubleshooting).
