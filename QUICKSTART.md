# Quick start

Start with the checkpoint-free synthetic smoke test:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-test.txt -e '.[dev]'
python tests/test_synthetic_integration.py
python tests/run_all.py
```

Requires Apple Silicon, GPU access, macOS 15+, and Python 3.11/3.12. Synthetic outputs
are random-model tokens, not a language-quality demo. Missing-checkpoint suites skip.

For the full 93-layer text model, first complete [INSTALL.md](INSTALL.md): obtain the
~1.42 TiB upstream snapshot, record its revision, build and verify the manifest, and
optionally stage dense weights. Then, from the repository root:

```sh
export K3_CKPT=/path/to/fast-ssd/Kimi-K3
export K3_MANIFEST="$PWD/manifest.jsonl.gz"
python -m air_llm.run --model "$K3_CKPT" --manifest "$K3_MANIFEST" \
  --mode runtime --resident-gb 8 --max-memory-gb 48 \
  --prompt 'The capital of France is' --max-new-tokens 4 --check-finite
```

The recorded hardware had 64 GiB unified memory. This run may take several minutes.
The best historical ~32.4 s/token excludes prefill; ~15 GiB is MLX allocator memory,
not whole-process/system usage. The checkpoint includes a vision tower that this
text-only implementation does not execute. [Measurement scope](docs/evidence.md).
