"""Plan B2: validate AirLLM's Mac/MLX layer-streaming path on a small Llama model.

This does NOT run Kimi-K3 (K3 is CUDA-only: flash-attn + Triton KDA). It proves the
AirLLM MLX pipeline (re-shard -> per-layer stream -> generate) works on this M4 Pro.
"""
import sys, time
import numpy as np
import mlx.core as mx
from airllm import AutoModel

REPO = sys.argv[1] if len(sys.argv) > 1 else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT = "The capital of France is"

print(f"MLX device: {mx.default_device()}")
print(f"Loading (and re-sharding on first run): {REPO}")
t0 = time.time()
model = AutoModel.from_pretrained(REPO, max_seq_len=128)
print(f"model class: {type(model).__name__}  (setup {time.time()-t0:.1f}s)")

ids = model.tokenizer(PROMPT, return_tensors="np")["input_ids"]
x = mx.array(np.asarray(ids))
print(f"prompt={PROMPT!r}  input_ids shape={x.shape}")

t1 = time.time()
out = model.generate(x, max_new_tokens=12)
print(f"\n=== GENERATION ({time.time()-t1:.1f}s) ===")
print(f"{PROMPT}{out}")
print("\nOK: AirLLM MLX streaming path works on this Mac.")
