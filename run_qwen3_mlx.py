"""Run Qwen3-32B through AirLLM's Mac/MLX layer-streaming path.

airllm 3.1.0's MLX backend only models Llama-2/3 attention, so Qwen3 fails three ways:
  1. no per-head QK-norm (q_norm/k_norm)  -> "Module does not have parameter named q_norm"
  2. head_dim inferred as hidden/n_heads (5120/64=80) but Qwen3's real head_dim is 128
  3. rope_theta defaulted to 1e4 but Qwen3 uses 1e6

This module monkeypatches AirLLM's MLX classes with a Qwen3-correct Attention/TransformerBlock
and model-args, then reuses the ALREADY re-sharded weights on disk (no re-download/reshard).
"""
import sys, time
import numpy as np
import mlx.core as mx
import mlx.nn as nn

import airllm.airllm_llama_mlx as amlx
from airllm.airllm_llama_mlx import RMSNorm, ModelArgs, FeedForward, AirLLMLlamaMlx

REPO = "Qwen/Qwen3-32B"
PROMPT = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is"
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 12


class Qwen3Attention(nn.Module):
    """Llama-style GQA attention + Qwen3 per-head RMSNorm on Q and K."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.repeats = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        # Qwen3: RMSNorm over head_dim, applied per head
        self.q_norm = RMSNorm(args.head_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(args.head_dim, eps=args.norm_eps)
        self.rope = nn.RoPE(args.head_dim, traditional=args.rope_traditional, base=args.rope_theta)

    def __call__(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        # reshape to (B, L, H, D), apply QK-norm over D, then to (B, H, L, D)
        queries = self.q_norm(self.wq(x).reshape(B, L, self.n_heads, self.head_dim)).transpose(0, 2, 1, 3)
        keys = self.k_norm(self.wk(x).reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        values = self.wv(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # expand GQA kv heads to full head count
        keys = mx.concatenate([mx.expand_dims(keys, 2)] * self.repeats, axis=2).reshape(B, self.n_heads, L, self.head_dim)
        values = mx.concatenate([mx.expand_dims(values, 2)] * self.repeats, axis=2).reshape(B, self.n_heads, L, self.head_dim)

        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores += mask
        scores = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
        output = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.wo(output), (keys, values)


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention = Qwen3Attention(args)
        self.feed_forward = FeedForward(args=args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def __call__(self, x, mask=None, cache=None):
        r, cache = self.attention(self.attention_norm(x), mask, cache)
        h = x + r
        r = self.feed_forward(self.ffn_norm(h))
        return h + r, cache


def qwen3_model_args(config):
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", None) or (config.hidden_size // n_heads)
    return ModelArgs(
        dim=config.hidden_size,
        n_layers=config.num_hidden_layers,
        head_dim=head_dim,
        hidden_dim=config.intermediate_size,
        n_heads=n_heads,
        n_kv_heads=getattr(config, "num_key_value_heads", n_heads),
        norm_eps=config.rms_norm_eps,
        vocab_size=config.vocab_size,
        rope_theta=getattr(config, "rope_theta", 1000000.0),
        rope_traditional=False,
    )


# --- install the patch (used by AirLLMLlamaMlx.__init__ and .model_generate via module globals) ---
amlx.TransformerBlock = Qwen3TransformerBlock
amlx.get_model_args_from_config = qwen3_model_args

if __name__ == "__main__":
    print(f"MLX device: {mx.default_device()}")
    print(f"Loading {REPO} (reusing existing re-shard)...")
    t0 = time.time()
    model = AirLLMLlamaMlx(REPO, max_seq_len=128)
    ma = model.model_args
    print(f"model_args: head_dim={ma.head_dim} n_heads={ma.n_heads} n_kv={ma.n_kv_heads} "
          f"rope_theta={ma.rope_theta} n_layers={ma.n_layers}  (setup {time.time()-t0:.1f}s)")

    ids = model.tokenizer(PROMPT, return_tensors="np")["input_ids"]
    x = mx.array(np.asarray(ids))
    print(f"prompt={PROMPT!r} input_ids shape={x.shape}")

    t1 = time.time()
    out = model.generate(x, max_new_tokens=MAX_NEW)
    dt = time.time() - t1
    print(f"\n=== GENERATION ({dt:.1f}s, {dt/max(MAX_NEW,1):.1f}s/token) ===")
    print(f"{PROMPT}{out}")
    print("\nOK: Qwen3-32B ran on AirLLM's Mac/MLX streaming path.")
