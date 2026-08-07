"""Spec section 19: the required run interface.

    python -m air_llm.run \
      --model /Volumes/USB4_2TBB/Kimi-K3 \
      --architecture kimi_k3 \
      --backend mlx \
      --stream-experts \
      --max-memory-gb 48 \
      --prompt-file prompts/test.txt \
      --max-new-tokens 16 \
      --temperature 0 \
      --validation-log results/run.jsonl

`--max-memory-gb` defaults to 48, not the spec's 112: this host has 64 GiB of unified
memory total (K3_MLX_PLAN.md section 1 row 5).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx

from .models.kimi_k3.loader import TensorSource
from .models.kimi_k3.model import KimiK3Model


def load_config(model_dir: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from k3ref.layers import K3Config
    return K3Config.load(model_dir / "config.json")


def tokenize(model_dir: Path, text: str) -> list[int]:
    """Build the tiktoken Encoding directly (air_llm...tokenizer), avoiding
    trust_remote_code and any transformers-version coupling. Verified to produce the
    same ids as the shipped tokenizer."""
    from .models.kimi_k3.tokenizer import load as load_tok
    try:
        return load_tok(model_dir).encode(text)
    except Exception as e:
        raise SystemExit(
            f"could not load the K3 tokenizer ({type(e).__name__}: {e}).\n"
            f"Pass --prompt-tokens with explicit ids instead, e.g. "
            f"--prompt-tokens 1,2,3")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="air_llm.run")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--architecture", default="kimi_k3", choices=["kimi_k3"])
    ap.add_argument("--backend", default="mlx", choices=["mlx"])
    ap.add_argument("--stream-experts", action="store_true", default=True)
    ap.add_argument("--max-memory-gb", type=float, default=48.0)
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--prompt", type=str)
    ap.add_argument("--prompt-tokens", type=str, help="comma-separated token ids")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--validation-log", type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("manifest.jsonl.gz"))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--check-finite", action="store_true",
                    help="spec section 16: assert no NaN/inf in hidden states, recurrent "
                         "states, routing weights, or logits (costs a sync per layer)")
    ap.add_argument("--layers", type=int, default=None,
                    help="run only the first N layers (debugging)")
    ap.add_argument("--resident-gb", type=float, default=0.0,
                    help="hold this many GiB of dense weights resident (0 = strict "
                         "streaming). Dense weights are identical every token, so this "
                         "removes proportional I/O; results are unchanged.")
    ap.add_argument("--mode", choices=["validation", "runtime"], default="validation",
                    help="spec section 13. validation: fp32 weights, no fused kernels, "
                         "exact parity with k3ref. runtime: bf16 weights (the on-disk "
                         "dtype, lossless) + fused kernels, half the resident memory.")
    a = ap.parse_args(argv)

    if a.temperature != 0.0:
        raise SystemExit("only greedy decoding (--temperature 0) is implemented")

    cfg = load_config(a.model)
    if a.layers:
        cfg.num_hidden_layers = a.layers

    if a.prompt_tokens:
        ids = [int(t) for t in a.prompt_tokens.split(",") if t.strip()]
    else:
        text = a.prompt or (a.prompt_file.read_text().strip() if a.prompt_file else None)
        if not text:
            raise SystemExit("need --prompt, --prompt-file, or --prompt-tokens")
        ids = tokenize(a.model, text)

    wdtype = mx.bfloat16 if a.mode == "runtime" else mx.float32
    src = TensorSource(manifest=a.manifest, ckpt=a.model, cache=a.cache,
                       resident_gb=a.resident_gb, weight_dtype=wdtype)
    model = KimiK3Model(cfg, src, max_memory_gb=a.max_memory_gb,
                        diagnostic=a.diagnostic, check_finite=a.check_finite)

    print(f"Kimi-K3 / mlx / {mx.default_device()}")
    print(f"  checkpoint {a.model}")
    print(f"  {cfg.num_hidden_layers} layers ({len(cfg.mla_layers)} MLA, "
          f"{len(cfg.kda_layers)} KDA), {cfg.num_experts} experts top-"
          f"{cfg.num_experts_per_token}")
    print(f"  prompt {len(ids)} tokens {ids[:12]}"
          f"{'...' if len(ids) > 12 else ''}, generating {a.max_new_tokens} greedy")
    print(f"  memory ceiling {a.max_memory_gb} GiB | mode {a.mode} "
          f"({'bf16' if a.mode == 'runtime' else 'fp32'} weights)"
          f" | resident budget {a.resident_gb} GiB")
    if a.validation_log:
        a.validation_log.parent.mkdir(parents=True, exist_ok=True)

    def on_token(rec):
        print(f"  [{rec.token_index:>2}] id={rec.token_id:<7} "
              f"{rec.elapsed_seconds:7.1f}s  read {rec.bytes_read/2**30:6.2f} GiB  "
              f"peak {rec.peak_unified_memory/2**30:5.2f} GiB  "
              f"experts {rec.experts_loaded}  state {rec.cache_bytes/2**20:.0f} MiB"
              + (f"  NONFINITE[{rec.nonfinite_where}]" if rec.nonfinite else ""),
              flush=True)

    t0 = time.time()
    out = model.generate(ids, a.max_new_tokens, log_path=a.validation_log,
                         on_token=on_token)
    dt = time.time() - t0
    print(f"\ngenerated {len(out)} tokens in {dt/60:.1f} min "
          f"({dt/max(len(out),1):.1f}s/token)")
    print(f"token ids: {out}")
    if not a.prompt_tokens:
        try:
            from .models.kimi_k3.tokenizer import load as load_tok
            tk = load_tok(a.model)
            print(f"continuation: {tk.decode(out)!r}")
            print(f"full        : {tk.decode(list(ids) + out)!r}")
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
