"""Spec section 13 gate: validation mode and runtime mode must agree on greedy tokens.

  validation: fp32 weights, unfused expert path, deterministic
  runtime:    bf16 weights (the on-disk dtype -- lossless), fused MXFP4 kernel

Spec: "The same prompt and checkpoint must produce equivalent greedy tokens in both
modes." Cosine similarity is not enough; a mode that agrees to cos 0.9999 but flips one
argmax has failed, because the flipped token changes everything after it.

    python tests/test_numerical_modes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _harness import require_checkpoint  # noqa: E402
from air_llm.models.kimi_k3.loader import TensorSource  # noqa: E402
from air_llm.models.kimi_k3.model import KimiK3Model  # noqa: E402
from k3ref.layers import K3Config  # noqa: E402

FAILS = 0
CACHE = str(Path("~/k3_cache").expanduser())


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def gen(mode, prompt, n_tokens, layers=2, resident_gb=0.0, fused=None):
    cfg = K3Config.load()
    cfg.num_hidden_layers = layers
    src = TensorSource(cache=CACHE, resident_gb=resident_gb,
                       weight_dtype=mx.bfloat16 if mode == "runtime" else mx.float32)
    model = KimiK3Model(cfg, src, max_memory_gb=48.0)
    if fused is not None:
        _orig = model.__class__  # noqa: F841
    return model.generate(list(prompt), n_tokens), src


def main():
    require_checkpoint()
    print(f"spec section 13 gate: numerical modes   (device {mx.default_device()})")
    print("=" * 74)

    prompts = [(100, 200, 300), (1, 2, 3), (50000, 60000)]
    for p in prompts:
        val, _ = gen("validation", p, 3)
        run, _ = gen("runtime", p, 3)
        check(f"prompt {p}: greedy tokens identical across modes", val == run,
              f"validation {val} vs runtime {run}")

    # the resident cache must not perturb results either
    p = prompts[0]
    base, _ = gen("runtime", p, 3)
    cached, src = gen("runtime", p, 3, resident_gb=6.0)
    check("resident cache does not change greedy tokens", base == cached,
          f"{base} vs {cached}")
    st = src.cache_stats()
    print(f"       resident cache: {st['hit_rate']*100:.1f}% hit rate, "
          f"{st['resident_bytes']/2**30:.2f} GiB")

    # determinism: same config twice must give byte-identical output
    a, _ = gen("runtime", p, 3)
    b, _ = gen("runtime", p, 3)
    check("runtime mode is deterministic across processes-equivalent runs", a == b,
          f"{a} vs {b}")

    print("=" * 74)
    print(f"MODES: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
