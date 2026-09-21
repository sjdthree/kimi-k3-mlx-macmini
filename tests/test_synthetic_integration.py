"""Spec section 15 gate: the synthetic integration model, all execution paths.

Spec requires six paths over a miniature model that keeps every real operation, and that
ALL of them produce matching greedy output:

  1. eager full-memory execution
  2. layer streaming
  3. expert streaming
  4. standard MLX kernels
  5. fused Metal kernels
  6. saved-state continuation

This is the only test that can compare all paths against each other, because the 1.4 TB
checkpoint cannot be run six ways inside a test loop. It also covers a full attention
residual block period (attn_res_block_size=2, so layers 0 and 2 both open blocks) --
coverage the real-checkpoint tests cannot afford until the local copy lands.

    python tools/make_synthetic_k3.py --out /tmp/mini-k3
    python tools/k3_manifest.py build --ckpt /tmp/mini-k3 --out /tmp/mini-k3/manifest.jsonl.gz
    python tests/test_synthetic_integration.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import air_llm.models.kimi_k3.loader as loader  # noqa: E402
from air_llm.models.kimi_k3.loader import TensorSource  # noqa: E402
from air_llm.models.kimi_k3.model import KimiK3Model  # noqa: E402
from air_llm.models.kimi_k3.recurrent_state import KimiK3State  # noqa: E402
from k3ref.layers import K3Config  # noqa: E402

MINI = Path("/tmp/mini-k3")
MANIFEST = MINI / "manifest.jsonl.gz"
PROMPT = [7, 42, 128, 9]
NTOK = 6
FAILS = 0
_ORIG_BUILD = loader.build_layer


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def ensure_model():
    if not MANIFEST.exists():
        root = Path(__file__).resolve().parent.parent
        subprocess.run([sys.executable, str(root / "tools/make_synthetic_k3.py"),
                        "--out", str(MINI)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(root / "tools/k3_manifest.py"), "build",
                        "--ckpt", str(MINI), "--out", str(MANIFEST)],
                       check=True, capture_output=True)


class MemorySource(TensorSource):
    """Preload every tensor, including every expert and lm_head; no later disk reads."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.arrays = {name: super(MemorySource, self).raw(name) for name in self.recs}
        self.reset_counters()

    def raw(self, name):
        return self.arrays[name]

    def rows(self, name, indices):
        a, dtype = self.raw(name)
        a = a[indices]
        return (a.astype(np.uint32) << 16).view(np.float32) if dtype == "BF16" else a

    def row_block(self, name, start, n):
        return self.rows(name, slice(start, start + n))

    def _read_pair(self, *args):
        raise loader._NotAdjacent


def make(resident_gb=0.0, dtype=mx.float32, eager=False):
    cfg = K3Config.load(MINI / "config.json")
    src = (MemorySource if eager else TensorSource)(manifest=MANIFEST, ckpt=MINI,
                       resident_gb=resident_gb, weight_dtype=dtype)
    return cfg, src, KimiK3Model(cfg, src, max_memory_gb=48.0)


def use_fused(flag: bool):
    """Force every MoE block onto the fused or unfused expert path."""
    def wrapped(cfg, layer, src):
        lyr = _ORIG_BUILD(cfg, layer, src)
        if hasattr(lyr.mlp, "fused"):
            lyr.mlp.fused = flag
            assert lyr.mlp.fused is flag
        return lyr
    loader.build_layer = wrapped


def main():
    ensure_model()
    cfg = K3Config.load(MINI / "config.json")
    opens = [i for i in range(cfg.num_hidden_layers)
             if i % cfg.attn_res_block_size == 0]
    print(f"spec section 15 gate: synthetic integration model  "
          f"(device {mx.default_device()})")
    print(f"  {cfg.num_hidden_layers} layers ({len(cfg.mla_layers)} MLA / "
          f"{len(cfg.kda_layers)} KDA), {cfg.num_experts} experts top-"
          f"{cfg.num_experts_per_token}, {cfg.num_shared_experts} shared")
    print(f"  attn_res_block_size {cfg.attn_res_block_size}, block-opening layers {opens}")
    print("=" * 78)

    results = {}
    try:
        use_fused(False)
        _, _, m = make()
        results["4. standard MLX kernels"] = m.generate(list(PROMPT), NTOK)

        use_fused(True)
        _, _, m = make()
        results["5. fused Metal kernels"] = m.generate(list(PROMPT), NTOK)

        _, src_s, m = make(resident_gb=0.0)
        results["2+3. layer/expert streaming"] = m.generate(list(PROMPT), NTOK)
        stream_read, stream_proj = src_s.bytes_read, src_s.experts_loaded

        _, src_e, m = make(resident_gb=8.0, eager=True)
        results["1. eager full-memory"] = m.generate(list(PROMPT), NTOK)
        eager = src_e.cache_stats()
        check("eager generation reads no checkpoint bytes", src_e.bytes_read == 0)

        _, _, m = make(resident_gb=8.0, dtype=mx.bfloat16)
        results["   runtime mode (bf16)"] = m.generate(list(PROMPT), NTOK)

        base_name = "5. fused Metal kernels"
        base = results[base_name]
        print(f"\n  reference: {base_name} -> {base}\n")
        for name, out in results.items():
            check(f"{name:<28} matches", out == base, f"{out}")

        # ---- path 6: saved-state continuation
        _, _, m = make()
        state = KimiK3State()
        first, ids = [], list(PROMPT)
        for i in range(3):
            step = ids if i == 0 else [first[-1]]
            h = m.forward_token(m.embed(step), state)
            tok, _, _ = m.logits(h[:, -1, :])
            first.append(tok)
            state.advance(len(step))

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "s.npz"
            state.save(p)
            restored = KimiK3State.load(p)

        cont = list(first)
        for _ in range(NTOK - 3):
            h = m.forward_token(m.embed([cont[-1]]), restored)
            tok, _, _ = m.logits(h[:, -1, :])
            cont.append(tok)
            restored.advance(1)
        check(f"{'6. saved-state continuation':<28} matches", cont == base, f"{cont}")

        # ---- streaming behaviour, exact on a model small enough to reason about
        n_moe = cfg.num_hidden_layers - cfg.first_k_dense_replace
        print(f"\n       strict streaming: {stream_read/2**20:.2f} MiB read, "
              f"{stream_proj} expert projections over {NTOK} tokens")
        print(f"       eager cache: {eager['hit_rate']*100:.1f}% hit rate, "
              f"{eager['entries']} tensors resident")
        check("eager path reaches a high cache hit rate", eager["hit_rate"] > 0.7,
              f"{eager['hit_rate']*100:.1f}%")

        _, src2, m2 = make()
        st = KimiK3State()
        src2.reset_counters()
        mx.eval(m2.forward_token(m2.embed([7]), st))
        want = cfg.num_experts_per_token * 3 * n_moe
        check("expert projections per token == top_k x 3 x MoE layers",
              src2.experts_loaded == want, f"{src2.experts_loaded} vs {want}")

        # the whole point of a 3-layer model: a complete residual block period runs
        check("block-opening layers cover a full attn_res period", opens == [0, 2],
              f"{opens}")
    finally:
        loader.build_layer = _ORIG_BUILD

    print("=" * 78)
    print(f"SYNTHETIC: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
