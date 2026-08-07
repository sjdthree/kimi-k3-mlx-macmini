"""M9 gate: full 93-layer parity, MLX vs the k3ref torch/CPU oracle.

Spec section 16 asks a release candidate to match the reference on:
  * top-1 token for every generated token
  * **router expert selection at every MoE layer**
  * bounded numerical error across all layers

The router check is the one that matters most. Cosine similarity can look perfect while a
single expert differs, and one different expert means a different computation from that
point on. So this compares expert INDICES exactly, at all 92 MoE layers, not with a
tolerance.

Both implementations run end-to-end independently -- each carries its own hidden state
through all 93 layers -- so any divergence accumulates rather than being reset each
layer. That is the honest test.

k3ref is torch/CPU fp32 and costs roughly 8 minutes per token at full depth, so token
count is a parameter.

    python tests/test_full_parity.py --tokens 1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.layer import LayerState, empty_block_residual  # noqa: E402
from air_llm.models.kimi_k3.loader import TensorSource, build_layer  # noqa: E402
from air_llm.models.kimi_k3.model import (  # noqa: E402
    EMBED, FINAL_NORM, LM_HEAD, OUT_RES_NORM, OUT_RES_PROJ, KimiK3Model)
from air_llm.models.kimi_k3.recurrent_state import KimiK3State  # noqa: E402
from k3ref.layers import (K3Config, DecoderLayer as RefLayer,  # noqa: E402
                          apply_attn_res as ref_res, rmsnorm as ref_norm)
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--prompt-tokens", default="1008,10484,318,15383,387")
    ap.add_argument("--ckpt", default="/Volumes/USB4_2TBB/Kimi-K3")
    ap.add_argument("--cache", default=str(Path("~/k3_dense").expanduser()))
    ap.add_argument("--manifest", default="manifest_local.jsonl.gz")
    ap.add_argument("--resident-gb", type=float, default=0.0)
    a = ap.parse_args()

    cfg = K3Config.load(Path(a.ckpt) / "config.json")
    ws = WeightStore()
    src = TensorSource(manifest=Path(a.manifest), ckpt=Path(a.ckpt),
                       cache=Path(a.cache), resident_gb=a.resident_gb,
                       weight_dtype=mx.float32)      # validation mode for parity
    model = KimiK3Model(cfg, src, max_memory_gb=48.0)

    ids = [int(t) for t in a.prompt_tokens.split(",")]
    print(f"M9 gate: full {cfg.num_hidden_layers}-layer parity, {a.tokens} token(s)")
    print(f"  prompt {ids}   device {mx.default_device()}")
    print("=" * 78)

    H = cfg.hidden_size
    ref_state: dict = {"kda": {}, "conv": {}, "mla": {}}
    mlx_state = KimiK3State()
    out_ref, out_mlx = [], []

    for step in range(a.tokens):
        step_ids = ids if step == 0 else [out_mlx[-1]]
        T = len(step_ids)

        emb = ws.get(EMBED)[step_ids].reshape(1, T, H)
        x_ref = emb.clone()
        x_mlx = mx.array(emb.numpy())
        br_ref = torch.zeros(T, 0, H)
        br_mlx = empty_block_residual(T, H)

        worst_cos, worst_L = 1.0, -1
        router_bad, router_checked = [], 0
        t0 = time.time()

        for L in range(cfg.num_hidden_layers):
            rl = RefLayer(cfg, L, ws)
            # k3ref populates diag["router"] with the router result computed INSIDE the
            # layer, on the true MoE input. Recomputing it outside from the layer's
            # OUTPUT routes on the wrong tensor entirely -- an earlier version of this
            # test did that and reported 0/84 routers matching, which was the test
            # being wrong, not the model.
            ref_diag: dict = {}
            x_ref, br_ref, ks, kc, mc = rl(
                x_ref, br_ref,
                state=ref_state["kda"].get(L), conv=ref_state["conv"].get(L),
                cache=ref_state["mla"].get(L), diag=ref_diag)
            if cfg.is_kda(L):
                ref_state["kda"][L], ref_state["conv"][L] = ks, kc
            else:
                ref_state["mla"][L] = mc
            r_router = ref_diag.get("router")
            del rl

            ml = build_layer(cfg, L, src)
            st = LayerState(kda_state=mlx_state.kda_states.get(L),
                            kda_conv=mlx_state.kda_conv.get(L))
            if not cfg.is_kda(L):
                from air_llm.models.kimi_k3.attention_mla import MLACache
                c = mlx_state.mla_caches.get(L)
                st.mla_cache = MLACache(*c) if c else None
            from air_llm.models.kimi_k3.layer import LayerDiag
            diag = LayerDiag()
            x_mlx, br_mlx, st = ml(x_mlx, br_mlx, state=st, diag=diag)
            if cfg.is_kda(L):
                mlx_state.kda_states[L], mlx_state.kda_conv[L] = st.kda_state, st.kda_conv
            else:
                mlx_state.mla_caches[L] = (st.mla_cache.key, st.mla_cache.value)
            mx.eval(x_mlx, br_mlx)
            del ml

            am, bm = np.array(x_mlx, copy=False), x_ref.detach().numpy()
            cos = float((am * bm).sum() / (np.linalg.norm(am) * np.linalg.norm(bm)))
            if cos < worst_cos:
                worst_cos, worst_L = cos, L

            if diag.router_indices is not None and r_router is not None:
                router_checked += 1
                i_m = np.sort(np.array(diag.router_indices, copy=False), axis=-1)
                i_r = np.sort(r_router.expert_indices.numpy(), axis=-1)
                if not np.array_equal(i_m, i_r):
                    router_bad.append(L)

            if L % 12 == 0 or L == cfg.num_hidden_layers - 1:
                print(f"    L{L:>2} cos {cos:.8f}  routers ok {router_checked-len(router_bad)}"
                      f"/{router_checked}  {time.time()-t0:6.0f}s", flush=True)

        # model-level tail
        x_ref = ref_res(x_ref.reshape(-1, H), br_ref, ws.get(OUT_RES_PROJ),
                        ws.get(OUT_RES_NORM), cfg.rms_norm_eps).reshape(1, T, H)
        x_ref = ref_norm(x_ref, ws.get(FINAL_NORM), cfg.rms_norm_eps)
        head = ws.get(LM_HEAD)
        tok_ref = int(torch.argmax((x_ref[:, -1, :].reshape(1, -1) @ head.T).reshape(-1)))
        del head

        from air_llm.models.kimi_k3.ops import apply_attn_res as mres, rmsnorm as mnorm
        x_mlx = mres(x_mlx.reshape(-1, H), br_mlx, model._g(OUT_RES_PROJ),
                     model._g(OUT_RES_NORM), cfg.rms_norm_eps).reshape(1, T, H)
        x_mlx = mnorm(x_mlx, model._g(FINAL_NORM), cfg.rms_norm_eps)
        tok_mlx, _, _ = model.logits(x_mlx[:, -1, :])

        out_ref.append(tok_ref)
        out_mlx.append(tok_mlx)
        mlx_state.advance(T)
        print(f"\n  token {step}: k3ref {tok_ref}  mlx {tok_mlx}  "
              f"({time.time()-t0:.0f}s)")
        check(f"token {step} top-1 matches the oracle", tok_ref == tok_mlx,
              f"{tok_ref} vs {tok_mlx}")
        check(f"token {step} router selection exact at all {router_checked} MoE layers",
              not router_bad, f"{len(router_bad)} layers differ"
              + (f": {router_bad[:6]}" if router_bad else ""))
        check(f"token {step} worst per-layer cosine >= 0.999", worst_cos >= 0.999,
              f"{worst_cos:.8f} at layer {worst_L}")

    print("\n" + "=" * 78)
    print(f"  k3ref : {out_ref}")
    print(f"  mlx   : {out_mlx}")
    check("all generated tokens identical", out_ref == out_mlx)
    print("=" * 78)
    print(f"M9: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
