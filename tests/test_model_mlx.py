"""M8 (partial) gate: the MODEL-level plumbing vs k3ref.

Layer parity is covered by test_layer_mlx.py. What is only exercised at model level:
  * the model-wide output attention residual (_apply_output_attn_res)
  * the final RMSNorm
  * the streamed lm_head argmax (chunked row blocks with a running max)
  * block_residual being rebuilt per forward pass rather than carried

Runs with --layers N so it is usable before the full checkpoint is local.

    python tests/test_model_mlx.py --layers 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.loader import TensorSource  # noqa: E402
from air_llm.models.kimi_k3.model import (  # noqa: E402
    EMBED, FINAL_NORM, LM_HEAD, OUT_RES_NORM, OUT_RES_PROJ, KimiK3Model)
from air_llm.models.kimi_k3.recurrent_state import KimiK3State  # noqa: E402
from k3ref.layers import (K3Config, DecoderLayer as RefLayer,  # noqa: E402
                          apply_attn_res as ref_attn_res, rmsnorm as ref_rmsnorm)
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--cache", default=str(Path("~/k3_cache").expanduser()))
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    cfg = K3Config.load()
    cfg.num_hidden_layers = a.layers
    ws = WeightStore()
    src = TensorSource(ckpt=a.ckpt, cache=a.cache)
    model = KimiK3Model(cfg, src, max_memory_gb=48.0)

    print(f"M8 (partial) gate: model plumbing, {a.layers} layers   "
          f"(device {mx.default_device()})")
    print("=" * 72)

    ids = [100, 200, 300]

    # ---- embedding: row read must equal the real table rows
    emb_mlx = model.embed(ids)
    tbl = ws.get(EMBED)
    emb_ref = tbl[ids].reshape(1, len(ids), cfg.hidden_size)
    d = np.abs(np.array(emb_mlx, copy=False) - emb_ref.numpy()).max()
    check("embedding row read matches the full table", d == 0.0, f"max|d| {d:.3e}")

    # ---- reference forward through the same layers, then model-level ops
    x_ref = emb_ref.clone()
    br_ref = torch.zeros(len(ids), 0, cfg.hidden_size)
    for L in range(a.layers):
        layer = RefLayer(cfg, L, ws)
        x_ref, br_ref, _, _, _ = layer(x_ref, br_ref)
    H = cfg.hidden_size
    x_ref = ref_attn_res(x_ref.reshape(-1, H), br_ref,
                         ws.get(OUT_RES_PROJ), ws.get(OUT_RES_NORM),
                         cfg.rms_norm_eps).reshape(1, len(ids), H)
    x_ref = ref_rmsnorm(x_ref, ws.get(FINAL_NORM), cfg.rms_norm_eps)

    # ---- MLX model forward
    state = KimiK3State()
    x_mlx = model.forward_token(emb_mlx, state)
    mx.eval(x_mlx)

    am, bm = np.array(x_mlx, copy=False), x_ref.detach().numpy()
    d = np.abs(am - bm).max()
    cos = float((am * bm).sum() / (np.linalg.norm(am) * np.linalg.norm(bm)))
    check("full model forward (incl. output attn-res + final norm) matches k3ref",
          cos >= 0.999, f"cos {cos:.8f}  max|d| {d:.3e}")

    # ---- streamed lm_head argmax vs a direct full-matrix argmax
    head = ws.get(LM_HEAD)
    logits_ref = (x_ref[:, -1, :].reshape(1, -1) @ head.T).reshape(-1)
    want_id = int(torch.argmax(logits_ref))
    got_id, got_val, nbytes = model.logits(x_mlx[:, -1, :])
    check("streamed lm_head argmax == full-matrix argmax", got_id == want_id,
          f"got {got_id} want {want_id}")
    dv = abs(got_val - float(logits_ref[want_id]))
    check("streamed lm_head max logit matches", dv < 1e-2,
          f"|d| {dv:.3e} on {float(logits_ref[want_id]):.4f}")
    print(f"       lm_head streamed in {model.lm_head_chunk}-row chunks, "
          f"{nbytes/2**30:.2f} GiB read, never materialised whole")

    # ---- block_residual must NOT be carried across forward passes
    st2 = KimiK3State()
    _ = model.forward_token(model.embed(ids), st2)
    x_second = model.forward_token(model.embed([500]), st2)   # 1 token after 3
    mx.eval(x_second)
    check("a 1-token pass after a 3-token pass succeeds (block_residual is per-pass)",
          x_second.shape == (1, 1, H), f"shape {tuple(x_second.shape)}")

    # ---- state grew as expected
    rep = st2.memory_report()
    n_kda = sum(1 for L in range(a.layers) if cfg.is_kda(L))
    check("state holds one entry per KDA layer",
          len(st2.kda_states) == n_kda, f"{len(st2.kda_states)} of {n_kda}")
    print(f"       state after 4 tokens: {rep['total']/2**20:.1f} MiB "
          f"(kda {rep['kda_recurrent']/2**20:.1f} + conv {rep['kda_conv']/2**20:.1f} "
          f"+ mla {rep['mla_cache']/2**20:.1f})")

    print("=" * 72)
    print(f"M8-partial: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
