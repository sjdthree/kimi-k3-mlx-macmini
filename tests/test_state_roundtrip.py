"""Spec section 12 gate: state save/restore, and the conv-layout trap specifically.

The decisive test is not "does the round-trip preserve bytes" -- it is "does a run that
was interrupted, saved, and restored produce the SAME continuation as an uninterrupted
one". A transposed conv window survives a byte round-trip perfectly and still corrupts
generation three tokens later.

    python tests/test_state_roundtrip.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from air_llm.models.kimi_k3.attention_kda import KDAAttention as KDAmlx  # noqa: E402
from air_llm.models.kimi_k3.recurrent_state import (  # noqa: E402
    CANONICAL_CONV_LAYOUT, KimiK3State, conv_from_canonical, conv_to_canonical, fmt_bytes)
from k3ref.layers import K3Config  # noqa: E402
from k3ref.weights import WeightStore  # noqa: E402

FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def load_mx(ws, layer, keys):
    return {k: mx.array(np.ascontiguousarray(
        ws.get(f"language_model.model.layers.{layer}.self_attn.{k}").numpy())) for k in keys}


def main():
    cfg = K3Config.load()
    ws = WeightStore()
    L = cfg.kda_layers[1]
    kda = KDAmlx(cfg, load_mx(ws, L, KDAmlx.KEYS))
    print(f"state gate: KDA layer {L}   (device {mx.default_device()})")
    print("=" * 72)

    torch.manual_seed(0)
    T = 8
    x = mx.array((torch.randn(1, T, cfg.hidden_size) * 0.02).numpy())

    # ---- uninterrupted reference run
    st = conv = None
    ref_outs = []
    for i in range(T):
        o, st, conv = kda(x[:, i:i + 1], state=st, conv=conv)
        ref_outs.append(o)
    ref = mx.concatenate(ref_outs, axis=1)
    mx.eval(ref)

    # ---- interrupted run: stop after 4 tokens, SAVE, restore, continue
    st = conv = None
    outs = []
    for i in range(4):
        o, st, conv = kda(x[:, i:i + 1], state=st, conv=conv)
        outs.append(o)

    state = KimiK3State(token_position=4)
    state.kda_states[L] = st
    state.kda_conv[L] = conv
    state.block_residual = mx.zeros((1, 1, cfg.hidden_size))

    rep = state.memory_report()
    print("  state memory by component:")
    for k, v in rep.items():
        print(f"      {k:<16} {fmt_bytes(v)}")

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "state.npz"
        state.save(p)
        size = p.stat().st_size
        back = KimiK3State.load(p)

        check("save/load preserves token_position", back.token_position == 4)
        d_state = float(mx.max(mx.abs(back.kda_states[L] - st)))
        check("recurrent state survives the round-trip exactly", d_state == 0.0,
              f"max|d| {d_state:.3e}")

        # the trap: on-disk must be channels-FIRST, in-memory channels-LAST
        z = np.load(p)
        on_disk = z["kda_conv.%d.q" % L]
        in_mem = np.array(conv[0], copy=False)
        check("on-disk conv layout is canonical channels-first",
              on_disk.shape == (in_mem.shape[0], in_mem.shape[2], in_mem.shape[1]),
              f"disk {on_disk.shape} vs mlx {in_mem.shape}")
        d_conv = max(float(mx.max(mx.abs(b - c))) for b, c in zip(back.kda_conv[L], conv))
        check("conv state restores to the MLX layout exactly", d_conv == 0.0,
              f"max|d| {d_conv:.3e}")

        # ---- continue from restored state; must match the uninterrupted run
        st2, conv2 = back.kda_states[L], back.kda_conv[L]
        for i in range(4, T):
            o, st2, conv2 = kda(x[:, i:i + 1], state=st2, conv=conv2)
            outs.append(o)
        got = mx.concatenate(outs, axis=1)
        mx.eval(got)
        d = float(mx.max(mx.abs(got - ref)))
        check("restored continuation == uninterrupted run (spec section 12)", d == 0.0,
              f"max|d| {d:.3e}")
        print(f"       state file {fmt_bytes(size)} for 1 KDA layer; "
              f"x69 layers ~= {fmt_bytes(size*69)}")

    # ---- the trap itself: a transposed conv window must NOT silently pass
    bad = tuple(mx.array(np.array(c, copy=False)) for c in conv)
    wrong = tuple(mx.array(np.ascontiguousarray(conv_to_canonical(c))) for c in conv)
    st3, conv3 = st, wrong
    try:
        o, _, _ = kda(x[:, 4:5], state=st3, conv=conv3)
        mx.eval(o)
        check("a channels-first conv state is rejected or visibly wrong (not silent)",
              False, "it ran and produced a value")
    except Exception as e:
        check("a channels-first conv state fails loudly rather than silently",
              True, f"{type(e).__name__}")

    # ---- layout helpers are exact inverses
    rt = conv_from_canonical(conv_to_canonical(conv[0]))
    check("conv_from_canonical(conv_to_canonical(x)) == x",
          float(mx.max(mx.abs(rt - conv[0]))) == 0.0)

    # ---- a file claiming a different layout must be refused
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "s.npz"
        state.save(p)
        z = dict(np.load(p))
        import json as _json
        m = _json.loads(bytes(z["__meta__"]).decode())
        m["conv_layout"] = "channels_last"
        z["__meta__"] = np.frombuffer(_json.dumps(m).encode(), dtype=np.uint8)
        np.savez(p, **z)
        try:
            KimiK3State.load(p)
            check("loading a foreign conv layout is refused", False, "it loaded")
        except ValueError:
            check("loading a foreign conv layout is refused", True)

    print("=" * 72)
    print(f"STATE: {'PASS' if FAILS == 0 else f'FAIL ({FAILS})'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
