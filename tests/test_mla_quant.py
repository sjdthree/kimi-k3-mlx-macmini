"""Quality gates for QUANTISED MLA latent storage (mla_quant.py). Off by default.

Latent caching already won 53.3x losslessly. Quantisation is the first thing in this
area that trades accuracy for bytes, so it is held to end-to-end token agreement, not
just reconstruction cosine.

Why cosine alone would be misleading here: every one of the 96 heads reads the SAME
latent through kv_b, and kv_b feeds the VALUE half as well as the key half. So latent
error is common-mode across heads (it does not average out under softmax) and part of it
lands on the output without passing through a softmax at all. A latent that reconstructs
at cos 0.995 can still move greedy tokens.

WHAT THIS HARNESS CAN AND CANNOT TELL YOU
-----------------------------------------
Q1, Q2 and Q6 run on RANDOM weights and random inputs. That makes the latent a random
projection of random input, i.e. Gaussian by the central limit theorem -- and Q6 prints
the kurtosis so this is visible rather than assumed. Real transformer activations are
not Gaussian; they are known for heavy-tailed outlier channels, which is precisely the
condition that rotation-based schemes exist to fix.

So these numbers are a REGRESSION harness, not a quality verdict:

    the cosine floors below are set to catch a BUG -- a wrong group size, mis-packed
    codes, a dequantisation that silently drops a scale -- all of which collapse cosine
    far below any of these thresholds. They are not an endorsement of any bit width.

Deciding an actual operating point needs real latents from the real checkpoint. Those
runs are blocked on the checkpoint volume being free (see tests marked REAL-CKPT), and
Q6 in particular should be treated as UNRESOLVED until then.

Gates
-----
    Q0  the default configuration is NOT quantised
    Q1  latent reconstruction quality per bit width      (regression floor)
    Q2  attention-output deviation vs the lossless path  (regression floor)
    Q3  end-to-end greedy token agreement, synthetic model
    Q4  storage frontier at real Kimi-K3 dimensions      (exact arithmetic)
    Q5  a quantised cache survives save/load             (exact)
    Q6  does the article's "rotate, then quantise" insight transfer?  (INCONCLUSIVE
        on random weights -- see above)

Run:
    source .venv/bin/activate && python tests/test_mla_quant.py
"""
from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_mla_cache import (MLACfg, REAL_LATENT, REAL_MLA_LAYERS, as_mx,  # noqa: E402
                            oracle_mla, random_weights, run_new)

FAILS = 0
BIT_WIDTHS = (8, 6, 4, 3, 2)

# REGRESSION floors, not quality targets -- see the module docstring. Each sits roughly
# an order of magnitude below the value measured on a correct implementation, so a real
# packing/scaling bug trips it while ordinary numerical drift does not. Widths 2 and 3
# are reported without a floor: nothing here justifies shipping them.
OUT_COS_MIN = {8: 0.9999, 6: 0.999, 4: 0.99}


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return ok


def cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def cfg_with(cfg, **kw):
    c = copy.copy(cfg)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# --------------------------------------------------------------------------- Q0

def gate_Q0(cfg, wmx, x_mx):
    print("\nQ0 -- quantisation is opt-in")
    _, cache = run_new(cfg, wmx, x_mx)
    check("a default cache stores the latent unquantised", not cache.quantised,
          f"latent is {type(cache.latent).__name__}")
    check("bytes/token is the fp32 latent size",
          abs(cache.bytes_per_token() - REAL_LATENT * 4) < 1e-9,
          f"{cache.bytes_per_token():.0f} B/tok/layer")


# --------------------------------------------------------------------------- Q1

def gate_Q1(cfg, wmx, x_mx):
    """Reconstruction of the cached latent itself."""
    from air_llm.models.kimi_k3.mla_quant import QuantLatent, pick_group_size
    print("\nQ1 -- latent reconstruction")
    _, ref = run_new(cfg, wmx, x_mx)
    truth = ref.latent[:, :ref.length]
    B, T, r = truth.shape
    gs = pick_group_size(r)
    for bits in BIT_WIDTHS:
        q = QuantLatent(B, T, r, bits, gs)
        q.write(0, truth)
        c = cos(q.read(T), truth)
        print(f"       {bits}-bit  group {gs}   {q.bytes_per_token():7.1f} B/tok/layer   "
              f"cos {c:.7f}")


# --------------------------------------------------------------------------- Q2

def gate_Q2(cfg, w, wmx, x_np, x_mx):
    """What actually matters at this layer: deviation of the attention OUTPUT."""
    print("\nQ2 -- attention output vs the lossless path")
    exact = np.asarray(oracle_mla(cfg, w, x_np), dtype=np.float64)
    base, _ = run_new(cfg, wmx, x_mx)
    print(f"       fp32 (shipped)                       cos {cos(base, exact):.7f}")
    for bits in BIT_WIDTHS:
        got, cache = run_new(cfg_with(cfg, mla_cache_bits=bits), wmx, x_mx)
        c = cos(got, exact)
        line = (f"       {bits}-bit  {cache.bytes_per_token():6.1f} B/tok/layer  "
                f"({REAL_LATENT * 4 / cache.bytes_per_token():4.1f}x)   cos {c:.7f}")
        if bits in OUT_COS_MIN:
            check(line.strip(), c >= OUT_COS_MIN[bits], f"floor {OUT_COS_MIN[bits]}")
        else:
            print(line + "   (measured, not gated)")


# --------------------------------------------------------------------------- Q3

MINI = Path("/tmp/mini-k3")
MANIFEST = MINI / "manifest.jsonl.gz"


def gate_Q3(ntok: int = 24):
    """Greedy token agreement on the synthetic model -- the gate that decides.

    The synthetic model has ONE MLA layer out of three, so this understates the effect
    that 24-of-93 MLA layers would produce on the real checkpoint. Read a divergence
    here as a lower bound on the real risk, never as an upper bound.
    """
    import air_llm.models.kimi_k3.loader as loader
    from air_llm.models.kimi_k3.loader import TensorSource
    from air_llm.models.kimi_k3.model import KimiK3Model
    from k3ref.layers import K3Config
    print(f"\nQ3 -- end-to-end greedy tokens on the synthetic model ({ntok} tokens)")

    if not MANIFEST.exists():
        root = Path(__file__).resolve().parent.parent
        subprocess.run([sys.executable, str(root / "tools/make_synthetic_k3.py"),
                        "--out", str(MINI)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(root / "tools/k3_manifest.py"), "build",
                        "--ckpt", str(MINI), "--out", str(MANIFEST)],
                       check=True, capture_output=True)

    def gen(bits):
        cfg = K3Config.load(MINI / "config.json")
        if bits:
            cfg.mla_cache_bits = bits
            cfg.mla_cache_group_size = 32     # kv_lora_rank is 32 in the mini model
        src = TensorSource(manifest=MANIFEST, ckpt=MINI, resident_gb=0.0,
                           weight_dtype=mx.float32)
        return KimiK3Model(cfg, src, max_memory_gb=48.0).generate([7, 42, 128, 9], ntok)

    ref = gen(None)
    print(f"       fp32 reference: {ref}")
    for bits in BIT_WIDTHS:
        got = gen(bits)
        same = got == ref
        first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None)
        detail = "identical" if same else f"first divergence at token {first}"
        agree = sum(a == b for a, b in zip(ref, got)) / len(ref)
        line = f"{bits}-bit greedy tokens: {detail} ({agree:.0%} agree)"
        if bits in OUT_COS_MIN:
            check(line, same)
        else:
            print(f"       {line}   (measured, not gated)")


# --------------------------------------------------------------------------- Q4

def gate_Q4():
    print("\nQ4 -- storage frontier at real Kimi-K3 dimensions (24 MLA layers)")
    master = 2_949_120
    fp32 = REAL_MLA_LAYERS * REAL_LATENT * 4
    rows = [("master (expanded, fp32)", master), ("latent fp32 (shipped)", fp32)]
    for bits in BIT_WIDTHS:
        # 512 latent at `bits` + 8 groups x (scale+bias fp32) + 64-wide rope at fp32
        per = REAL_MLA_LAYERS * ((512 * bits / 8) + 8 * 8 + 64 * 4)
        rows.append((f"latent {bits}-bit", per))
    print(f"       {'':<26}{'B/token':>10}{'vs master':>11}"
          f"{'32k ctx':>10}{'128k ctx':>10}")
    for name, per in rows:
        print(f"       {name:<26}{per:>10,.0f}{master/per:>10.1f}x"
              f"{32768*per/2**30:>9.2f}G{131072*per/2**30:>9.2f}G")
    rope = REAL_MLA_LAYERS * 64 * 4
    print(f"\n       note: the fp32 rope slice is {rope:,} B/token of every row above --"
          f"\n       {rope/rows[-1][1]:.0%} of the 2-bit row. The frontier flattens because the"
          f"\n       latent shrinks and the rope slice does not. Anyone turning this on"
          f"\n       should quantise or bf16 the rope slice first; it is cheaper and safer"
          f"\n       than the last two bits of the latent.")


# --------------------------------------------------------------------------- Q5

def gate_Q5(cfg, wmx, x_mx):
    from air_llm.models.kimi_k3.recurrent_state import KimiK3State
    print("\nQ5 -- a quantised cache survives save/load")
    c4 = cfg_with(cfg, mla_cache_bits=4)
    T, split = x_mx.shape[1], 24

    _, straight = run_new(c4, wmx, x_mx[:, :split])
    for t in range(split, T):
        o_ref, straight = run_new(c4, wmx, x_mx[:, t:t + 1], cache=straight)

    _, cache = run_new(c4, wmx, x_mx[:, :split])
    st = KimiK3State(token_position=split)
    st.mla_caches[0] = cache
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "q.npz"
        st.save(p)
        back = KimiK3State.load(p)
        size = p.stat().st_size
    resumed = back.mla_caches[0]
    check("reloads as a quantised cache (codes kept, not dequantised)",
          resumed.quantised, f"{type(resumed.latent).__name__}, file {size/1024:.1f} KiB")
    for t in range(split, T):
        o_got, resumed = run_new(c4, wmx, x_mx[:, t:t + 1], cache=resumed)
    check("continuation after reload is bit-identical",
          float(np.abs(np.asarray(o_got, np.float64)
                       - np.asarray(o_ref, np.float64)).max()) == 0.0)


# --------------------------------------------------------------------------- Q6

def _fwht(a):
    """In-place fast Walsh-Hadamard transform along the last axis (power-of-two)."""
    a = a.copy()
    n = a.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[..., i:i + h].copy()
            y = a[..., i + h:i + h * 2].copy()
            a[..., i:i + h] = x + y
            a[..., i + h:i + h * 2] = x - y
        h *= 2
    return a / np.sqrt(n)


def gate_Q6(cfg, wmx, x_mx):
    """The article's central claim is that rotating keys before quantising buys a large
    accuracy gain, because the rotated distribution is near-Gaussian and spends its bits
    better. Test it directly on THIS model's latents rather than assuming it transfers.
    """
    from air_llm.models.kimi_k3.mla_quant import QuantLatent, pick_group_size
    print("\nQ6 -- does a randomised Hadamard rotation help these latents?  [INCONCLUSIVE]")
    print("       random weights make the latent Gaussian by CLT, which is the one case")
    print("       where rotation cannot help. Watch the kurtosis: if it prints ~3.0 the")
    print("       experiment has not tested the article's claim. Needs real latents.")
    _, ref = run_new(cfg, wmx, x_mx)
    truth = np.asarray(ref.latent[:, :ref.length], dtype=np.float32)
    B, T, r = truth.shape
    gs = pick_group_size(r)

    rng = np.random.default_rng(0)
    signs = rng.choice([-1.0, 1.0], size=r).astype(np.float32)
    rot = _fwht(truth * signs)                       # orthogonal, so invertible exactly

    print(f"       kurtosis: raw {_kurt(truth):.2f}  rotated {_kurt(rot):.2f} "
          f"(3.0 = Gaussian)")
    for bits in BIT_WIDTHS:
        q = QuantLatent(B, T, r, bits, gs)
        q.write(0, mx.array(truth))
        plain = cos(q.read(T), truth)

        q2 = QuantLatent(B, T, r, bits, gs)
        q2.write(0, mx.array(rot))
        back = _fwht(np.asarray(q2.read(T), dtype=np.float32)) * signs
        rotated = cos(back, truth)

        verdict = "helps" if rotated > plain + 1e-4 else "no gain"
        print(f"       {bits}-bit   plain cos {plain:.7f}   rotated cos {rotated:.7f}"
              f"   -> {verdict}")


def _kurt(a):
    x = np.asarray(a, dtype=np.float64).ravel()
    x = x - x.mean()
    return float((x ** 4).mean() / max((x ** 2).mean() ** 2, 1e-30))


# -------------------------------------------------------------------------- main

def main():
    cfg = MLACfg()
    w = random_weights(cfg)
    wmx = as_mx(w)
    rng = np.random.default_rng(7)
    x_np = (rng.standard_normal((1, 48, cfg.hidden_size)) * 0.5).astype(np.float32)
    x_mx = mx.array(x_np)

    print("MLA quantisation gates  --  latent caching is already 53.3x and lossless;\n"
          "everything below is the additional, LOSSY tier and is off by default.")
    gate_Q0(cfg, wmx, x_mx)
    gate_Q1(cfg, wmx, x_mx)
    gate_Q2(cfg, w, wmx, x_np, x_mx)
    gate_Q4()
    gate_Q5(cfg, wmx, x_mx)
    gate_Q6(cfg, wmx, x_mx)
    gate_Q3()

    print(f"\n{'ALL GATES PASS' if FAILS == 0 else f'{FAILS} GATE(S) FAILED'}")
    return FAILS


if __name__ == "__main__":
    sys.exit(main())
