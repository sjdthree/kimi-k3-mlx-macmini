"""MLA KV-cache gates: latent caching must be LOSSLESS, small, and O(T).

Why this file exists
--------------------
`attention_mla.py` caches the EXPANDED per-head key/value:

    key   [B, 96, T, 192]      value [B, 96, T, 128]      = 30,720 floats/token/layer

but the tensors those are built from are only

    c_kv  [B, T, 512]          k_rot [B, T, 64]           =    576 floats/token/layer

The expansion is a fixed linear map (`kv_b`), so caching the latent and folding `kv_b`
into the query/output projections is ALGEBRAICALLY IDENTICAL -- 53.3x less memory for
exactly the same numbers. That is the claim these gates exist to hold to account.

Three properties have to hold simultaneously, and it is easy to get one by breaking
another:

    NUMERICAL   the absorbed form must reproduce the expanded form to fp32 noise
    MEMORY      bytes/token must actually drop by >=50x, not just look smaller
    GROWTH      appending token T+1 must not copy the whole cache (today it does)

Gates
-----
    N1  full-sequence parity vs an independent fp64 numpy oracle
    N2  incremental decode == single-shot full sequence   (the ring-buffer gate)
    N3  new latent path == frozen expanded path           (the losslessness gate)
    N4  MQA rope slice is not materialised 96x
    M1  bytes/token <= 576 * itemsize * n_mla_layers
    M2  appending N tokens causes O(N/step) reallocations, not O(N)
    Q*  quantisation quality gates -- see test_mla_quant.py (only meaningful once
        N1-N3 and M1-M2 are green)

Run:
    source .venv/bin/activate && python tests/test_mla_cache.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FAILS = 0
# Spec section 16, fp32 validation mode. The absorbed form reassociates matmuls, so the
# difference from the expanded form is pure fp32 rounding -- it must stay at this level.
ATOL = 1e-5


def check(label, ok, detail=""):
    global FAILS
    if not ok:
        FAILS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    return ok


def cmp(a, b, label, atol=ATOL):
    """Compare two arrays (mx or np) in fp64. Reports max abs, relative, and cosine."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ad = float(np.abs(a - b).max())
    scale = max(float(np.abs(b).max()), 1e-30)
    af, bf = a.ravel(), b.ravel()
    cos = float(af @ bf / max(np.linalg.norm(af) * np.linalg.norm(bf), 1e-30))
    check(label, ad <= atol, f"max|d| {ad:.3e}  rel {ad/scale:.2e}  cos {cos:.10f}")
    return ad, cos


# --------------------------------------------------------------------------- config

@dataclass
class MLACfg:
    """Only the fields MLA reads. Head dims are kept at the REAL K3 values because the
    slice boundaries (512 | 64) and (128 | 128) are exactly where indexing bugs live;
    only head COUNT and hidden size are shrunk so the test runs in a second."""
    hidden_size: int = 1024
    num_attention_heads: int = 8
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512          # real
    qk_nope_head_dim: int = 128      # real
    qk_rope_head_dim: int = 64       # real
    v_head_dim: int = 128            # real
    rms_norm_eps: float = 1e-5
    mla_use_output_gate: bool = True

    @property
    def q_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


# Real Kimi-K3 numbers, used by the memory-accounting gate only (no allocation).
REAL_MLA_LAYERS = 24          # layer 0 + {3,7,...,91} once full_attn_layers is 0-indexed
REAL_NH = 96
REAL_QHD = 192
REAL_VHD = 128
REAL_LATENT = 512 + 64


def random_weights(cfg: MLACfg, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    H, nh = cfg.hidden_size, cfg.num_attention_heads
    qhd, vhd = cfg.q_head_dim, cfg.v_head_dim
    nope, rope, klr = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.kv_lora_rank

    def w(*shape, s=0.02):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    return {
        "q_a_proj.weight": w(cfg.q_lora_rank, H),
        "q_a_layernorm.weight": (1.0 + rng.standard_normal(cfg.q_lora_rank) * 0.05).astype(np.float32),
        "q_b_proj.weight": w(nh * qhd, cfg.q_lora_rank),
        "kv_a_proj_with_mqa.weight": w(klr + rope, H),
        "kv_a_layernorm.weight": (1.0 + rng.standard_normal(klr) * 0.05).astype(np.float32),
        "kv_b_proj.weight": w(nh * (nope + vhd), klr),
        "g_proj.weight": w(H, H),
        "o_proj.weight": w(H, nh * vhd),
    }


def as_mx(w: dict) -> dict:
    return {k: mx.array(v) for k, v in w.items()}


# ----------------------------------------------------------------- the fp64 oracle

def _rmsnorm64(x, w, eps):
    return w * (x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)))


def oracle_mla(cfg: MLACfg, w: dict, x: np.ndarray) -> np.ndarray:
    """Independent fp64 reimplementation of KimiMLAAttention (NoPE), full sequence.

    Deliberately written from modeling_kimi_linear.py's structure rather than from
    air_llm's code, so that a shared misreading cannot make both agree.
    """
    x = x.astype(np.float64)
    B, T, _ = x.shape
    nh, qhd, vhd = cfg.num_attention_heads, cfg.q_head_dim, cfg.v_head_dim
    nope, rope, klr = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.kv_lora_rank
    eps = cfg.rms_norm_eps
    f = {k: v.astype(np.float64) for k, v in w.items()}

    q = _rmsnorm64(x @ f["q_a_proj.weight"].T, f["q_a_layernorm.weight"], eps) @ f["q_b_proj.weight"].T
    q = q.reshape(B, T, nh, qhd).transpose(0, 2, 1, 3)                    # [B,H,T,qhd]

    ckv = x @ f["kv_a_proj_with_mqa.weight"].T
    k_lat, k_rot = ckv[..., :klr], ckv[..., klr:]
    kv = _rmsnorm64(k_lat, f["kv_a_layernorm.weight"], eps) @ f["kv_b_proj.weight"].T
    kv = kv.reshape(B, T, nh, nope + vhd).transpose(0, 2, 1, 3)
    k_pass, v = kv[..., :nope], kv[..., nope:]

    k_rot_b = np.broadcast_to(k_rot.reshape(B, 1, T, rope), (B, nh, T, rope))
    key = np.concatenate([k_pass, k_rot_b], axis=-1)                      # [B,H,T,qhd]

    scores = q @ key.transpose(0, 1, 3, 2) * (qhd ** -0.5)
    causal = np.where(np.arange(T)[None, :] > np.arange(T)[:, None], -np.inf, 0.0)
    scores = scores + causal
    scores -= scores.max(-1, keepdims=True)
    e = np.exp(scores)
    probs = e / e.sum(-1, keepdims=True)

    o = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * vhd)
    if cfg.mla_use_output_gate:
        o = o * (1.0 / (1.0 + np.exp(-(x @ f["g_proj.weight"].T))))
    return o @ f["o_proj.weight"].T


# ------------------------------------------------- frozen copy of the CURRENT path
# Pinned here so the gates keep a reference to compare against after attention_mla.py
# is rewritten. This is byte-for-byte the logic at attention_mla.py:43-77 on master.

def frozen_expanded_mla(cfg: MLACfg, w: dict, x: mx.array, cache=None):
    from air_llm.models.kimi_k3.ops import causal_mask, rmsnorm
    B, T, _ = x.shape
    nh, qhd, vhd = cfg.num_attention_heads, cfg.q_head_dim, cfg.v_head_dim
    nope, rope = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim
    eps = cfg.rms_norm_eps

    q = rmsnorm(x @ w["q_a_proj.weight"].T, w["q_a_layernorm.weight"], eps) @ w["q_b_proj.weight"].T
    q = q.reshape(B, T, nh, qhd).transpose(0, 2, 1, 3)

    ckv = x @ w["kv_a_proj_with_mqa.weight"].T
    k_lat, k_rot = ckv[..., :cfg.kv_lora_rank], ckv[..., cfg.kv_lora_rank:]
    kv = rmsnorm(k_lat, w["kv_a_layernorm.weight"], eps) @ w["kv_b_proj.weight"].T
    kv = kv.reshape(B, T, nh, nope + vhd).transpose(0, 2, 1, 3)
    k_pass, v = kv[..., :nope], kv[..., nope:]

    k_rot = mx.broadcast_to(k_rot.reshape(B, 1, T, rope), (B, nh, T, rope))
    key = mx.concatenate([k_pass, k_rot], axis=-1)

    if cache is not None:
        key = mx.concatenate([cache[0], key], axis=2)
        v = mx.concatenate([cache[1], v], axis=2)

    scores = (q @ key.transpose(0, 1, 3, 2)) * (qhd ** -0.5)
    scores = scores + causal_mask(T, key.shape[2])
    probs = mx.softmax(scores.astype(mx.float32), axis=-1)
    o = (probs @ v).transpose(0, 2, 1, 3).reshape(B, T, nh * vhd)
    if cfg.mla_use_output_gate:
        o = o * mx.sigmoid(x @ w["g_proj.weight"].T)
    return o @ w["o_proj.weight"].T, (key, v)


# --------------------------------------------------------------- path under test

def _new_impl():
    """Import the implementation under test; None if it does not exist yet."""
    try:
        from air_llm.models.kimi_k3.attention_mla import MLAAttention
        return MLAAttention
    except Exception:
        return None


def run_new(cfg, wmx, x, cache=None):
    MLA = _new_impl()
    attn = MLA(cfg, wmx)
    return attn(x, cache=cache)


# ------------------------------------------------------------------------- gates

def gate_N1(cfg, w, wmx, x_np, x_mx):
    print("\nN1 -- full-sequence parity vs fp64 oracle")
    ref = oracle_mla(cfg, w, x_np)
    got, _ = run_new(cfg, wmx, x_mx)
    cmp(got, ref, "new impl vs oracle (T tokens, one shot)")
    old, _ = frozen_expanded_mla(cfg, wmx, x_mx)
    cmp(old, ref, "frozen expanded vs oracle (sanity: oracle is right)")


def gate_N2(cfg, w, wmx, x_np, x_mx, prefill: int):
    """Prefill `prefill` tokens, then decode the rest one at a time. The concatenated
    result must equal the single-shot full-sequence result. This is the gate that
    catches ring-buffer wraparound, stale-length, and off-by-one masking bugs."""
    T = x_mx.shape[1]
    print(f"\nN2 -- incremental decode == single shot  (prefill {prefill}, then {T - prefill} steps)")

    full, _ = run_new(cfg, wmx, x_mx)

    out, cache = run_new(cfg, wmx, x_mx[:, :prefill])
    pieces = [np.asarray(out, dtype=np.float64)]
    for t in range(prefill, T):
        step, cache = run_new(cfg, wmx, x_mx[:, t:t + 1], cache=cache)
        pieces.append(np.asarray(step, dtype=np.float64))
    inc = np.concatenate(pieces, axis=1)

    cmp(inc, np.asarray(full, dtype=np.float64), "incremental vs one-shot")
    cmp(inc, oracle_mla(cfg, w, x_np), "incremental vs fp64 oracle")


def gate_N3(cfg, wmx, x_mx, prefill: int):
    """The losslessness claim: latent+absorbed must equal expanded, step for step."""
    T = x_mx.shape[1]
    print(f"\nN3 -- latent path == frozen expanded path, every decode step")

    o_new, c_new = run_new(cfg, wmx, x_mx[:, :prefill])
    o_old, c_old = frozen_expanded_mla(cfg, wmx, x_mx[:, :prefill])
    worst = cmp(o_new, o_old, "prefill output")[0]

    for t in range(prefill, T):
        o_new, c_new = run_new(cfg, wmx, x_mx[:, t:t + 1], cache=c_new)
        o_old, c_old = frozen_expanded_mla(cfg, wmx, x_mx[:, t:t + 1], cache=c_old)
        d = float(np.abs(np.asarray(o_new, dtype=np.float64)
                         - np.asarray(o_old, dtype=np.float64)).max())
        worst = max(worst, d)
    check(f"worst deviation over {T - prefill} decode steps", worst <= ATOL,
          f"max|d| {worst:.3e}")


def gate_N4(cfg, wmx, x_mx):
    """The MQA rope slice is shared across all heads. It must be stored ONCE."""
    print("\nN4 -- rope slice stored once, not broadcast to every head")
    _, cache = run_new(cfg, wmx, x_mx)
    nh = cfg.num_attention_heads
    shapes = [tuple(a.shape) for a in _cache_arrays(cache)]
    check("no cached tensor carries a per-head axis of size num_attention_heads",
          all(nh not in s for s in shapes), f"shapes {shapes}")


def _cache_arrays(cache):
    """Every mx.array reachable from a cache object, whatever shape the cache takes."""
    if cache is None:
        return []
    if isinstance(cache, mx.array):
        return [cache]
    if isinstance(cache, (tuple, list)):
        return [a for c in cache for a in _cache_arrays(c)]
    out = []
    for v in getattr(cache, "__dict__", {}).values():
        out += _cache_arrays(v)
    return out


def gate_M1(cfg, wmx, long_x):
    """Marginal resident bytes per token, measured.

    Taken as b(2L) - b(L) over L rather than b(T)/T, because a chunked buffer carries
    slack: dividing total by length would charge the new cache for headroom it has not
    used yet, and would let a per-token allocator hide behind a small T. The marginal is
    the number that decides whether a 32k context fits.
    """
    print("\nM1 -- cache bytes/token (marginal)")
    L = long_x.shape[1] // 2
    bL = sum(a.nbytes for a in _cache_arrays(run_new(cfg, wmx, long_x[:, :L])[1]))
    b2L = sum(a.nbytes for a in _cache_arrays(run_new(cfg, wmx, long_x[:, :2 * L])[1]))
    per_tok = (b2L - bL) / L

    nh, qhd, vhd = cfg.num_attention_heads, cfg.q_head_dim, cfg.v_head_dim
    expanded = (nh * qhd + nh * vhd) * 4
    ceiling = REAL_LATENT * 4                      # fp32 latent, the lossless target
    # The latent size is independent of head COUNT, so this ceiling is the same one the
    # real 96-head model must meet -- that is why the shrunk config still tests the claim.
    check("per-token bytes at or below the fp32 latent size", per_tok <= ceiling + 1e-9,
          f"{per_tok:.0f} B/tok  vs ceiling {ceiling} B/tok  "
          f"(expanded is {expanded} B/tok, {expanded / max(per_tok, 1):.1f}x)")


def gate_M2(cfg, wmx, x_mx, steps: int = 256):
    """Appending must not reallocate-and-copy the whole cache every token.

    Measured structurally: the backing buffer's token capacity must change O(steps/step)
    times, not once per token.
    """
    print(f"\nM2 -- append is amortised O(1), not O(T) copy  ({steps} steps)")
    _, cache = run_new(cfg, wmx, x_mx[:, :1])
    # Total backing-store bytes. Layout-independent on purpose: the old cache is
    # [B,H,T,D] and the new one is [B,T,r], so any per-axis probe would measure a
    # different thing in each. A per-token concat grows this EVERY step; a chunked
    # buffer grows it only when a chunk fills.
    caps = []
    for _ in range(steps):
        _, cache = run_new(cfg, wmx, x_mx[:, :1], cache=cache)
        caps.append(sum(a.nbytes for a in _cache_arrays(cache)))
    grows = sum(1 for i in range(1, len(caps)) if caps[i] != caps[i - 1])
    check("backing store grows in chunks, not per token", grows <= steps // 8 + 2,
          f"{grows} reallocations over {steps} appends "
          f"(per-token concat gives {steps - 1})")


def gate_S1(cfg, wmx, x_mx, tmpdir: Path):
    """Save/restore must be a no-op on the continuation.

    tests/test_state_roundtrip.py exercises the KDA half of KimiK3State but never
    populates mla_caches, so this path has had no coverage. It is exactly the kind of
    place the conv-layout trap already bit once: a cache that round-trips to a plausible
    but wrong buffer diverges silently a few tokens later.
    """
    from air_llm.models.kimi_k3.recurrent_state import KimiK3State
    print("\nS1 -- MLA cache survives save/load")
    T, split = x_mx.shape[1], 24

    _, straight = run_new(cfg, wmx, x_mx[:, :split])
    for t in range(split, T):
        o_ref, straight = run_new(cfg, wmx, x_mx[:, t:t + 1], cache=straight)

    _, cache = run_new(cfg, wmx, x_mx[:, :split])
    st = KimiK3State(token_position=split)
    st.mla_caches[0] = cache
    p = tmpdir / "mla_state.npz"
    st.save(p)
    back = KimiK3State.load(p)
    check("restored length matches", back.mla_caches[0].length == split,
          f"{back.mla_caches[0].length} vs {split}")

    resumed = back.mla_caches[0]
    for t in range(split, T):
        o_got, resumed = run_new(cfg, wmx, x_mx[:, t:t + 1], cache=resumed)
    cmp(o_got, o_ref, f"continuation after reload ({T - split} tokens)", atol=0.0)

    # the buffer must still grow after a reload -- a restored cache is trimmed exactly
    # to `length`, so the very next append hits the growth path
    check("restored cache still appends", resumed.length == T, f"{resumed.length} vs {T}")


def gate_D1(cfg, w, wmx, x_np, x_mx):
    """bf16 latent storage: a further 2x, but no longer exact. Measured, not asserted."""
    print("\nD1 -- storage dtype trade (report only; fp32 is the default)")
    ref = oracle_mla(cfg, w, x_np)
    import copy
    for name, dt in (("fp32 (default)", mx.float32), ("bf16", mx.bfloat16)):
        c2 = copy.copy(cfg)
        c2.mla_cache_dtype = dt
        got, cache = run_new(c2, wmx, x_mx)
        a = np.asarray(got, dtype=np.float64).ravel()
        b = np.asarray(ref, dtype=np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        per_tok = cache.latent.shape[2] * cache.latent.itemsize + \
            cache.rot.shape[2] * cache.rot.itemsize
        print(f"       {name:<16} {per_tok:>5} B/tok/layer   "
              f"max|d| {np.abs(a-b).max():.3e}   cos {cos:.10f}")


def gate_M3():
    """Pure arithmetic on the real model -- no allocation, no checkpoint access."""
    print("\nM3 -- real Kimi-K3 accounting")
    now = REAL_MLA_LAYERS * (REAL_NH * REAL_QHD + REAL_NH * REAL_VHD) * 4
    tgt = REAL_MLA_LAYERS * REAL_LATENT * 4
    check("measured master value is 2,949,120 B/tok", now == 2_949_120, f"{now}")
    check("latent target is >=50x smaller", now / tgt >= 50, f"{now/tgt:.1f}x")
    for ctx in (2_349, 8_192, 32_768, 131_072):
        print(f"       ctx {ctx:>7}: {ctx*now/2**30:8.2f} GiB  ->  {ctx*tgt/2**30:6.3f} GiB")


# ---------------------------------------------------------------------------- main

def main():
    cfg = MLACfg()
    w = random_weights(cfg)
    wmx = as_mx(w)
    rng = np.random.default_rng(7)
    T = 48
    x_np = (rng.standard_normal((1, T, cfg.hidden_size)) * 0.5).astype(np.float32)
    x_mx = mx.array(x_np)

    print(f"MLA cache gates  --  heads {cfg.num_attention_heads}, "
          f"kv_lora {cfg.kv_lora_rank}, nope/rope {cfg.qk_nope_head_dim}/"
          f"{cfg.qk_rope_head_dim}, v {cfg.v_head_dim}, T {T}")

    if _new_impl() is None:
        print("\n!! air_llm.models.kimi_k3.attention_mla.MLAAttention not importable")
        return 1

    gate_M3()
    gate_N1(cfg, w, wmx, x_np, x_mx)
    gate_N2(cfg, w, wmx, x_np, x_mx, prefill=16)
    gate_N3(cfg, wmx, x_mx, prefill=16)
    gate_N4(cfg, wmx, x_mx)
    long_x = mx.array((rng.standard_normal((1, 512, cfg.hidden_size)) * 0.5).astype(np.float32))
    gate_M1(cfg, wmx, long_x)
    gate_M2(cfg, wmx, x_mx)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        gate_S1(cfg, wmx, x_mx, Path(d))
    gate_D1(cfg, w, wmx, x_np, x_mx)

    print(f"\n{'ALL GATES PASS' if FAILS == 0 else f'{FAILS} GATE(S) FAILED'}")
    return FAILS


if __name__ == "__main__":
    sys.exit(main())
