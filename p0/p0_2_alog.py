"""P0-2: is K3's A_log [128] per-head (fla: [HV]=96) or per-K-channel (head_dim=128)?

fla's contract (fla/ops/kda/gate.py, fla/layers/kda.py:174-179):
    A_log   shape [HV]        (num_v_heads)
    dt_bias shape [HV * K]
    g = -exp(A_log).view(HV,1) * softplus(g_raw + dt_bias.view(HV,K))          (default)
    g = lower_bound * sigmoid(exp(A_log).view(HV,1) * (g_raw + dt_bias.view(HV,K)))  (safe_gate)

K3 shapes force HV=96 (b_proj [96,7168], o_norm [128] => head_v_dim=128, v_proj 12288=96*128),
so fla wants A_log [96]. The checkpoint ships [128]. Test the alternative empirically.
"""
import json, struct, numpy as np

CKPT = "/Volumes/LLMModels/Kimi-K3/"
IDX = json.load(open(CKPT + "model.safetensors.index.json"))["weight_map"]


def load(name):
    f = IDX[name]
    with open(CKPT + f, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
        base = 8 + hlen
        meta = hdr[name]
        s, e = meta["data_offsets"]
        fh.seek(base + s)
        raw = fh.read(e - s)
    dt = {"F32": np.float32, "BF16": np.uint16, "U8": np.uint8, "F16": np.float16}[meta["dtype"]]
    a = np.frombuffer(raw, dtype=dt).reshape(meta["shape"])
    if meta["dtype"] == "BF16":
        a = (a.astype(np.uint32) << 16).view(np.float32)
    return a


def structure_score(x, name):
    """Lag-1 autocorrelation along the flattened axis: high => smooth/structured."""
    d = x.astype(np.float64).ravel()
    d = d - d.mean()
    return float((d[:-1] * d[1:]).sum() / (d * d).sum())


for L in (0, 1, 5, 40):
    A = load(f"language_model.model.layers.{L}.self_attn.A_log")
    dt_bias = load(f"language_model.model.layers.{L}.self_attn.dt_bias")
    print(f"\n===== layer {L}  A_log{list(A.shape)}  dt_bias{list(dt_bias.shape)}")
    print(f"  A_log   min {A.min():+.4f}  max {A.max():+.4f}  mean {A.mean():+.4f}  "
          f"lag1-autocorr {structure_score(A,'A'):+.3f}")
    print(f"  exp(A_log) range [{np.exp(A).min():.3f}, {np.exp(A).max():.3f}]  "
          f"(fla init log(U(1,16)) => exp in [1,16])")

    # If dt_bias's true layout is (H=96, K=128), rows are heads: within-row stats should be
    # more self-similar than under the transposed (128, 96) reading.
    for shape in ((96, 128), (128, 96)):
        m = dt_bias.reshape(shape).astype(np.float64)
        rowvar = m.var(axis=1).mean()
        colvar = m.var(axis=0).mean()
        print(f"  dt_bias.reshape{shape}: mean within-row var {rowvar:9.5f} | "
              f"mean within-col var {colvar:9.5f}  ratio {rowvar/colvar:6.3f}")

    # Decisive: does A_log's length-128 axis align with dt_bias's length-128 axis?
    # Under (96,128) the column index is the K channel; correlate A_log against per-column stats.
    m = dt_bias.reshape(96, 128).astype(np.float64)
    colmean = m.mean(axis=0)
    rowmean = m.mean(axis=1)
    c128 = np.corrcoef(A.astype(np.float64), colmean)[0, 1]
    print(f"  corr(A_log[128], dt_bias(96,128).mean over heads -> per-K-channel[128]) = {c128:+.4f}")
    mt = dt_bias.reshape(128, 96).astype(np.float64)
    c128b = np.corrcoef(A.astype(np.float64), mt.mean(axis=1))[0, 1]
    print(f"  corr(A_log[128], dt_bias(128,96).mean over axis1        -> [128]) = {c128b:+.4f}")

# cross-layer: is A_log[i] stable across layers (per-channel) or layer-idiosyncratic (per-head)?
As = np.stack([load(f"language_model.model.layers.{L}.self_attn.A_log") for L in (1, 2, 5, 6, 9, 10)])
print("\n===== cross-layer A_log correlation matrix (layers 1,2,5,6,9,10)")
print(np.round(np.corrcoef(As.astype(np.float64)), 3))
