"""Sampling and logprobs on top of the streamed lm_head.

`KimiK3Model.logits()` streams the 2.19 GiB lm_head in row chunks and keeps only a
running argmax, which is all greedy decoding needs. The OpenAI surface additionally
needs `temperature`, `top_p`, `seed`, `n`, and `logprobs`/`top_logprobs`, all of which
need the full distribution.

That is cheap: chunking exists to bound the *weight* matrix, not the logits. The full
logit vector is 163,840 fp32 = 640 KiB, so it is collected in one piece and everything
else is ordinary array work.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import numpy as np


@dataclass
class TokenChoice:
    token_id: int
    logprob: float
    top: list[tuple[int, float]]      # [(token_id, logprob), ...] length = top_logprobs


def full_logits(model, h: mx.array) -> tuple[mx.array, int]:
    """[V] fp32 logits for the last position, plus bytes read.

    Same streaming read pattern as `KimiK3Model.logits` -- the lm_head is never
    materialised whole -- but the partial logit vectors are concatenated.
    """
    from .model import LM_HEAD

    v = h.reshape(-1)[None, :]
    rec = model.src.recs[LM_HEAD]
    V = rec["shape"][0]
    parts, nbytes = [], 0
    for start in range(0, V, model.lm_head_chunk):
        n = min(model.lm_head_chunk, V - start)
        w = mx.array(model.src.row_block(LM_HEAD, start, n))
        nbytes += n * rec["shape"][1] * 2
        part = (v @ w.T).reshape(-1).astype(mx.float32)
        mx.eval(part)
        parts.append(part)
        del w
    return mx.concatenate(parts, axis=0), nbytes


def _logsoftmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return (x - m) - math.log(float(e.sum()))


def choose(logits: mx.array, *, temperature: float = 0.0, top_p: float = 1.0,
           seed: int | None = None, n_logprobs: int = 0,
           logit_bias: dict[int, float] | None = None) -> TokenChoice:
    """Pick one token and report its logprob plus the top-k alternatives.

    temperature == 0 is exact greedy (argmax), not a near-zero-temperature sample --
    that keeps `temperature=0` bit-reproducible, which the determinism tests rely on.

    Logprobs are always reported from the UNMODIFIED distribution, i.e. before
    temperature and top_p. That matches OpenAI: `logprobs` describes the model's belief,
    not the shape of the sampler you happened to use.
    """
    x = np.array(logits, copy=False).astype(np.float64)
    if logit_bias:
        for tid, b in logit_bias.items():
            if 0 <= tid < x.shape[0]:
                x[tid] += b

    lp_true = _logsoftmax(x)

    if temperature <= 0.0:
        tok = int(np.argmax(x))
    else:
        z = x / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            order = np.argsort(-p)
            csum = np.cumsum(p[order])
            keep = int(np.searchsorted(csum, top_p) + 1)
            mask = np.zeros_like(p, dtype=bool)
            mask[order[:keep]] = True
            p = np.where(mask, p, 0.0)
            s = p.sum()
            p = p / s if s > 0 else np.full_like(p, 1.0 / p.size)
        rng = np.random.default_rng(seed)
        tok = int(rng.choice(p.shape[0], p=p))

    top: list[tuple[int, float]] = []
    if n_logprobs > 0:
        k = min(int(n_logprobs), x.shape[0])
        idx = np.argpartition(-lp_true, k - 1)[:k]
        idx = idx[np.argsort(-lp_true[idx])]
        top = [(int(i), float(lp_true[i])) for i in idx]

    return TokenChoice(token_id=tok, logprob=float(lp_true[tok]), top=top)
