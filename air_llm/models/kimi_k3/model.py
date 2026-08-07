"""M8: full Kimi-K3 model in MLX, with per-layer streaming.

Execution schedule (spec section 7). Per generated token:

    embedding -> layer 0 .. layer 92 -> output attn-residual -> final norm -> lm_head

and per layer: load non-expert tensors, restore state, attention, attention-residual,
router, stream selected experts serially, shared experts, residual, save state, RELEASE.

Only model state (KDA recurrent + conv, MLA caches, block_residual), the hidden state,
and small globals stay resident between layers. Nothing keeps a layer alive after use.

Memory notes that drive the design:
  * embed_tokens is [163840, 7168] = 2.35 GiB, but a token needs ONE row (14 KiB).
    `TensorSource.rows` seeks to just those rows.
  * lm_head is the same size and genuinely needs every row, so it is streamed in row
    chunks with a running argmax rather than materialised whole.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np

from .layer import LayerDiag, empty_block_residual
from .loader import build_layer
from .ops import apply_attn_res, rmsnorm
from .recurrent_state import KimiK3State

EMBED = "language_model.model.embed_tokens.weight"
LM_HEAD = "language_model.lm_head.weight"
FINAL_NORM = "language_model.model.norm.weight"
OUT_RES_NORM = "language_model.model.output_attn_res_norm.weight"
OUT_RES_PROJ = "language_model.model.output_attn_res_proj.weight"


def _peak_memory() -> int:
    for fn in ("get_peak_memory", "get_active_memory"):
        f = getattr(mx, fn, None)
        if f is not None:
            try:
                return int(f())
            except Exception:
                pass
    return 0


def _reset_peak() -> None:
    f = getattr(mx, "reset_peak_memory", None)
    if f is not None:
        try:
            f()
        except Exception:
            pass


@dataclass
class TokenRecord:
    """Spec section 14: one structured record per generated token."""
    token_index: int
    token_id: int
    elapsed_seconds: float
    bytes_read: int
    expert_bytes: int
    peak_unified_memory: int
    layers_completed: int
    experts_loaded: int
    cache_bytes: int
    logit_checksum: str
    nonfinite: int = 0
    nonfinite_where: str = ""
    layer_diags: list = field(default_factory=list)

    def as_json(self) -> str:
        d = dict(self.__dict__)
        d.pop("layer_diags")
        return json.dumps(d)


class KimiK3Model:
    def __init__(self, cfg, src, max_memory_gb: float | None = None,
                 diagnostic: bool = False, lm_head_chunk: int = 16384,
                 check_finite: bool = False):
        self.cfg, self.src = cfg, src
        self.max_memory = int(max_memory_gb * 2 ** 30) if max_memory_gb else None
        self.diagnostic = diagnostic
        # Spec section 16 reliability: no NaN or infinity in hidden states, recurrent
        # states, routing weights, or logits. Each check forces a device sync, so it is
        # opt-in rather than always on.
        self.check_finite = check_finite
        self.nonfinite = []
        self.lm_head_chunk = lm_head_chunk
        self._globals: dict[str, mx.array] = {}

    # ------------------------------------------------------------------ globals
    def _g(self, name: str) -> mx.array:
        if name not in self._globals:
            self._globals[name] = self.src.mx(name)
        return self._globals[name]

    def embed(self, token_ids: list[int]) -> mx.array:
        """One row per token -- never the whole 2.35 GiB table."""
        rows = self.src.rows(EMBED, token_ids)
        return mx.array(rows).reshape(1, len(token_ids), self.cfg.hidden_size)

    def logits(self, h: mx.array) -> tuple[int, float, int]:
        """Streamed argmax over the vocabulary. Returns (argmax_id, max_logit, bytes)."""
        v = h.reshape(-1)[None, :]                     # last position only, [1, H]
        best_id, best_val, nbytes = -1, -np.inf, 0
        rec = self.src.recs[LM_HEAD]
        V = rec["shape"][0]
        for start in range(0, V, self.lm_head_chunk):
            n = min(self.lm_head_chunk, V - start)
            w = mx.array(self.src.row_block(LM_HEAD, start, n))
            nbytes += n * rec["shape"][1] * 2
            part = (v @ w.T).reshape(-1)
            mx.eval(part)
            i = int(mx.argmax(part).item())
            val = float(part[i].item())
            if val > best_val:
                best_val, best_id = val, start + i
            del w, part
        return best_id, best_val, nbytes

    # ------------------------------------------------------------------ forward
    def forward_token(self, hidden: mx.array, state: KimiK3State,
                      record: TokenRecord | None = None) -> mx.array:
        """One full pass over all 93 layers, streaming each and releasing it."""
        c = self.cfg
        B, T, H = hidden.shape

        # block_residual is INTRA-FORWARD, not persistent state. The shipped model
        # allocates it fresh inside every forward() call (modeling_kimi_linear.py:1188)
        # sized to the current batch*seqlen; it accumulates across LAYERS and is then
        # discarded. Carrying it between tokens is wrong twice over: the row count no
        # longer matches after prefill (N=prompt_len) gives way to decode (N=1), and the
        # checkpoints would come from the wrong token's residual stream.
        #
        # (Spec section 12 lists `attention_residuals` as part of KimiK3State. That does
        # not match the architecture -- there is nothing to carry.)
        br = empty_block_residual(B * T, H)

        for L in range(c.num_hidden_layers):
            diag = LayerDiag() if (self.diagnostic or self.check_finite) else None
            layer = build_layer(c, L, self.src)

            ls = layer.__class__.__mro__  # noqa: F841  (kept for readability of stack)
            from .attention_mla import MLACache
            from .layer import LayerState
            cached = state.mla_caches.get(L)
            st = LayerState(
                kda_state=state.kda_states.get(L),
                kda_conv=state.kda_conv.get(L),
                # KimiK3State stores MLA caches as plain (key, value) tuples because that
                # is what the serialiser writes; rebuild the dataclass on the way in.
                mla_cache=MLACache(*cached) if cached is not None else None,
            )
            hidden, br, st = layer(hidden, br, state=st, diag=diag)

            if c.is_kda(L):
                state.kda_states[L], state.kda_conv[L] = st.kda_state, st.kda_conv
            else:
                state.mla_caches[L] = (st.mla_cache.key, st.mla_cache.value)

            mx.eval(hidden, br)          # force completion before the layer is dropped
            if self.check_finite:
                self._assert_finite(hidden, f"hidden@L{L}")
                if c.is_kda(L):
                    self._assert_finite(st.kda_state, f"kda_state@L{L}")
                    for i, cv in enumerate(st.kda_conv or ()):
                        self._assert_finite(cv, f"kda_conv{i}@L{L}")
                elif st.mla_cache is not None:
                    self._assert_finite(st.mla_cache.key, f"mla_key@L{L}")
                    self._assert_finite(st.mla_cache.value, f"mla_value@L{L}")
                if diag is not None and diag.router_weights is not None:
                    self._assert_finite(diag.router_weights, f"router_w@L{L}")
            del layer                    # spec section 6: no retained references
            if record is not None:
                record.layers_completed = L + 1
                if diag is not None and diag.router_indices is not None:
                    record.experts_loaded += int(
                        len(set(np.array(diag.router_indices, copy=False).ravel().tolist())))
                    if self.diagnostic:
                        record.layer_diags.append({
                            "layer": L,
                            "attn_checksum": diag.attn_checksum,
                            "moe_checksum": diag.moe_checksum,
                            "out_checksum": diag.out_checksum,
                            "experts": sorted(set(
                                np.array(diag.router_indices, copy=False).ravel().tolist())),
                        })
            self._check_memory(L)

        # model-level attention residual, then final norm
        hidden = apply_attn_res(hidden.reshape(-1, H), br,
                                self._g(OUT_RES_PROJ), self._g(OUT_RES_NORM),
                                c.rms_norm_eps).reshape(B, T, H)
        return rmsnorm(hidden, self._g(FINAL_NORM), c.rms_norm_eps)

    def _assert_finite(self, a, where: str) -> None:
        if a is None:
            return
        if not bool(mx.all(mx.isfinite(a.astype(mx.float32)))):
            self.nonfinite.append(where)

    def _check_memory(self, layer: int) -> None:
        if self.max_memory is None:
            return
        peak = _peak_memory()
        if peak > self.max_memory:
            raise MemoryError(
                f"peak unified memory {peak/2**30:.1f} GiB exceeded the configured "
                f"ceiling {self.max_memory/2**30:.1f} GiB at layer {layer}")

    # --------------------------------------------------------------- generation
    def generate(self, prompt_ids: list[int], max_new_tokens: int = 16,
                 log_path: str | Path | None = None, on_token=None) -> list[int]:
        """Greedy decode. Prefill runs the prompt, then one token at a time."""
        state = KimiK3State()
        out_ids: list[int] = []
        log = open(log_path, "w") if log_path else None

        try:
            ids = list(prompt_ids)
            for i in range(max_new_tokens):
                _reset_peak()
                t0 = time.time()
                self.src.reset_counters()

                step_ids = ids if i == 0 else [out_ids[-1]]
                h = self.embed(step_ids)
                h = self.forward_token(h, state, None)
                rec = TokenRecord(
                    token_index=i, token_id=-1, elapsed_seconds=0.0, bytes_read=0,
                    expert_bytes=0, peak_unified_memory=0,
                    layers_completed=self.cfg.num_hidden_layers,
                    experts_loaded=0, cache_bytes=0, logit_checksum="")

                self.nonfinite = []
                if self.check_finite:
                    self._assert_finite(h, "final_hidden")
                tok, val, head_bytes = self.logits(h[:, -1, :])
                if self.check_finite and not (val == val and abs(val) != float("inf")):
                    self.nonfinite.append("logit")
                state.advance(len(step_ids))
                out_ids.append(tok)

                mem = state.memory_report()
                rec.token_id = tok
                rec.elapsed_seconds = round(time.time() - t0, 3)
                # NOT + head_bytes: row_block already counted the lm_head stream into
                # src.bytes_read. Adding it again double-counts 2.19 GiB per token.
                rec.bytes_read = self.src.bytes_read
                rec.expert_bytes = self.src.expert_bytes
                rec.peak_unified_memory = _peak_memory()
                rec.cache_bytes = mem["total"]
                rec.logit_checksum = hashlib.sha256(
                    f"{tok}:{val:.6f}".encode()).hexdigest()[:16]
                rec.experts_loaded = self.src.experts_loaded
                rec.nonfinite = len(self.nonfinite)
                rec.nonfinite_where = ",".join(self.nonfinite[:5])

                if log:
                    log.write(rec.as_json() + "\n")
                    log.flush()
                if on_token:
                    on_token(rec)
            return out_ids
        finally:
            if log:
                log.close()
