"""Launch the OpenAI-compatible server.

    python -m air_llm.serve --model /path/to/Kimi-K3 --cache ~/k3_dense \
        --manifest manifest.jsonl.gz --resident-gb 32 --port 8000

Then:

    export OPENAI_BASE_URL=http://localhost:8000/v1
    export OPENAI_API_KEY=sk-local
    python -c "
    from openai import OpenAI
    c = OpenAI()
    print(c.chat.completions.create(model='chat',
          messages=[{'role':'user','content':'The capital of France is'}],
          max_tokens=2).choices[0].message.content)"

The model is built once at startup and stays resident, so the pinned weight cache is
populated by the first request and warm for every one after it -- unlike the CLI, which
re-populates per invocation.

Use `--mock` to run the full HTTP surface against a trivial stub model. That exercises
routing, auth, SSE framing, cancellation and error shapes in seconds instead of minutes,
which is what the conformance tests use.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def build_engine(a):
    import mlx.core as mx
    from .models.kimi_k3.loader import TensorSource
    from .models.kimi_k3.tokenizer import load as load_tok
    from .server.engine import Engine

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from k3ref.layers import K3Config

    cfg = K3Config.load(Path(a.model) / "config.json")
    if a.layers:
        cfg.num_hidden_layers = a.layers
    src = TensorSource(manifest=Path(a.manifest), ckpt=Path(a.model),
                       cache=Path(a.cache) if a.cache else None,
                       resident_gb=a.resident_gb,
                       weight_dtype=mx.bfloat16 if a.mode == "runtime" else mx.float32)
    tok = load_tok(a.model)
    return Engine(cfg, src, tok, max_memory_gb=a.max_memory_gb,
                  queue_size=a.queue_size, max_output_tokens=a.max_output_tokens,
                  max_input_tokens=a.max_input_tokens)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="air_llm.serve")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("manifest.jsonl.gz"))
    ap.add_argument("--cache", type=Path, default=None)
    ap.add_argument("--mode", default="runtime", choices=["validation", "runtime"])
    ap.add_argument("--resident-gb", type=float, default=0.0)
    ap.add_argument("--max-memory-gb", type=float, default=48.0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--queue-size", type=int, default=2,
                    help="requests admitted beyond the running one before 429")
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--max-input-tokens", type=int, default=8192)
    ap.add_argument("--max-n", type=int, default=1)
    ap.add_argument("--default-max-tokens", type=int, default=16)
    ap.add_argument("--mock", action="store_true",
                    help="serve a stub model: exercises the HTTP surface without weights")
    a = ap.parse_args(argv)

    logging.basicConfig(
        level=os.environ.get("AIR_LLM_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    import uvicorn
    from .server.app import create_app

    if a.mock:
        from .server.mock import MockEngine
        engine = MockEngine()
    else:
        if not a.model:
            raise SystemExit("--model is required (or use --mock)")
        engine = build_engine(a)

    config = {"max_n": a.max_n, "default_max_tokens": a.default_max_tokens,
              "mode": a.mode, "resident_gb": a.resident_gb, "mock": a.mock}
    app = create_app(engine, config)

    @app.on_event("startup")
    async def _startup():
        await engine.warmup()

    if not os.environ.get("AIR_LLM_API_KEYS") and \
            not os.environ.get("AIR_LLM_API_KEYS_FILE") and \
            os.environ.get("AIR_LLM_ALLOW_ANONYMOUS") != "1":
        raise SystemExit(
            "No API keys configured. Set AIR_LLM_API_KEYS=sk-...,sk-... or "
            "AIR_LLM_API_KEYS_FILE=keys.json, or set AIR_LLM_ALLOW_ANONYMOUS=1 to "
            "explicitly disable authentication for local development.")

    uvicorn.run(app, host=a.host, port=a.port, log_level="info",
                timeout_keep_alive=75)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
