# Contributing

Issues and pull requests are welcome. This is a one-person research project, so reviews
may be slow; small, focused changes with evidence are the easiest to take.

## Set up and run the tests

You need an Apple Silicon Mac (MLX has no x86 build), macOS 15+, and Python 3.11 or 3.12.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-test.txt -e '.[dev]'
python tests/run_all.py
```

A fresh clone should report **8 passed, 10 skipped, 0 failed**. This is what CI runs on
every push and pull request. The skipped suites compare against the real ~1.42 TiB
checkpoint. They run when `K3_CKPT` points at one:

```sh
K3_CKPT=/path/to/Kimi-K3 K3_MANIFEST=/path/to/manifest.jsonl.gz python tests/run_all.py
K3_CKPT=... python tests/run_all.py --include-slow     # + full 93-layer parity, ~10 min on an M4 Pro
```

| variable | what |
|---|---|
| `K3_CKPT` | checkpoint directory |
| `K3_MANIFEST` | tensor index built by `tools/k3_manifest.py build` |
| `K3_CACHE` | optional staged dense weights from `tools/k3_cache.py` |

A skipped suite is not a passing one. If your change touches model math, loading, or
kernels, say in the pull request whether you ran the checkpoint suites, and paste the
summary if you did.

Each suite is a standalone script that exits non-zero on failure, so you can run one
directly: `python tests/test_mla_cache.py`.

## What a good change looks like

- **Numbers need provenance.** A performance or accuracy claim in code, docs, or a PR
  description should say what was measured, on what hardware, with which flags, and
  where the raw log is. Follow the protocol at the end of
  [docs/evidence.md](docs/evidence.md). Re-run `python tools/summarize_evidence.py` if
  you touch the committed logs or the ledger.
- **Refuse rather than approximate.** The API server returns a structured error for
  anything it cannot do exactly (tools, JSON mode, `n>1`, unknown fields). Keep it that
  way; don't add a parameter that is silently ignored.
- **Keep checkpoint-free tests checkpoint-free.** New behaviour should come with a test
  that runs in CI: the synthetic model (`tools/make_synthetic_k3.py`) and the mock server
  engine exist for this.
- **If you change the HTTP surface**, regenerate `docs/openapi.json` — the conformance
  test fails when it drifts from the served spec:

  ```sh
  python -c "import json; from air_llm.server.app import create_app; \
  from air_llm.server.mock import MockEngine; \
  open('docs/openapi.json','w').write(json.dumps(create_app(MockEngine(),{}).openapi(), separators=(',',':')))"
  ```

## What not to commit

Model weights, manifests (they embed local paths), staged caches, `.env` files, keys,
and new `runs/` output. `.gitignore` covers these; don't force-add them. Historical
records (`PROGRESS.md`, `SESSION.md`, `docs/kv-cache/`, the tracked `runs/`) are kept as
written. Correct them with a note or in `docs/evidence.md` instead of rewriting them.

## Licensing of contributions

This repository mixes licenses; read [NOTICE](NOTICE) and
[docs/licensing.md](docs/licensing.md). Unless you say otherwise, your contribution to
first-party code is licensed under the MIT terms in [LICENSE](LICENSE).

Do not paste code from Moonshot's checkpoint (`modeling_kimi_linear.py`,
`tokenization_kimi.py`, …), flash-linear-attention, AirLLM, or any other project
without saying so in the pull request. Copied or adapted code needs its source, revision,
and license recorded in NOTICE and `docs/licensing.md` before it can be merged.
