# Release checklist

This lists the gates for making the repository public and for any later tagged release.
Each gate says what to run or decide and what counts as passing. Tick a gate only with
the evidence in hand. Record where that evidence lives next to the box.

## 1. Checkpoint-free checks (any Apple Silicon Mac)

- [ ] Fresh clone into an empty directory, then the README commands verbatim, on
      Python 3.11 **and** 3.12:
      `python -m pip install -c constraints-test.txt -e '.[dev]'` then
      `python tests/run_all.py` → **8 passed, 10 skipped, 0 failed**.
- [ ] `python tools/summarize_evidence.py` reproduces every figure in
      [evidence.md](evidence.md).
- [ ] Every relative link in the Markdown files resolves.
- [ ] `python -m pip wheel --no-deps -w dist .` builds, and the wheel contains
      `LICENSE`, `NOTICE` and `third_party/LICENSE.*`.
- [ ] CI is green on the commit being released.

## 2. Full-checkpoint validation (the benchmark machine)

The release-preparation changes altered validation-mode execution, nonfinite handling,
the logit checksum and MXFP4 scale checks. The historical results predate them, so they
must be re-run on the release commit before they are quoted for it. Record the upstream
checkpoint revision (`K3_REVISION`) for every run.

- [ ] `tools/k3_manifest.py verify` passes on the checkpoint.
- [ ] `K3_CKPT=... python tests/run_all.py --include-slow` → 0 failed. Record the
      worst-layer cosine, router matches (expect 92/92) and greedy token from
      `test_full_parity`.
- [ ] `python tests/test_numerical_modes.py` passes.
- [ ] Full-depth smoke test from [INSTALL.md](../INSTALL.md#first-full-depth-run)
      (`--check-finite --validation-log smoke.jsonl`): 93 layers, no nonfinite values,
      coherent text.
- [ ] Either add the new results to [evidence.md](evidence.md) with their logs, or leave
      the README's statement that the quoted numbers are historical.

## 3. Human sign-off: provenance and licensing

These are decisions, not tests. [licensing.md](licensing.md) has the details.

- [ ] Upstream revisions and hashes recorded for `third_party/kda_naive.py`
      (flash-linear-attention), Moonshot's `modeling_kimi_linear.py` (basis of `k3ref/`
      and `config.py`) and `tokenization_kimi.py` (the `PAT_STR` in `tokenizer.py`).
- [ ] Decision recorded on whether the Kimi K3 License permits publishing `k3ref/`,
      `config.py` and the tokenizer pattern here, including its commercial and branding
      conditions.
- [ ] `p0/run_qwen3_mlx.py` (AirLLM-derived) and `p0/p0_4_kda_layer.py` (Moonshot/FLA
      operations) reviewed: notices added, or the files removed.
- [ ] README credit links to the canonical URL of Gavin Li's Kimi-K3 article.
- [ ] README, NOTICE, `pyproject.toml` and the wheel all describe the distribution as
      mixed-license, never as uniformly MIT.

## 4. Repository hygiene before the visibility change

- [ ] History scan: no secrets, keys, tokens, private email addresses or personal data in
      any reachable commit (`git log --all -p`). A scanner such as `gitleaks` is preferred.
- [ ] No large or unexpected blobs in history
      (`git rev-list --objects --all | git cat-file --batch-check`).
- [ ] Machine-specific paths remain only in files marked historical and in raw logs.
- [ ] GitHub settings: description and topics set; private vulnerability reporting
      enabled (for [SECURITY.md](../SECURITY.md)); decide whether issues/discussions are on;
      branch protection on `main` requiring CI.

## 5. Going public

- [ ] Push `main`; wait for CI to pass on that exact commit.
- [ ] Change the visibility to public (owner action).
- [ ] From a signed-out browser: README renders, the CI badge (if any) and links work,
      and the license is detected.
- [ ] Optional: tag `v0.1.0` with release notes that repeat the claim scope from the
      README (historical measurements, one machine, text-only, mixed licensing).
