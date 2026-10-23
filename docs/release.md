# Release checklist

This lists the gates for making the repository public and for any later tagged release.
Each gate says what to run or decide and what counts as passing. Tick a gate only with
the evidence in hand. Record where that evidence lives next to the box.

## 1. Checkpoint-free checks (any Apple Silicon Mac)

- [x] Fresh clone into an empty directory, then the README commands verbatim, on
      Python 3.11 **and** 3.12:
      `python -m pip install -c constraints-test.txt -e '.[dev]'` then
      `python tests/run_all.py` → **8 passed, 10 skipped, 0 failed**. 2026-09-21: both
      versions on the audit machine; 3.12 again from a fresh clone of `4c487f5` on the
      benchmark machine; CI runs both on every push.
- [x] `python tools/summarize_evidence.py` reproduces every figure in
      [evidence.md](evidence.md). Checked 2026-09-21; CI also runs it.
- [x] Every relative link in the Markdown files resolves. 2026-09-21: 31 files, none broken.
- [x] From a clean tree (`rm -rf build dist`; a stale `build/` leaks old files into the
      wheel), `python -m pip wheel --no-deps -w dist .` builds, and the wheel contains
      `LICENSE`, `NOTICE`, `licenses/LICENSE.*` and `k3ref/_vendor/kda_naive.py`.
      2026-09-21: all present; metadata `License-Expression: LicenseRef-Project-Mixed`.
- [x] CI is green on the commit being released. 2026-09-23: `852dc4d` (the v0.1.0 tag),
      run 35840448253, which ran it as `3388f30` before the identity rewrite (same tree).

## 2. Full-checkpoint validation (the benchmark machine)

The release-preparation changes altered validation-mode execution, nonfinite handling,
the logit checksum and MXFP4 scale checks. The historical results predate them, so they
must be re-run on the release commit before they are quoted for it. Record the upstream
checkpoint revision (`K3_REVISION`) for every run.

- [x] `tools/k3_manifest.py verify` passes on the checkpoint. Evidence: `runs/release-validation-2026-09-21/`.
- [x] `K3_CKPT=... python tests/run_all.py --include-slow` → 0 failed. Record the
      worst-layer cosine, router matches (expect 92/92) and greedy token from
      `test_full_parity`. 2026-09-21: 18 passed, 0 failed; 0.99999976 (layer 14), 92/92,
      token 17374. Evidence: `runs/release-validation-2026-09-21/`.
- [x] `python tests/test_numerical_modes.py` passes. Evidence: `runs/release-validation-2026-09-21/`.
- [x] Full-depth smoke test from [INSTALL.md](../INSTALL.md#first-full-depth-run)
      (`--check-finite --validation-log smoke.jsonl`): 93 layers, no nonfinite values,
      coherent text. 2026-09-21: 93 layers, 0 nonfinite, tokens `[17374, 20829, 10, 427]`
      ("Paris." and then code-like punctuation), matching the historical runs. The owner accepted this, as a raw five-token prompt with
      no chat template. Evidence: `runs/release-validation-2026-09-21/`.
- [x] Either add the new results to [evidence.md](evidence.md) with their logs, or leave
      the README's statement that the quoted numbers are historical.

## 3. Human sign-off: provenance and licensing

These are decisions, not tests. [licensing.md](licensing.md) has the details.

- [x] Upstream revisions and hashes recorded for `k3ref/_vendor/kda_naive.py`
      (flash-linear-attention), Moonshot's `modeling_kimi_linear.py` (basis of `k3ref/`
      and `config.py`) and `tokenization_kimi.py` (the `PAT_STR` in `tokenizer.py`).
      See [licensing.md](licensing.md#upstream-sources-verified-2026-09-21).
- [x] Decision recorded on whether the Kimi K3 License permits publishing `k3ref/`,
      `config.py` and the tokenizer pattern here, including its commercial and branding
      conditions. 2026-09-21, owner: publish under the Kimi K3 License with its notice,
      and summarise sections 2 and 3 in the README.
- [x] `p0/run_qwen3_mlx.py` (AirLLM-derived) and `archive/p0/p0_4_kda_layer.py` (Moonshot/FLA
      operations) reviewed: notices added, or the files removed. 2026-09-21:
      `run_qwen3_mlx.py` and `validate_mlx.py` removed; `p0_4_kda_layer.py` has a
      provenance header.
- [x] README credit links to the canonical URL of Gavin Li's Kimi-K3 article.
- [x] README, NOTICE, `pyproject.toml` and the wheel all describe the distribution as
      mixed-license, never as uniformly MIT. `CITATION.cff` deliberately has no license
      field.

## 4. Repository hygiene before the visibility change

- [x] History scan: no secrets, keys, tokens, private email addresses or personal data in
      any reachable commit: `gitleaks git --log-opts=--all --redact .` and
      `gitleaks dir --redact .` report no leaks. Reviewed false positives are listed in
      `.gitleaksignore`; add to it only after reading the match. 2026-09-21: gitleaks
      8.30.1 reports no leaks in 63 commits or the tree (rerun 2026-09-23). Commit
      metadata: on 2026-09-23 the owner had history rewritten so that every commit's
      author and committer is the GitHub noreply address; a university address and a
      machine hostname were removed. Commit IDs changed; IDs cited in `runs/` raw logs
      are the pre-rewrite ones. 2026-10-23: two later commits had picked up the machine
      hostname again (no `user.email` was configured) and were rewritten; the `v0.1.0`
      tag and release, which still pointed at pre-rewrite history, were deleted, to be
      recreated on the cleaned `main` in gate 5. Check `git log --all --format='%ae %ce'`
      shows only the noreply address before going public.
- [x] The evaluator answer key for the private-document demo is not in the tree or in
      any reachable commit: `git log --all --name-only --format= | grep -i -e do_not_publish -e ground_truth -e handoff`
      prints nothing, and `demos/private-document-analysis/SHA256SUMS` still verifies
      (`cd demos/private-document-analysis && shasum -a 256 -c SHA256SUMS`).
      (`--format=` keeps commit messages that mention these words out of the match.)
      2026-09-23: no matches in 63 commits; all 5 hashes OK.
- [x] No large or unexpected blobs in history
      (`git rev-list --objects --all | git cat-file --batch-check`). Largest blob 482 KB.
- [x] Machine-specific paths remain only in files marked historical and in raw logs.
- [x] GitHub settings: description and topics set; private vulnerability reporting
      enabled (for [SECURITY.md](../SECURITY.md)); decide whether issues/discussions are on;
      branch protection on `main` requiring CI. 2026-09-21: issues and Discussions on
      (blank issues routed to Discussions), wiki and projects off, ruleset "protect main"
      (no deletion or force-push, both CI checks required, admin bypass). Vulnerability
      reporting needs a public repo: see gate 5.

## 5. Going public

- [ ] Push `main`; wait for CI to pass on that exact commit.
- [ ] Change the visibility to public (owner action).
- [ ] Enable private vulnerability reporting (GitHub offers it only on public repos).
- [ ] Upload [social-preview.png](social-preview.png) under Settings → General → Social
      preview (there is no API for it).
- [ ] From a signed-out browser: README renders, the CI badge (if any) and links work,
      and the license is detected.
- [ ] Publish the drafted `v0.1.0` release, which creates the tag. Its notes repeat the
      claim scope from the README (historical measurements, one machine, text-only,
      mixed licensing).
