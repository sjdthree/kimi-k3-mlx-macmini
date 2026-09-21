# Release validation: full checkpoint, 2026-09-21

Step 2 of [docs/release.md](../../docs/release.md), run on the benchmark machine.
Everything ran in sequence from [run.sh](run.sh) under `caffeinate -is`, one model
process at a time. All steps passed.

## Configuration

| | |
|---|---|
| Commit | `797f0acb98b46027ca4154c43c2e312f8d07e228`, clean tree |
| `K3_REVISION` | `c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721` (moonshotai/Kimi-K3; content identical in every public revision, see [licensing.md](../../docs/licensing.md)) |
| Checkpoint | USB4 NVMe enclosure, 2 TB APFS, rear Thunderbolt port; `IONVMeController`, PCI-Express, external |
| Dense cache | `~/k3_dense` on the internal SSD (106 GiB, staged earlier; not rebuilt) |
| Manifest | rebuilt: 497,220 tensors, 96 shards; `verify` PASS ([build](manifest-build.log), [verify](manifest-verify.log)) |
| Hardware | Mac16,11, Apple M4 Pro, 68,719,476,736 bytes unified memory |
| OS | macOS 26.6.2 (25G83) |
| Python | 3.12.12, fresh venv, `pip install -c constraints-test.txt -e '.[dev]'` |
| Key packages | mlx 0.29.3, mlx-metal 0.29.3, numpy 2.2.6, torch 2.14.0, tiktoken 0.14.0 (full list in [environment.txt](environment.txt)) |
| Other processes | none matching `air_llm`, `run_long_job` or `bench`, checked before every step |
| Swap | 1,743.56 MiB used before, 1,746.12 MiB after, of 3,072 MiB. The swap was already in use before the run started (memory_pressure reported 94% free); it grew 2.6 MiB during the run. |

Timeline with swap at each step boundary: [timeline.txt](timeline.txt).

## Results

| Step | Result | Duration | Log |
|---|---|---|---|
| a. `tests/run_all.py` with `K3_CKPT` | **18 passed, 0 skipped, 0 failed** | 3 m 12 s | [4a-run_all.log](4a-run_all.log) |
| b. Full-depth smoke test (`--check-finite --validation-log`) | exit 0, 93 layers, 0 nonfinite | 3 m 02 s | [4b-smoke.log](4b-smoke.log), [smoke.jsonl](smoke.jsonl) |
| c. `tests/test_numerical_modes.py` | **MODES: PASS** (5/5) | 50 s | [4c-numerical_modes.log](4c-numerical_modes.log) |
| d. `tests/test_full_parity.py` | **M9: PASS** | 10 m 16 s | [4d-full_parity.log](4d-full_parity.log) |

Step d is `run_all.py --include-slow` split in two. `run_all.py` runs `test_full_parity`
with no arguments, so under the same environment `--include-slow` is step a plus step d.
Running step d directly keeps its full output, which `run_all.py` discards for passing
suites.

### Full parity (d)

Prompt tokens `[1008, 10484, 318, 15383, 387]`, one generated token, fp32 validation
weights, against the Moonshot-derived PyTorch/CPU oracle.

| | This run | Historical (`docs/kv-cache/suite-logs/`) |
|---|---|---|
| Worst per-layer cosine | **0.99999976 at layer 14** | 0.99999988 at layer 22 |
| Router selections | **92/92 exact** | 92/92 |
| Greedy token | **17374** (oracle 17374) | 17374 |
| Wall time | 613 s | 3,075 s |

The worst-layer cosine moved from 0.99999988 to 0.99999976. It is still far above the
0.999 gate. The validation path is now genuinely unfused, so some movement is expected.

The run took 5× less wall time than the historical one. The cause was not isolated.
The historical runs used the conda `airllm` environment (SESSION.md). Today that
environment has torch 2.13.0, mlx 0.32.0 and numpy 2.5.1, but nothing records whether
those were the versions installed at the time of the historical run. The oracle runs on
the CPU, so a dependency change is a plausible cause but an unverified one. So is the
changed validation path. INSTALL.md's "potentially hours" is now pessimistic for one
token on this machine.

### Smoke test (b)

`--mode runtime --resident-gb 8 --max-memory-gb 48`, prompt `The capital of France is`,
4 greedy tokens.

| Token | id | Seconds | Logical read GiB | MLX peak GiB | nonfinite |
|---:|---:|---:|---:|---:|---:|
| 0 (includes prefill) | 17374 | 78.6 | 196.38 | 11.47 | 0 |
| 1 | 20829 | 33.8 | 119.58 | 11.35 | 0 |
| 2 | 10 | 33.3 | 119.58 | 11.32 | 0 |
| 3 | 427 | 33.7 | 119.58 | 11.32 | 0 |

Output: `'The capital of France is Paris.",\n+           '`. The answer is correct. After
it, the model continues with code/diff punctuation, which is plausible for a raw
five-token prompt with no chat template. Treat this as a finite, full-depth run with a
correct first answer, not as a quality evaluation.

Three decode samples are not a benchmark and are not comparable with the 39-sample
32.43 s/token mean. The logical read per decode token (119.58 GiB) matches the
historical 8 GiB steady figure (119.5796 GiB) exactly. MLX peak is an allocator
high-water mark, not process or system memory.

### Numerical modes (c)

Validation and runtime modes choose identical greedy tokens for three prompts. The
resident cache (65.4% hit rate, 3.36 GiB) leaves the tokens unchanged, and runtime mode
is deterministic across repeated runs. These are short two-layer runs, not full-depth
mode equivalence.
