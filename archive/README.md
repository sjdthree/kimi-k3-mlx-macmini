# Archive

Older engineering records, kept because they are sometimes interesting and because code
comments cite them. None of this is current guidance. Each file carries a banner that
points to the current documents. They keep hypotheses that proved wrong, failures,
machine-local commands and intermediate measurements; that is useful history, not
release claims. For the project as it stands, start at the [root README](../README.md)
and the [evidence ledger](../docs/evidence.md).

| File | What it is |
|---|---|
| [PROGRESS.md](PROGRESS.md), [SESSION.md](SESSION.md), [FINDINGS.md](FINDINGS.md) | Development chronology and corrections |
| [PORTING_BRIEF.md](PORTING_BRIEF.md), [K3_MLX_PLAN.md](K3_MLX_PLAN.md), [apple_si_kimik3.md](apple_si_kimik3.md) | Original plans and specifications |
| [P0_RESULTS.md](P0_RESULTS.md), [VALIDATION_REPORT.md](VALIDATION_REPORT.md) | Scoped historical checks |
| [p0/](p0/README.md) | Feasibility spikes that ran before any MLX code existed; extra dependencies, unsupported |

Historical evidence that current claims rely on is **not** here: the cache investigation
and sweep data stay in [docs/kv-cache/](../docs/kv-cache/README.md), and the small run
artifacts in `results/` and the tracked `runs/`.

KEEP raw parity logs, performance records, and commit history. ARCHIVE planning and
working notes as above. REMOVE FROM ACTIVE GUIDANCE the hardcoded sample credential,
machine-specific invocation paths, broad process-kill/output-deletion scripts, and
unsupported claims; those executable scripts have portable replacements. No historical
raw logs or commits were deleted. Generated caches, model weights, environments, package
builds, and secrets are ignored and must not be added with force.
