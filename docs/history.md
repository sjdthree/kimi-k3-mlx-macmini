# Historical engineering records

These records are archived **in place** to preserve incoming links and source anchors.
Their banners direct readers to current guidance. They retain hypotheses that proved
wrong, failures, machine-local commands, and intermediate measurements; those are
useful engineering history, not current release claims.

- `PROGRESS.md`, `SESSION.md`, `FINDINGS.md`: development chronology and corrections.
- `PORTING_BRIEF.md`, `K3_MLX_PLAN.md`, `apple_si_kimik3.md`: original plans/specifications.
- `P0_RESULTS.md`, `VALIDATION_REPORT.md`: scoped historical checks.
- `p0/`: exploratory programs, with additional optional dependencies; not the supported entry point.
- `docs/kv-cache/`: cache investigation, raw sweep data, and superseded interpretation.
- `results/`, tracked `runs/`: retained small evidence artifacts. New run directories are ignored.

KEEP raw parity logs, performance records, and commit history. ARCHIVE planning and
working notes as above. REMOVE FROM ACTIVE GUIDANCE the hardcoded sample credential,
machine-specific invocation paths, broad process-kill/output-deletion scripts, and
unsupported claims; those executable scripts have portable replacements. No historical
raw logs or commits were deleted. Generated caches, model weights, environments, package
builds, and secrets are ignored and must not be added with force.
