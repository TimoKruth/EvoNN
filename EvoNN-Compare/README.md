# EvoNN-Compare

Protocol-first comparison layer for Prism, Topograph, Stratograph, Hybrid, and contender exports.

Prefer root-level `uv` commands from monorepo root:

```bash
uv sync --package evonn-compare --extra dev
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
uv run --package evonn-compare python -m evonn_compare --help
```

Package metadata stays in [pyproject.toml](./pyproject.toml). Workspace lock lives at monorepo root.

## Milestone 5: trend-capable reporting

EvoNN-Compare now treats fair-matrix outputs as longitudinal research artifacts instead of markdown-only snapshots.

### Structured trend artifacts

Each fair-matrix case writes:

- `fair_matrix_summary.md`
- `fair_matrix_summary.json`
- `trend_rows.json`
- `trend_report.md`

Each fair-matrix workspace also accumulates:

- `fair_matrix_trend_rows.jsonl`
- `fair_matrix_trends.md`

This means repeated `smoke` lane runs can be appended to one shared trend dataset without per-engine parsers or markdown scraping.

### Fixed minimum longitudinal dimensions

Every trend row preserves the shared Milestone-5 dimensions:

- `system`
- `benchmark_id`
- `pack_name`
- `budget`
- `seed`
- `run_id`
- `outcome_status`
- `failure_reason`
- `metric_name`
- `metric_direction`
- `metric_value`
- `evaluation_count`
- `epochs_per_candidate`
- `budget_policy_name`
- `wall_clock_seconds`
- `matrix_scope`
- `fairness_metadata`

`fairness_metadata` keeps the comparison context visible in downstream reports, including benchmark-pack identity, seed, evaluation count, budget-policy disclosure, data-signature provenance, code version, and whether the matrix remained fully fair or fell back to reference-only scope.

### Trend reporting CLI

Use `trend-report` to merge and query one or more trend datasets:

```bash
uv run --package evonn-compare evonn-compare trend-report \
  /path/to/fair_matrix_trend_rows.jsonl \
  --system prism \
  --benchmark iris_classification \
  --output trend_report.md
```

Accepted inputs:

- `trend_rows.json`
- `fair_matrix_summary.json`
- `fair_matrix_trend_rows.jsonl`

Optional filters:

- `--system`
- `--benchmark`
- `--pack`

When `--output` is provided, the command writes:

- markdown report at the requested path
- filtered JSON rows beside it as the same path with `.json` suffix
