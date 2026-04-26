# EvoNN-Compare

Protocol-first comparison layer for Prism, Topograph, Stratograph, Hybrid, and contender exports.

Prefer root-level `uv` commands from monorepo root:

```bash
uv sync --package evonn-compare --extra dev
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
uv run --package evonn-compare python -m evonn_compare --help
```

Package metadata stays in [pyproject.toml](./pyproject.toml). Workspace lock lives at monorepo root.

## Fair-matrix smoke lane

The provisional low-cost repeatable lane is the `smoke` preset. Both
`fair-matrix` and `campaign` default to this lane when neither `--pack` nor
`--preset` is supplied:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke

uv run --package evonn-compare python -m evonn_compare campaign \
  --workspace .tmp/campaign-smoke
```

Phase-1 acceptance for milestones 4-5 is captured directly in the emitted artifacts:

- `reports/<case>/lane_acceptance.json`
  - artifact completeness
  - pairwise fairness status
  - classification + regression task coverage
  - budget consistency against the requested lane budget
  - seed consistency against the requested lane seed
- `reports/<case>/fair_matrix_summary.json`
  - machine-readable fair/reference/parity summary
- `reports/<case>/fair_matrix_trends.jsonl`
  - structured longitudinal records derived from JSON artifacts only
- `trends/fair_matrix_trends.jsonl`
  - append-only workspace trend dataset for repeated reruns

Minimum longitudinal fields preserved per record:

- engine
- benchmark
- pack
- budget
- seed
- outcome status
- metric direction/value
- fairness metadata

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

## Intentional remaining engine-specific branches

After the shared-substrate convergence work, Compare still keeps a small set of
engine-specific branches on purpose:

- benchmark/module resolution remains system-specific because each engine still
  owns native benchmark identifiers and its own registry/runtime loading path
- config generation and command invocation remain system-specific because
  Prism, Topograph, Stratograph, Primordia, and Contenders still expose
  different CLIs and runtime prerequisites
- portable smoke exporters may keep small system-local fields when they
  describe a real runtime difference rather than shared compare semantics

These branches are intentional. The debt that still needs elimination is any
Compare-side branch that exists only because shared contracts/helpers were not
adopted yet, or any special handling that changes comparability semantics
without reflecting a real engine/runtime difference.
