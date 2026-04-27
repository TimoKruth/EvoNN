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

The local-first preset ladder now exposes the quarter-critical `tier1_core`
budgets directly:

- `smoke` → `tier1_core_smoke` @ `16`
- `local` → `tier1_core` @ `64`
- `overnight` → `tier1_core` @ `256`
- `weekend` → `tier1_core` @ `1000`

`fair-matrix` and `campaign` default to `smoke` when neither `--pack` nor
`--preset` is supplied:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke

uv run --package evonn-compare python -m evonn_compare campaign \
  --workspace .tmp/campaign-smoke

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset overnight \
  --workspace .tmp/fair-matrix-overnight

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset weekend \
  --workspace .tmp/fair-matrix-weekend
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
- `fair_matrix_dashboard.html`
- `fair_matrix_dashboard.json`

After `fair-matrix` execution, the CLI refreshes the workspace-level trend report and dashboard automatically from the canonical JSON artifacts. It prints the paths for the case summary markdown/JSON, lane acceptance metadata, structured case trend JSON/JSONL artifacts, workspace trend dataset/report, and workspace dashboard so reruns can be reviewed from the longitudinal surface first.

This means repeated `smoke` lane runs can be appended to one shared trend dataset without per-engine parsers or markdown scraping.

### Workspace-first review flow

The default review surface is now the workspace, not an individual markdown snapshot:

```bash
uv run --package evonn-compare evonn-compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke
```

That command refreshes:

- `.tmp/fair-matrix-smoke/trends/fair_matrix_trends.md`
- `.tmp/fair-matrix-smoke/fair_matrix_dashboard.html`
- `.tmp/fair-matrix-smoke/fair_matrix_dashboard.json`

To rebuild those workspace-level views later without rerunning engines:

```bash
uv run --package evonn-compare evonn-compare workspace-report \
  .tmp/fair-matrix-smoke
```

Use the workspace trend report and dashboard first for questions like:

- did this improve anything?
- did fairness/accounting status drift?
- which lane operating state are we actually in?
- are failures or missing benchmarks increasing over time?

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
- `fair_matrix_trends.json`
- `fair_matrix_trends.jsonl`

Optional filters:

- `--system`
- `--benchmark`
- `--pack`

When `--output` is provided, the command writes:

- markdown report at the requested path
- filtered JSON rows beside it as the same path with `.json` suffix

The CLI also prints both output paths directly so the generated trend artifacts can
be picked up without inspecting the filesystem manually.

### Static dashboard

Use `dashboard` to scan one or more fair-matrix workspaces or summary files and
render a static HTML overview with:

- five-system benchmark-winner tables
- project-only benchmark-winner tables that recompute winners without contenders
- aggregate leaderboards across all discovered runs

By default it scans `EvoNN-Compare/manual_compare_runs`:

```bash
uv run --package evonn-compare evonn-compare dashboard
```

Optional output path:

```bash
uv run --package evonn-compare evonn-compare dashboard \
  EvoNN-Compare/manual_compare_runs \
  --output EvoNN-Compare/manual_compare_runs/fair_matrix_dashboard.html
```

The command writes:

- HTML dashboard at `--output`
- structured dashboard payload beside it as the same path with `.json` suffix

For ad hoc `compare` runs, using `--output` also prints both the markdown report
path and the sibling JSON artifact path directly in CLI output.

## Milestone 6: Prism default operating path

Prism is now the default operating path for the routine low-cost compare flow:

- `campaign` defaults to the `smoke` lane when no `--pack` or `--preset` is supplied
- `fair-matrix` defaults to the same `smoke` lane when no `--pack` or `--preset` is supplied
- the default path still keeps Topograph on the same shared compare/report surface as the first challenger

Functional shared-surface checks currently covered in Compare tests:

- Prism + Topograph default/smoke fair-matrix flow
- Stratograph config generation on the shared budget/benchmark surface
- Primordia config generation on the shared budget/benchmark surface
- four-system fair-matrix orchestration with Prism, Topograph, Stratograph, and Primordia contributing artifact-complete outputs in the test harness

This is a default operating path, not a monoculture claim: Topograph remains the first challenger lane, while Stratograph and Primordia stay on the same compare/export substrate and are checked for continued functional participation.

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
