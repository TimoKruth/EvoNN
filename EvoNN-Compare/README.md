# EvoNN-Compare

Protocol-first comparison layer for Prism, Topograph, Stratograph, Hybrid, and contender exports.

Prefer root-level `uv` commands from monorepo root:

```bash
uv sync --package evonn-compare --extra dev
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
uv run --package evonn-compare python -m evonn_compare --help
```

Package metadata stays in [pyproject.toml](./pyproject.toml). Workspace lock lives at monorepo root.

## Fair-matrix preset ladder

The local-first preset ladder now exposes the quarter-critical `tier1_core`
budgets directly and includes explicit Tier B variants:

- `smoke` → `tier1_core_smoke` @ `16`
- `local` → `tier1_core` @ `64`
- `overnight` → `tier1_core` @ `256`
- `weekend` → `tier1_core` @ `1000`
- `tier_b_local` → `tier_b_core` @ `64`
- `tier_b_overnight` → `tier_b_core` @ `256`
- `tier_b_weekend` → `tier_b_core` @ `1000`

`fair-matrix` and `campaign` now default to the trusted daily `local`
lane (`tier1_core` @ `64`) when neither `--pack` nor `--preset` is supplied.
If you target a parity pack directly with `--pack`
without a preset, the default budget now comes from that pack's declared
`budget_policy.evaluation_count` unless you override `--budgets` explicitly:

```bash
uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --workspace .tmp/fair-matrix-local

uv run --package evonn-compare python -m evonn_compare campaign \
  --workspace .tmp/campaign-local

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset overnight \
  --workspace .tmp/fair-matrix-overnight

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset weekend \
  --workspace .tmp/fair-matrix-weekend

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset tier_b_overnight \
  --workspace .tmp/fair-matrix-tier-b-overnight

uv run --package evonn-compare python -m evonn_compare fair-matrix \
  --preset tier_b_weekend \
  --workspace .tmp/fair-matrix-tier-b-weekend
```

`tier_b_core` is the canonical benchmark-ladder Tier B pack. It is a broader
local research workbench than the trusted `tier1_core` recurring lane, and it
resolves from `shared-benchmarks/suites/parity/` when you reference it by
pack name. The checked-in Tier B preset ladder currently stops at `1000`; an
around-`2500` preset is intentionally not named yet because the repo does not
carry stable runtime evidence for it.

For `campaign`, the CLI prints the generated `campaign.yaml` manifest,
per-case compare markdown/JSON report paths, Prism/Topograph run directories,
and per-case log directory so the core two-system lane can be inspected without
manually walking the workspace tree.

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

## Transfer-regime runner

`evonn-compare transfer-regimes` is the compare-facing surface for the transfer
workstream behind `EVO-52`. It keeps `none`, `direct`, and `staged` as
separate regimes by construction and writes transfer verdicts against the
no-seed control instead of collapsing everything into one anonymous seeded
bucket.

Use it from the repo root:

```bash
uv run --package evonn-compare evonn-compare transfer-regimes \
  --workspace .tmp/transfer-tier-b \
  --preset tier_b_local \
  --seeds 41,42 \
  --open
```

What it does:

- resolves the requested pack and budget into a workspace-local compare pack
- runs Primordia once per seed to materialize a direct seed artifact
- gates direct seed artifacts before they are consumed
- runs Topograph in three explicit regimes on the same pack/budget/seed:
  - `none`
  - `direct`
  - `staged`
- builds a compare-owned staged seed artifact from the prior direct run so the
  staged regime is auditable even before native staged runtime support exists
- writes per-seed regime-vs-control reports plus a multi-seed aggregate summary
- refreshes the shared trend report and dashboard for the workspace

Key outputs:

- `reports/transfer_regime_summary.md`
- `reports/transfer_regime_summary.json`
- `reports/seed*/02-direct_vs_control.md`
- `reports/seed*/03-staged_vs_control.md`
- `seed_artifacts/seed*_direct_quality.json`
- `seed_artifacts/seed*_staged_quality.json`

Current boundary:

- direct and staged provenance are carried in compare manifests and summaries
- portable fallback remains a compare-plumbing proof path, not native staged
  transfer proof for Topograph

By default, repeated `fair-matrix` runs preserve the managed workspace so trend,
report, and dashboard artifacts continue to accumulate across recurring reruns.
Use `--reset-workspace` only when you intentionally want a fresh workspace.

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

After `fair-matrix` execution, the CLI refreshes the workspace-level trend report and dashboard automatically from the canonical JSON artifacts. It prints the paths for the case summary markdown/JSON, lane acceptance metadata, structured case trend JSON/JSONL artifacts, workspace trend dataset/report/report-JSON, and workspace dashboard so reruns can be reviewed from the longitudinal surface first.

This means repeated recurring-lane runs can be appended to one shared trend dataset without per-engine parsers or markdown scraping. The trend markdown now also surfaces lane accounting and repeatability state directly, so budget-truth drift is visible from the default human review surface. It also separates per-seed aggregate snapshots from multi-seed evidence so raw seed variance remains visible beside the statistical rollup.

### Workspace-first review flow

The default review surface is now the workspace, not an individual markdown snapshot:

```bash
uv run --package evonn-compare evonn-compare fair-matrix \
  --workspace .tmp/fair-matrix-smoke \
  --open
```

That command refreshes:

- `.tmp/fair-matrix-smoke/trends/fair_matrix_trends.md`
- `.tmp/fair-matrix-smoke/trends/fair_matrix_trends.json`
- `.tmp/fair-matrix-smoke/fair_matrix_dashboard.html`
- `.tmp/fair-matrix-smoke/fair_matrix_dashboard.json`

With `--open`, the command lands directly on the canonical dashboard after the
run, so the recurring-lane review path stays one command from execution to the
full-system and projects-only leaderboard views.

To rebuild those workspace-level views later without rerunning engines:

```bash
uv run --package evonn-compare evonn-compare workspace-report \
  .tmp/fair-matrix-smoke
```

`workspace-report` prints the refreshed markdown and JSON trend-report paths directly, along with the dashboard outputs.

Use the workspace trend report and dashboard first for questions like:

- did this improve anything?
- did fairness/accounting status drift?
- which lane operating state are we actually in?
- are failures or missing benchmarks increasing over time?

For branch-advancement claims, this workspace-first review flow feeds the
repo-wide [research decision gate](../RESEARCH_DECISION_GATE.md). PRs should
link the exact workspace trend report, dashboard, case IDs, run IDs, and named
dashboard slices used for the claim.

### Historical baseline comparison workflow

`historical-baseline` is the compare-owned path for loading prior fair-matrix
campaigns into a live workspace without editing JSON by hand. It imports one or
more historical summary directories or files under `workspace/baselines/<label>/`,
records compatibility and integrity metadata, and then refreshes the canonical
trend report plus dashboard from both the active workspace and the imported
baseline cohort.

Use it from the repo root:

```bash
uv run --package evonn-compare evonn-compare historical-baseline \
  .tmp/fair-matrix-local \
  /path/to/historical/workspace \
  --label release-2026-04-01
```

What it does:

- discovers `fair_matrix_summary.json` artifacts from the supplied historical input
- imports them into a compare-owned baseline cohort inside the active workspace
- annotates imported trend rows with:
  - comparison cohort
  - comparison label
  - comparison case id
  - baseline source path
  - compatibility and integrity metadata
- rebuilds the workspace trend dataset, trend markdown, and dashboard from the
  merged current plus baseline evidence

Key outputs:

- `baselines/<label>/baseline_manifest.json`
- `baselines/<label>/trends/fair_matrix_trend_rows.jsonl`
- `trends/fair_matrix_trends.md`
- `fair_matrix_dashboard.html`

Current comparison semantics:

- seed ids are preserved exactly as emitted by the historical run
- overlapping numeric seeds stay separate because compare now carries
  `comparison_case_id` and `comparison_label` through the trend pipeline
- compatibility assumptions and integrity findings are written into the baseline
  manifest and surfaced in imported summary artifacts

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
- multi-seed aggregate evidence with score spread, confidence intervals, and pairwise seed deltas
- per-seed aggregate snapshots so noisy wins are not mistaken for stable wins

Those two leaderboard surfaces are the primary recurring review views: the
five-system table answers "what won across the full substrate?", and the
projects-only table answers "what changed among the four product engines once
contenders are removed from winner selection?"

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

Prism is now the default operating path for the routine trusted daily compare flow:

- `campaign` defaults to the `local` `tier1_core` lane when no `--pack` or `--preset` is supplied
- `fair-matrix` defaults to the same `local` `tier1_core` lane when no `--pack` or `--preset` is supplied
- the default path still keeps Topograph on the same shared compare/report surface as the first challenger

Functional shared-surface checks currently covered in Compare tests:

- Prism + Topograph default/local fair-matrix flow
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
