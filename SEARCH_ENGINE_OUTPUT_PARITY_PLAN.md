# Search Engine Output Parity Plan

_As of 2026-05-05._

## Purpose

Bring Prism, Topograph, Stratograph, Primordia, and the contender floor to the
same evidence quality level so near-future comparisons are measurable,
repeatable, and decision-grade.

## Implementation Status Snapshot

As of 2026-05-06, this plan is partly implemented already.

Landed or substantially landed:
- shared output envelopes in `EvoNN-Shared` (`RuntimeEnvelope`,
  `PerformanceEnvelope`, `DiagnosticsEnvelope`,
  `ArtifactCompletenessEnvelope`)
- Compare `output-quality` CLI plus normalized artifact/report generation
- workspace trend and dashboard surfaces in Compare
- historical baseline import workflow
- repo planning README now marks branch-plan status more truthfully
- Compare `performance-baseline` workflow now writes baseline bundles under
  `performance_baselines/<timestamp>-<git-sha>/`

Still ongoing after this branch:
- engine-by-engine L3/L4 closure on all named lanes, especially beyond the runs
  already represented in fixtures/tests
- dashboard surfacing of output-quality badges and downgrade reasons as a
  first-class visual layer everywhere
- cheap CI fixture coverage across all engines for recurring parity drift checks

This plan does **not** mean all engines should share the same architecture. It
means every engine must produce artifacts with the same audit quality:

- comparable budget semantics
- comparable benchmark coverage
- comparable runtime/backend metadata
- comparable performance metadata
- comparable failure diagnostics
- comparable summary/report surfaces
- engine-specific evidence carried through a shared envelope

The end state is that Compare can answer these questions without manual
interpretation:

- Did the engine complete the requested lane?
- Was the comparison fair enough to use?
- How much work did the engine actually spend?
- Which backend and hardware path produced the result?
- Did quality improve, regress, or stay inconclusive?
- Did performance improve without degrading trust or quality?

## Current Risks To Fix

### 1. Planning Hygiene Drift

Several branch plans are still listed as active after large parts of their work
have already landed. This makes it harder to tell whether the next task is
implementation, cleanup, or validation.

Fix:
- classify each active plan as `active`, `merged-validation`, `superseded`, or
  `archived`
- update `.hermes/plans/README.md` so it reflects execution reality
- keep this plan as the cross-engine output parity plan, not as another
  package-local advancement plan

### 2. Fragmented Performance Metadata

Some engines already emit useful runtime metadata, but the common Compare trend
surface does not yet normalize enough of it for a serious performance workflow.

Fix:
- define one shared performance metadata envelope
- make every engine export the fields it can support
- explicitly mark unsupported fields as unavailable instead of silently missing
  them
- propagate normalized performance fields into Compare trend rows and dashboard
  data

### 3. Compare Orchestration Growth

Compare is correctly becoming the evidence layer, but it risks becoming a large
procedural coordinator if validation, artifact normalization, dashboard logic,
historical import, campaign state, and future performance baselines continue to
grow without clearer boundaries.

Fix:
- keep Compare responsible for orchestration, validation, trend assembly, and
  decision surfaces
- keep canonical contracts and generic artifact helpers in Shared
- keep engine-specific search evidence in engine packages
- add small, named modules for output-quality validation and performance
  baseline handling instead of expanding fair-matrix internals indefinitely

### 4. Missing Performance Baseline Workflow

The code has timing and cache ingredients, but no canonical measurement loop:
baseline -> optimize one slice -> rerun -> approve or scrap.

Fix:
- add a Compare-owned performance baseline workflow
- require multiple budgets for performance claims
- require quality/fairness gates before speed claims count
- store performance baseline artifacts in a stable workspace layout

### 5. Uneven Engine Output Quality

The engines export comparable manifests and results, but richer evidence differs
by package. Topograph currently has stronger timing/cache details than the
others. Prism has a benchmark data cache. Primordia and Stratograph expose
runtime/backend state, but the common surface is still thinner.

Fix:
- define output quality levels
- audit each engine against the levels
- bring every engine to at least `L3 measurable` on `tier1_core`

## Output Quality Levels

### L0: Legacy Output

The engine writes some run artifacts, but Compare needs package-specific
interpretation or manual inspection.

This level is not acceptable for recurring comparisons.

### L1: Contract Output

The engine emits:

- `manifest.json`
- `results.json`
- `summary.json`
- `report.md`
- valid `RunManifest`
- valid `ResultRecord` entries for every benchmark in the requested pack

Minimum status:
- suitable for structural validation
- not yet enough for performance or decision-grade claims

### L2: Comparable Output

Everything from L1, plus:

- explicit fairness metadata
- budget policy name
- evaluation-counting semantics
- actual, cached, failed, invalid, and resumed evaluation counts where relevant
- data signature
- code version or git commit
- seed and pack identity
- complete benchmark status coverage, including failed, skipped, unsupported, or
  missing benchmarks

Minimum status:
- suitable for fair-matrix comparison
- failures are actionable

### L3: Measurable Output

Everything from L2, plus:

- requested backend
- resolved backend
- backend limitations
- device/framework metadata
- worker count
- wall-clock seconds
- per-benchmark train/evaluation seconds where supported
- data-load seconds where supported
- cache/reuse counts where supported
- peak memory or explicit unavailable marker
- artifact completeness summary
- performance fields propagated into Compare trend rows

Minimum status:
- suitable for performance baselines
- suitable for repeated trend analysis

### L4: Decision-Grade Output

Everything from L3, plus:

- quality-normalized performance summaries
- benchmark-family summaries
- failure taxonomy
- seeded/unseeded bucket identity where relevant
- confidence or variance view across seeds when multiple seeds are run
- dashboard slices that show full-system and EvoNN-only comparisons
- explicit decision recommendation:
  - `approve`
  - `reject`
  - `rerun-required`
  - `inconclusive`

Minimum status:
- suitable for PR advancement claims
- suitable for optimization approve/scrap decisions

## Required Artifact Bundle

Every engine should converge on this output bundle for Compare-grade runs.

### Required Files

- `manifest.json`: shared `RunManifest`
- `results.json`: list of shared `ResultRecord`
- `summary.json`: normalized run summary plus engine-specific evidence
- `report.md`: human-readable report generated from structured artifacts
- `diagnostics.json`: machine-readable failures, skips, unsupported benchmarks,
  warnings, and artifact completeness
- `performance.json`: normalized timing/cache/backend envelope, even when some
  values are unavailable
- `config_snapshot.*`: exact config used for the run

### Optional Files

- `dataset_manifest.json`
- `model_summary.json`
- `genome_summary.json`
- `primitive_bank_summary.json`
- `topology_atlas_summary.json`
- `contender_summary.json`
- package-local metrics database

Optional files may stay package-specific, but their existence and paths should
be referenced from `manifest.json` and summarized in `summary.json`.

## Normalized Summary Shape

Each `summary.json` should expose these top-level sections:

- `schema_version`
- `system`
- `run_identity`
- `lane`
- `budget`
- `runtime`
- `performance`
- `benchmark_coverage`
- `results_summary`
- `fairness`
- `seeding`
- `engine_evidence`
- `artifacts`
- `diagnostics`

### `run_identity`

Required fields:
- `run_id`
- `run_name`
- `created_at`
- `git_commit`
- `config_path`
- `config_hash`

### `lane`

Required fields:
- `pack_name`
- `benchmark_tier`
- `preset`
- `seed`
- `budget`
- `lane_operating_state` when Compare has assigned one

### `budget`

Required fields:
- `evaluation_count`
- `actual_evaluations`
- `cached_evaluations`
- `failed_evaluations`
- `invalid_evaluations`
- `resumed_from_run_id`
- `resumed_evaluations`
- `partial_run`
- `epochs_per_candidate`
- `effective_training_epochs`
- `budget_policy_name`
- `evaluation_semantics`

### `runtime`

Required fields:
- `runtime_backend_requested`
- `runtime_backend`
- `runtime_backend_limitations`
- `device_name`
- `framework`
- `framework_version`
- `precision_mode`
- `hardware_class`
- `worker_count`
- `os`
- `python_version`

### `performance`

Required fields:
- `wall_clock_seconds`
- `benchmark_total_seconds`
- `data_load_seconds`
- `evaluation_seconds`
- `export_seconds`
- `train_seconds_total`
- `train_seconds_mean`
- `evals_per_second`
- `quality_per_second`
- `cache_hits`
- `cache_misses`
- `cache_reuse_count`
- `cache_reuse_rate`
- `requested_worker_count`
- `resolved_worker_count`
- `worker_clamp_reason`
- `peak_memory_mb`

Unsupported values should be represented as `null` plus a diagnostic note, not
silently omitted.

### `benchmark_coverage`

Required fields:
- `requested_benchmarks`
- `completed_benchmarks`
- `failed_benchmarks`
- `skipped_benchmarks`
- `unsupported_benchmarks`
- `missing_benchmarks`
- `status_by_benchmark`
- `failure_reason_by_benchmark`

### `results_summary`

Required fields:
- `best_metric_by_benchmark`
- `quality_by_benchmark`
- `parameter_count_by_benchmark`
- `metric_direction_by_benchmark`
- `task_kind_by_benchmark`

### `fairness`

Required fields:
- `benchmark_pack_id`
- `data_signature`
- `seed`
- `evaluation_count`
- `budget_policy_name`
- `code_version`
- `pairwise_fairness_eligible`
- `fairness_blockers`

### `seeding`

Required fields:
- `seeding_enabled`
- `seeding_ladder`
- `seed_source_system`
- `seed_source_run_id`
- `seed_artifact_path`
- `seed_target_family`
- `seed_selected_family`
- `seed_rank`
- `seed_overlap_policy`

### `engine_evidence`

This section stays package-specific, but must be present for every engine.

Prism should include:
- family distribution
- archive occupancy
- inheritance or transfer usage
- default-engine decision notes

Topograph should include:
- topology size
- novelty metrics
- MAP-Elites occupancy
- mutation operator success
- parallel/cache behavior

Stratograph should include:
- macro depth
- cell library size
- reuse ratio
- motif frequency summary
- hierarchy/ablation evidence where available

Primordia should include:
- primitive count
- motif complexity
- primitive bank size
- promotion count
- fallback/runtime limitations

Contenders should include:
- contender family coverage
- contender configuration ids
- optional dependency skips
- baseline floor policy stage

## Workstreams

### 1. Planning Hygiene And Current-State Reset

Goal:
make the active plan hierarchy truthful before new branches multiply.

Tasks:
- audit `.hermes/plans/*` against merged work
- mark each plan as `active`, `merged-validation`, `superseded`, or `archived`
- update package READMEs if they still point to stale implementation plans
- add this plan to the active root plan list

Acceptance criteria:
- a new engineer can identify the active output-parity work in under 10 minutes
- no merged branch plan is presented as untouched future work

### 2. Shared Output Contract Extension

Goal:
make the common artifact vocabulary explicit enough for L3/L4 output quality.

Tasks:
- add shared models for:
  - `RuntimeEnvelope`
  - `PerformanceEnvelope`
  - `DiagnosticsEnvelope`
  - `ArtifactCompletenessEnvelope`
- decide whether these live inside `RunManifest`, `summary.json`, or both
- provide shared helper functions for normalized `summary.json`,
  `diagnostics.json`, and `performance.json`
- keep generic helpers in Shared, not in engine packages

Acceptance criteria:
- every engine can call one shared writer for normalized output files
- Compare can validate the normalized files without package-specific parsing

### 3. Compare Output Quality Validator

Goal:
make output quality measurable and enforceable.

Tasks:
- add `evonn-compare output-quality` CLI
- validate a run directory against L1/L2/L3/L4
- produce `output_quality_report.json` and `output_quality_report.md`
- include per-engine missing fields and downgrade reasons
- wire output-quality summaries into fair-matrix lane acceptance

Acceptance criteria:
- Compare can say `prism=L3`, `topograph=L3`, `stratograph=L2`, etc.
- fair-matrix can downgrade decision state when output quality is insufficient
- missing performance fields are visible before a performance claim is made

### 4. Engine Export Parity Passes

Goal:
bring each engine to L3 on `tier1_core`.

Tasks:
- Prism:
  - export benchmark cache stats
  - export backend/device metadata consistently
  - emit `performance.json` and `diagnostics.json`
  - ensure eval1000 runtime fixes are represented in measurable artifacts
- Topograph:
  - keep existing rich timing/cache metadata
  - move generic parts into shared envelopes
  - ensure worker clamp and data cache stats reach Compare trends
- Stratograph:
  - normalize backend/runtime metadata to the shared envelope
  - emit per-benchmark timing and diagnostic details
  - ensure hierarchy evidence is under `engine_evidence`
- Primordia:
  - normalize fallback limitations and primitive-bank evidence
  - emit per-benchmark timing and diagnostic details
  - make promotion/primitive evidence comparable at summary level
- Contenders:
  - normalize baseline coverage and optional dependency skips
  - emit per-contender timing and backend metadata
  - keep baseline floor policy visible in diagnostics

Acceptance criteria:
- all five systems reach L3 on `tier1_core@64`
- all five systems either reach L3 on `tier1_core@256/1000` or explain
  downgraded fields with actionable diagnostics

### 5. Performance Baseline Workflow

Goal:
turn performance work into a controlled measurement loop.

Tasks:
- add `evonn-compare performance-baseline`
- run multiple budgets:
  - `64`
  - `256`
  - `1000`
- support multiple seeds when runtime allows
- write artifacts under:
  - `performance_baselines/<date>-<git-sha>/`
- require quality/fairness gates before performance results count
- compute:
  - wall-clock trend
  - evals per second
  - quality per second
  - benchmark throughput
  - cache/reuse effect
  - backend/hardware label
  - failure-adjusted throughput

Acceptance criteria:
- a branch can compare itself against a baseline without manual spreadsheet work
- an optimization PR can be approved, rejected, or marked inconclusive from
  structured artifacts

### 6. Dashboard And Trend Surface Upgrade

Goal:
make output parity and performance visible in the same decision surface.

Tasks:
- add output-quality badges per system/run
- add performance slices:
  - budget-normalized runtime
  - evals per second
  - quality per second
  - backend/runtime comparison
  - cache/reuse impact
  - failure-adjusted leaderboard
- keep existing full-system and EvoNN-only leaderboards
- show downgrade reasons directly in dashboard data

Acceptance criteria:
- the dashboard answers whether a run is comparable, measurable, and useful for
  decisions
- performance claims do not require opening package-local reports first

### 7. CI And Fixtures

Goal:
prevent output drift from reappearing.

Tasks:
- add golden fixture exports for each engine
- validate fixture exports at L3 in CI
- add snapshot tests for normalized `summary.json`, `performance.json`, and
  `diagnostics.json`
- keep Linux-safe packages on Linux CI
- keep MLX-native truth paths on macOS CI
- add one cheap output-quality smoke case to Compare CI

Acceptance criteria:
- Shared contract changes break tests when engines drift
- engine export regressions are caught before fair-matrix runs fail

### 8. Branch Execution Model

Goal:
avoid duplicate and diverging implementations.

Recommended branch breakdown:
- `feat/shared-output-quality-contracts`
- `feat/compare-output-quality-validator`
- `feat/prism-output-parity`
- `feat/topograph-output-parity`
- `feat/stratograph-output-parity`
- `feat/primordia-output-parity`
- `feat/contenders-output-parity`
- `feat/performance-baseline-workflow`
- `feat/dashboard-output-quality-performance`

Merge order:
1. Shared contracts and writers
2. Compare validator
3. one engine parity pass as reference implementation
4. remaining engine parity passes
5. performance baseline workflow
6. dashboard/trend upgrades
7. planning cleanup and docs finalization

Rule:
- shared schemas, writers, and validators must not be reimplemented in engine
  branches
- engine branches may add package-local evidence only under `engine_evidence`
- Compare owns cross-engine judgment, not engine packages

## Milestones

### Milestone 1: Output Quality Spec Ratified

Exit criteria:
- this plan is accepted as the active output-parity plan
- L1/L2/L3/L4 definitions are documented
- required artifact bundle is documented
- branch breakdown is agreed

### Milestone 2: Shared And Compare Foundation

Exit criteria:
- shared runtime/performance/diagnostics envelopes exist
- Compare can validate output quality levels
- fixture tests cover at least one complete L3 run

### Milestone 3: All Engines Reach L3 On `tier1_core@64`

Exit criteria:
- Prism, Topograph, Stratograph, Primordia, and Contenders emit L3 output on
  the local lane
- Compare dashboard can show output quality per system
- failures are explicit and actionable

### Milestone 4: Higher-Budget L3 Coverage

Exit criteria:
- `tier1_core@256` reaches L3 for all systems or has documented downgrade
  reasons
- `tier1_core@1000` completes or reports exact blockers per system and
  benchmark
- no benchmark silently disappears from summary or trend data

### Milestone 5: Performance Baseline Ready

Exit criteria:
- performance baseline workflow runs across `64/256/1000`
- baseline artifacts are stored in a stable layout
- Compare can approve/scrap optimization claims from structured evidence

### Milestone 6: Decision-Grade Output

Exit criteria:
- at least Prism, Topograph, and Contenders reach L4 on the trusted core lane
- Stratograph and Primordia have clear L4 gaps or reach L4 as extended systems
- dashboard becomes the primary evidence surface for quality, fairness, and
  performance

## Definition Of Done

This plan is complete when:

- every search engine emits the required artifact bundle
- every engine reaches L3 on `tier1_core@64`
- `tier1_core@256` and `tier1_core@1000` are measurable even when not fully
  trusted
- Compare exposes output-quality levels in reports and dashboards
- performance baselines cover multiple budgets
- optimization PRs can be approved or rejected from structured evidence
- stale active plans are archived or marked with accurate status
- no shared output, budget, summary, diagnostics, or performance helper is
  duplicated inside package-local engine code

## First Implementation Slice

The first PR should not touch all engines.

Recommended first slice:
1. add Shared `RuntimeEnvelope`, `PerformanceEnvelope`, and
   `DiagnosticsEnvelope`
2. add Compare `output-quality` validator
3. add one L3 fixture for the engine with the richest current metadata,
   preferably Topograph
4. document exact missing fields for the other engines

This creates the measurement ruler before applying it broadly.
