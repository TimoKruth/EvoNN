# Primordia Quality Scorecard

_Last updated: 2026-04-27 on branch `feat/primordia-quality-parity-execution`._

This scorecard closes the branch with an explicit distinction between what was
validated on the shared Linux workspace and what still requires an Apple
Silicon / downstream seeding pass before stronger parity claims would be honest.

## Current decision

Primordia is ready to claim:
- workspace-integrated package validation
- benchmark-complete fallback execution on the official smoke and tier-1 lanes
- compare/fair-matrix participation with fair pairwise parity on the smoke lane
- explicit non-MLX portability without hiding the runtime used

Primordia is **not** ready to claim:
- MLX smoke or tier-1 validation from this workspace
- training-depth parity against stronger engines at higher budgets
- seeded-transfer usefulness parity downstream

## Scorecard

| Area | Status | Evidence | Decision |
|---|---|---|---|
| Workspace dependency fix | green | commit `95f96a2`; `bash scripts/ci/primordia-checks.sh all` passed on 2026-04-27 | closed |
| Fallback smoke lane | green | `.artifacts/primordia-smoke-linux-20260427` completed 7/7 benchmarks, 21 evals, 0 failures | closed |
| Fallback tier-1 lane | green | `.artifacts/primordia-tier1-64-linux-20260427` completed 8/8 benchmarks, 64 evals, 0 failures | closed |
| Compare fair-matrix smoke acceptance | yellow | `.artifacts/fair-matrix-smoke-core-20260427/reports/tier1_core_smoke_eval16_seed42/lane_acceptance.json` shows artifact completeness, fairness, task coverage, budget consistency, and seed consistency all true | accepted with scope note |
| MLX smoke validation | red | explicit `runtime.backend: mlx` smoke attempt on 2026-04-27 failed with `RuntimeError: mlx is not importable in this environment` | narrowed out of branch completion |
| MLX tier-1 validation | red | no Apple Silicon workspace evidence attached in this branch | narrowed out of branch completion |
| Higher-budget training-depth parity | red | no new branch-local parity evidence for `tier1_core_eval256` or `tier1_core_eval1000` beyond fallback baselines in `BASELINE_MATRIX.md` | narrowed out of branch completion |
| Seeded-transfer parity | red | seed artifacts exist, but no downstream seeded-vs-unseeded experiment is attached here | narrowed out of branch completion |

## Validated artifacts

### Package validation
- Command: `bash scripts/ci/primordia-checks.sh all`
- Result: 51 tests passed
- Meaning: the workspace dependency fix and the Primordia package surfaces are green from the dedicated branch worktree

### Official fallback runs
- Smoke: `.artifacts/primordia-smoke-linux-20260427`
  - runtime: `numpy-fallback`
  - framework version: `sklearn-1.8.0`
  - benchmarks completed: 7 / 7
  - failures: 0
- Tier-1 eval64: `.artifacts/primordia-tier1-64-linux-20260427`
  - runtime: `numpy-fallback`
  - framework version: `sklearn-1.8.0`
  - benchmarks completed: 8 / 8
  - failures: 0

### Compare/fair-matrix smoke lane
- Workspace: `.artifacts/fair-matrix-smoke-core-20260427`
- Report: `.artifacts/fair-matrix-smoke-core-20260427/reports/tier1_core_smoke_eval16_seed42/fair_matrix_summary.md`
- Acceptance JSON: `.artifacts/fair-matrix-smoke-core-20260427/reports/tier1_core_smoke_eval16_seed42/lane_acceptance.json`
- Observed result:
  - `artifact_completeness_ok: true`
  - `fairness_ok: true`
  - `task_coverage_ok: true`
  - `budget_consistency_ok: true`
  - `seed_consistency_ok: true`
  - `budget_accounting_ok: false`
  - `operating_state: reference-only`

## Acceptance interpretation

The fair-matrix smoke run is good enough to confirm that Primordia stays
compare-compatible on the shared lane and that its own budget/seed surfaces are
not drifting.

The lane does **not** rise above `reference-only` because the other participating
systems in this run still report incomplete budget-accounting metadata:
- Prism: missing `actual_evaluations` and `evaluation_semantics`
- Stratograph: missing `actual_evaluations` and `evaluation_semantics`
- Topograph: missing `actual_evaluations` and `evaluation_semantics`

That acceptance note is a shared-substrate limitation, not a new Primordia-only
branch failure.

## Explicit scope narrowing

The branch closeout should use the following language:

- Primordia is benchmark-complete and compare-compatible on the official smoke
  and tier-1 fallback lanes.
- Primordia now preserves the workspace dependency fix and passes package
  validation from the dedicated branch worktree.
- Primordia has fair pairwise smoke-lane compare results on the shared
  substrate, but the overall lane remains `reference-only` until the other
  systems fill in missing budget-accounting metadata.
- This branch does **not** close MLX validation, higher-budget training-depth
  parity, or seeded-transfer parity claims. Those require separate evidence on
  an Apple Silicon workspace and a downstream seeding experiment.
