# Branch Plans And Backlog

This directory holds branch-sized package and subsystem plans. Treat these files
as scoped backlog/reference material after the recurring-lane tranche has landed,
not as proof that a matching branch is currently open.

If a package README or older plan points somewhere else for future package
advancement work, this directory is the best starting point.

Use these files when opening a new issue-specific branch and merging work back in
slices.

## Execution Guardrails For Remaining Engine Advancement

Use the engine plans for package-local advancement only.

Belongs in `EvoNN-Shared` / `EvoNN-Compare` instead of engine branches:
- canonical fairness metadata normalization
- canonical cross-engine `summary.json` field assembly
- parity-pack native-id fallback ordering shared across packages
- fair-matrix, trend, dashboard, and repo-wide budget/accounting semantics

Belongs in the engine branch plans:
- package-local backend portability and runtime boundaries
- official-lane benchmark completeness fixes for that engine
- search-policy, training, scoring, and candidate-selection improvements
- checkpoint/resume/inspect/report surfaces that stay package-local
- engine-specific artifact extensions built on top of shared contracts

Merge rule:
- each engine merge-back should target one named slice from its plan and should
  not include a new shared-helper extraction unless the change is explicitly
  handed off to the Shared or Compare branch plans

## Branch-Sized Backlog

- `2026-04-26_211820-primordia-quality-parity-plan.md`
  - full Primordia engine-advancement plan
  - starts with backend portability, then benchmark completion, then search and
    quality work
- `2026-04-27_101500-prism-engine-advancement-plan.md`
  - Prism branch plan for backend portability, quality, runtime maturity, and
    stronger default-engine operation
- `2026-04-27_102000-topograph-engine-advancement-plan.md`
  - Topograph branch plan for Linux-capable runtime portability, search quality,
    cost discipline, and challenger maturity
- `2026-04-27_102500-stratograph-engine-advancement-plan.md`
  - Stratograph branch plan for turning the current challenger into a more
    complete, higher-quality hierarchical engine
- `2026-04-27_103000-contenders-floor-hardening-plan.md`
  - Contenders branch plan for a stronger, more portable, more auditable
    baseline floor
- `2026-04-27_103500-compare-trust-lane-maturation-plan.md`
  - Compare branch plan for fair-matrix trust, dashboards, trends, and
    higher-budget lane operation
- `2026-04-27_104000-shared-substrate-debt-reduction-plan.md`
  - Shared branch plan for substrate debt reduction without collapsing package
    identity

## Relationship To Root Plans

- `EVONN_90_DAY_PLAN.md`
  - remains the current quarter execution source of truth
- `README.md`, `MONOREPO.md`, `CONTRIBUTING.md`, and
  `RESEARCH_DECISION_GATE.md`
  - remain the current command and review-policy surface
- `ROADMAP.md`
  - remains the long-horizon umbrella sequence
- `SHARED_SUBSTRATE_FOUNDATION_PLAN.md`
  - remains the completed-foundation record plus debt list
- `BENCHMARK_EXTRACTION_PLAN.md`
  - remains the long-run benchmark/parity cleanup plan
- `CONTENDER_EXPANSION_PLAN.md`
  - remains the long-run contender breadth plan
- `SEEDING_LADDERS_IMPLEMENTATION_PLAN.md`
  - remains the long-run seeding-ladders plan

## Archived Package Bootstrap Plans

These older package-local plans are no longer active execution documents:

- `EvoNN-Primordia/IMPLEMENTATION_PLAN.md`
- `EvoNN-Stratograph/IMPLEMENTATION_PLAN.md`

They should be read only as historical/bootstrap records. Active advancement
work for those packages now lives in this directory.
