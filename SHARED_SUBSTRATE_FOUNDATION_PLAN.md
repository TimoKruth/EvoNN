# Shared Substrate Foundation Plan

## Purpose

This plan turns the `feat/shared-substrate-foundation` stream into an explicit
execution document.

The goal is not to merge EvoNN into one engine. The goal is to move recurring
research infrastructure into a shared substrate so that:
- comparisons get cheaper and more repeatable
- reports become more trustworthy
- package-specific search logic stays scientifically distinct
- Prism can become the default operating engine while still being compared
  fairly against real challengers such as Topograph

## North Star

EvoNN should support a repeatable, low-cost comparison lane where:
- **Prism** is the default engine
- **Topograph** is the first serious challenger
- Compare consumes shared contracts/helpers instead of package-local copies
- artifacts and reports are stable enough to track trends over time

## Scope Rules

Good shared-substrate targets:
- benchmark/parity-pack resolution helpers
- run manifests and summary builders
- compare-facing export contracts and validators
- fairness/budget/telemetry metadata models
- common report rendering helpers
- run storage/schema helpers
- recurring CLI support patterns

Keep package-local unless evidence says otherwise:
- genomes and candidate representations
- mutation/crossover logic
- compiler/runtime internals
- search coordinators
- engine-specific telemetry above the umbrella minimum contract

## Current Known Progress

Already established in the branch/history:
- shared package scaffold: `EvoNN-Shared`
- initial shared modules for contracts / benchmarks / budgets / runs
- root `uv` workspace updated
- `MONOREPO.md` updated for the shared substrate direction
- pre-commit + pre-push coverage added across Shared / Compare / Contenders
- Linux-safe hook behavior for Prism/Topograph MLX pytest paths
- Compare ingest/export/validation moved partly onto shared helpers/contracts
- Contenders exports validated through shared contracts
- Prism exports migrated onto shared contracts
- key commits mentioned in chat:
  - `1f12a1d`
  - `c94e0ad` — `feat: wire shared substrate checks and helpers`
  - `db384f3` — `refactor: validate contender exports with shared contracts`
  - `12138b0` — `refactor: reuse shared manifest helpers in compare exports`

## Milestones

### Milestone 1 — Shared contract baseline

Objective:
make `evonn_shared` the canonical home for minimum compare/export contracts.

Done when:
- Compare, Contenders, and Prism all import the same minimum contract models
- compatibility re-exports are reduced or clearly temporary
- shared contract tests exist and pass
- contract ownership is documented in `EvoNN-Shared`

Key tasks:
- consolidate compare/export schemas in `evonn_shared`
- remove remaining duplicate contract definitions where safe
- document canonical import paths
- keep package-local adapters only where needed for transition

### Milestone 2 — Shared manifest + summary substrate

Objective:
standardize run manifests, fairness metadata, and summary assembly.

Done when:
- Compare uses shared manifest/fairness helpers end-to-end for the common path
- summary generation follows one canonical shared pattern
- run outputs from different engines expose the same minimum compare surface

Key tasks:
- implement canonical run manifest builder
- implement shared summary/fairness helpers
- align exporter outputs around the same minimum schema
- add focused tests for manifest and summary compatibility

### Milestone 3 — Contenders/Prism/Topograph export convergence

Objective:
make engine outputs comparable without hiding real engine differences.

Done when:
- Prism export path cleanly targets shared contracts
- Topograph is integrated as the first serious challenger on the same surface
- Contenders export path is validated on the same substrate
- Compare consumes these outputs without engine-specific branching for the core path

Key tasks:
- finish Prism export cleanup onto shared contracts/helpers
- define and implement Topograph export adapter(s)
- close remaining Contenders substrate gaps
- remove avoidable Compare-side engine conditionals

### Milestone 4 — Small-budget compare lane

Objective:
establish one cheap, repeatable comparison lane that can run often.

Done when:
- one fixed benchmark pack/task set is agreed for the small lane
- one normalized budget profile exists for that lane
- Prism and Topograph both run on it through the same compare path
- reruns produce stable enough artifacts for sanity-check trend analysis

Key tasks:
- choose the smallest useful benchmark/task set
- define the budget contract for that lane
- wire lane execution into Compare/reporting flow
- record acceptance criteria for repeatability and artifact completeness

### Milestone 5 — Trend-capable reporting

Objective:
turn one-off outputs into longitudinal evidence.

Done when:
- reports/artifacts preserve enough structured metadata to compare runs over time
- failures and budget disclosures are visible and consistent
- run-to-run trend views are possible without ad hoc parsing

Key tasks:
- standardize artifact layout and metadata fields
- add common report rendering helpers where duplication exists
- define minimum trend dimensions: engine, benchmark, budget, seed/run identity, outcome
- ensure reports keep fairness context visible

### Milestone 6 — Prism default operating path

Objective:
make Prism the default engine without weakening fair comparison.

Done when:
- Prism is the default path in the normal workflow/docs where appropriate
- Topograph remains runnable as a first-class challenger
- Compare and reporting still treat both through the same fairness surface
- remaining engine-specific Compare branches are intentional and documented

Key tasks:
- update docs/CLIs/workflow defaults toward Prism
- keep Topograph lane healthy as the comparison baseline
- audit Compare branches and document the intentional survivors

## Suggested KPIs

### Platform KPIs
- time to run the smallest compare lane
- number of packages using shared contracts
- number of engine-specific branches left in Compare
- percent of compare/export logic sourced from `evonn_shared`
- repeatability of small-budget runs

### Research KPIs
- Prism vs Topograph comparable on fixed tasks
- Contenders vs Prism comparable on fixed tasks
- artifact completeness and trendability across reruns
- ability to explain losses/wins with shared budget/fairness context

## Working Rules

- use `uv` as the primary repo toolchain
- run pre-commit / pre-push before pushing and fix failures first
- push smaller verified slices rather than large speculative batches
- use commits as the work log
- do not store GitHub PATs persistently; rotate/revoke pasted tokens
- keep shared substrate changes focused on research plumbing, not search-core merger

## Recommended Execution Order From Here

1. finish shared contract cleanup
2. land canonical manifest + summary helpers
3. finish export convergence for Prism / Contenders / Topograph
4. define and automate the smallest compare lane
5. harden artifact/report structure for trend tracking
6. switch more docs/workflows to Prism as default

## Success State

This plan succeeds when EvoNN has a real shared research substrate instead of a
partial scaffold, and when that substrate supports an honest, repeatable,
small-budget Prism-vs-Topograph comparison lane with trend-capable artifacts.
