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

Interpretation rule:
- shared substrate means a shared minimum compare/export surface, not shared engine behavior semantics
- shared helpers must preserve existing field meaning unless a behavior change is explicit, tested, and intentionally adopted across consumers

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

Additional validated progress since the initial plan draft:
- portable Compare smoke exports now use shared contracts and shared fairness/signature helpers
- Topograph export uses shared fairness/signature helpers instead of local copies
- Primordia export uses shared fairness/signature helpers instead of local copies
- Stratograph export uses shared fairness/signature helpers instead of local copies
- shared JSON artifact writing helpers now exist in `EvoNN-Shared` and are used by:
  - Compare portable smoke
  - Contenders export
  - Primordia export
  - Stratograph export
- shared export summary-core logic now exists in `EvoNN-Shared`
- shared export summary-core logic is already used by:
  - Prism export summary
  - Topograph export summary
  - Primordia compare summary
  - Stratograph contract summary
  - Compare portable smoke summary
- Contenders regression smoke support was expanded so low-cost compare surfaces cover both classification and regression smoke cases better

Recent commits that materially advanced the plan:
- `96fa876` — `refactor: use shared contracts in portable smoke exports`
- `4b30fe0` — `refactor: reuse shared fairness helpers in topograph export`
- `f1231c6` — `refactor: reuse shared fairness helpers in primordia export`
- `34a0ff8` — `refactor: reuse shared fairness helpers in stratograph export`
- `4aea70f` — `refactor: add shared json artifact writers`
- `c2c0915` — `refactor: share export summary core logic`
- `bef3d9a` — `refactor: extend shared summary core across exporters`
- `5b0f947` — `fix shared summary median quality semantics`
- `fbaf5e5` — `feat: add named compare lane presets`

Important implementation lessons learned during execution:
- Linux and macOS verification paths differ in practice because Prism/Topograph MLX flows are macOS-specific; hook/test policy must keep Linux from pretending to validate MLX paths it cannot execute
- package-local exporters can converge on common compare/report surfaces without forcing the search runtimes themselves to merge
- the cleanest substrate progress so far has come from pulling duplicated manifest/fairness/summary/report plumbing into `evonn_shared` first, then rewiring call sites package by package
- shared summary/helper extraction is not automatically semantics-preserving; compatibility-focused regression tests need to land with each shared-helper move, not after it

Current status by milestone:
- Milestone 1: largely advanced, though some compatibility re-export cleanup may still remain
- Milestone 2: materially underway; shared manifest/fairness helpers, shared JSON writers, and shared summary-core logic now exist and are in active use
- Milestone 3: underway; Prism, Contenders, Topograph, Primordia, Stratograph, and Compare portable smoke have all been moved closer to the same compare/export substrate
- Milestone 4: partly advanced; named lane presets now exist in Compare, but the lane still needs explicit acceptance rules, repeatability checks, and routine execution
- Milestones 5-6: still ahead; these now depend more on reporting/trend surfaces and workflow default decisions than on basic substrate scaffolding

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

Current read:
- already substantially advanced for Compare / Contenders / Prism
- now also indirectly reinforced by portable smoke/export paths using the same shared contract surface
- remaining work is mostly cleanup, removal of transitional duplication, and sharper documentation of canonical ownership

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

Current read:
- shared fairness/signature helpers are already landed and reused across the relevant exporters
- shared JSON artifact writing helpers are landed
- shared summary-core logic is landed and already reused across multiple exporters
- remaining work is to decide how far to push toward a true canonical manifest builder versus keeping some per-package manifest assembly while sharing the derived/core logic
- the quality bar here must be semantic compatibility, not just reduced duplication

Implementation notes:
- a practical substrate line has emerged:
  - share fairness/signature helpers
  - share JSON artifact writing
  - share summary-core derivation
  - leave package-specific extra summary/report fields local unless commonality becomes obvious
- every shared-helper extraction in this area should ship with focused compatibility tests against at least one existing exporter/consumer path

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

Current read:
- Prism export is well onto the shared substrate
- Topograph export is materially closer to the shared substrate
- Contenders export is on shared contracts/fairness helpers and now benefits from the shared JSON writer path
- Compare portable smoke now participates in the same substrate direction, which is useful for low-cost compare lanes
- the remaining gap is less about basic export contract shape and more about compare-lane orchestration, adapter simplification, and elimination of leftover Compare-local special handling

Concrete leftovers to close:
- reduce or explicitly bless compatibility re-exports that still exist only to preserve older Compare import paths
- replace brittle Compare-side orchestration behavior that depends on matching human-readable contender cache-miss text
- document which Compare-side engine-specific branches are still intentional after the export surface converges

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

Additional detail captured from later discussion:
- this lane should be cheap enough to rerun often, not just theoretically portable
- it should likely include at least one classification and one regression case so smoke-grade comparisons do not overfit to one task family
- the lane should prefer artifacts that are valid on Linux fallback paths while remaining meaningful on macOS-native Prism/Topograph flows
- acceptance should include artifact completeness, fairness metadata completeness, and rerun-to-rerun sanity stability

Operational decision for now:
- the provisional smallest named lane is Compare preset `smoke`
- that preset currently means:
  - pack: `tier1_core_smoke`
  - seeds: `42`
  - budgets: `16`
  - purpose: lowest-cost repeatable contract/artifact validation lane

Phase-1 acceptance for this milestone:
- the `smoke` preset runs end-to-end from Compare in a single command path
- the lane covers both at least one classification case and at least one regression case
- each participating system emits `manifest.json`, `results.json`, and `summary.json`
- the compare lane emits a fair-matrix summary/report artifact without ad hoc post-processing
- rerunning the same preset must not produce contract/fairness incompatibility due to substrate drift
- any remaining incomparables must be attributable to declared engine/task limitations, not contract/schema breakage

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

Additional detail captured from later discussion:
- trendability should not depend on ad hoc parsing of package-local markdown only
- failures need to stay first-class in the reporting surface, not just success-only aggregates
- compare summaries should preserve enough shared structure that longitudinal reporting can be built on top of them rather than rebuilt per engine

Phase-1 acceptance for this milestone:
- trend inputs are derived from structured JSON artifacts, not markdown scraping
- the minimum longitudinal dimensions are fixed and documented:
  - engine
  - benchmark
  - pack
  - budget
  - seed
  - outcome status
  - metric direction/value
  - fairness metadata
- repeated `smoke` lane runs can be appended to a single trend dataset without per-engine parsers

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

Additional detail captured from later discussion:
- “Prism default” does not mean “Topograph neglected”
- the point is a default operating path plus a real challenger lane, not a de facto monoculture
- any remaining Compare-side engine-specific branches should be either eliminated or justified explicitly
- this milestone should follow a credible compare lane and reporting surface; it is not the proof point for the branch by itself

## Suggested KPIs

### Platform KPIs
- one named smallest lane preset exists in Compare and is stable: `smoke`
- `smoke` lane is executable through a single Compare entrypoint
- number of packages using shared contracts continues upward from the current baseline
- number of unexplained engine-specific branches left in Compare trends toward zero
- number of compatibility regressions introduced by shared-helper refactors trends toward zero
- repeated `smoke` runs stay contract-valid and artifact-complete

### Research KPIs
- Prism vs Topograph are comparable on the fixed `smoke` lane
- Contenders vs Prism are comparable on the fixed `smoke` lane where task support overlaps
- artifact completeness and trendability hold across repeated `smoke` reruns
- losses/wins can be explained with shared budget/fairness context instead of package-local interpretation only

## Working Rules

- use `uv` as the primary repo toolchain
- run pre-commit / pre-push before pushing and fix failures first
- push smaller verified slices rather than large speculative batches
- use commits as the work log
- do not store GitHub PATs persistently; rotate/revoke pasted tokens
- keep shared substrate changes focused on research plumbing, not search-core merger

## Recommended Execution Order From Here

1. finish the remaining shared-contract / compatibility cleanup
2. decide whether to introduce a fuller canonical manifest builder or keep manifest assembly package-local while the shared derived/core helpers grow
3. finish export convergence for the engines that matter to the low-cost compare lane first
4. treat Compare preset `smoke` as the provisional smallest lane and finish its acceptance criteria, comparability rules, and routine rerun path
5. wire that lane through Compare/reporting so it is genuinely repeatable and trend-ingestible
6. harden artifact/report structure for trend tracking
7. switch more docs/workflows to Prism as default while preserving Topograph as the first challenger

## Open Technical Leftovers

- `evonn_compare.contracts.models` is still a compatibility re-export layer and should either be retired or declared stable on purpose
- Compare-side contender orchestration still has at least one brittle string-matching path around baseline cache misses
- the boundary between canonical shared manifest derivation and package-local manifest assembly still needs an explicit decision
- Compare should document which remaining engine-specific branches are intentional after substrate convergence and which are debt

## Known Risks / Watchouts

- do not accidentally reintroduce Linux-only validation claims for MLX-backed Prism/Topograph paths that really need macOS
- do not over-share engine-specific telemetry or search-core semantics just because the export/report substrate is converging
- do not let the shared substrate stop at file-format cleanup; it needs to cash out in a real repeatable compare lane
- do not let the compare lane become classification-only if the long-term goal is cross-task evidence
- do not accept reduced duplication as success if the resulting shared helpers subtly change field semantics or comparability rules

## Success State

This plan succeeds when EvoNN has a real shared research substrate instead of a
partial scaffold, and when that substrate supports:
- a concrete, routinely rerunnable Compare lane beginning with the named `smoke` preset
- honest Prism-vs-Topograph and Contender-vs-Prism comparisons on a shared minimum fairness surface
- trend-capable structured artifacts that accumulate over time without per-engine parsing hacks
- a default-Prism operating path that is earned by comparable evidence rather than assumed in advance
