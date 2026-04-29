# Shared Substrate Debt Reduction Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Shared` and the remaining shared-substrate work from “foundation landed” into a cleaner, safer, lower-duplication umbrella layer without collapsing package identity.

**Architecture:** Keep `evonn_shared` focused on the minimum compare/export/report/fairness surface. Improve it along four axes: (1) remaining duplication removal, (2) semantic safety, (3) validation confidence, and (4) packaging/conventions for future consumers.

**Tech Stack:** Python, Pydantic, markdown/JSON helpers, uv workspace, sibling package exporters and reports.

**Scope note:** This is a dedicated shared-substrate branch plan intended to reduce substrate debt after the initial foundation landing. It is not a plan to merge the engines.

## Current Context

- The first shared-substrate foundation landed on `main`.
- Shared contracts, fairness helpers, JSON writers, and summary-core helpers now exist.
- The remaining risk is not lack of substrate. It is substrate drift, partial duplication, and semantic mismatch during future extraction moves.

## Verification Against Current Main (2026-04-27)

After reviewing the current tree, the broad "first extraction wave" in this plan is already materially complete.

Confirmed already shared on `main`:

- shared compare/export contracts in `EvoNN-Shared`
- shared fairness envelope + benchmark signature helpers reused by Compare, Contenders, Prism, Primordia, Stratograph, and Topograph
- shared JSON artifact writing reused across the export surface
- shared summary-core derivation reused across Compare portable smoke plus Prism, Primordia, Stratograph, and Topograph exporters
- Linux/macOS trust-layer CI split and package-specific shared check scripts already wired

That means this document should no longer be read as a broad substrate-extraction plan. The remaining useful work is a small residue of high-value convergence slices plus semantic-safety hardening.

## Recommended First Remaining Slices

These are the highest-leverage remaining shared-substrate moves that still preserve engine autonomy.

### Slice 1 — Canonical fairness metadata normalization

Why first:
- Compare still reconstructs fairness fallback metadata in multiple places during ingest and fair-matrix trend assembly.
- This is shared contract semantics, not engine behavior.

Current duplication to target:
- `EvoNN-Compare/src/evonn_compare/ingest/loader.py`
- `EvoNN-Compare/src/evonn_compare/comparison/fair_matrix.py`
- `EvoNN-Compare/src/evonn_compare/orchestration/fair_matrix.py`

Recommended extraction:
- add one `evonn_shared` helper that derives the canonical fairness metadata view from a manifest/payload, including legacy fallback defaults
- move Compare call sites onto that helper
- add semantic tests for missing-fairness legacy manifests and for lane metadata overlays

Why safe:
- it tightens shared compare semantics without moving any search/runtime behavior

### Slice 2 — Shared contract summary builder

Why next:
- exporters still build `summary.json` with repeated field assembly patterns even after sharing summary-core derivation
- Contenders still duplicates median/failure summarization logic locally instead of using the shared core

Current duplication to target:
- `EvoNN-Contenders/src/evonn_contenders/export/symbiosis.py`
- `EvoNN-Primordia/src/evonn_primordia/export/symbiosis.py`
- `EvoNN-Prism/src/prism/export/symbiosis.py`
- `EvoNN-Stratograph/src/stratograph/export/symbiosis.py`
- `EvoNN-Topograph/src/topograph/export/symbiosis.py`

Recommended extraction:
- introduce a minimal shared summary builder for the canonical cross-engine fields only
- keep engine-specific telemetry/extensions package-local and merged on top by each exporter
- convert Contenders first because it still owns the most duplicated summary math

Why safe:
- it shares compare-facing report semantics while leaving engine-specific summary sections local

### Slice 3 — Shared benchmark native-id fallback resolver

Why third:
- parity-pack fallback/native-id resolution logic is still duplicated across package-local benchmark/export modules
- this affects shared benchmark identity handling, not engine search logic

Current duplication to target:
- `EvoNN-Compare/src/evonn_compare/adapters/slots.py`
- `EvoNN-Contenders/src/evonn_contenders/benchmarks/parity.py`
- `EvoNN-Primordia/src/evonn_primordia/benchmarks/parity.py`
- `EvoNN-Stratograph/src/stratograph/benchmarks/parity.py`

Recommended extraction:
- move canonical fallback ordering for mixed old/new parity-pack native ids into a shared helper near `shared-benchmarks` / `evonn_shared.benchmarks`
- keep package-local benchmark registries and actual benchmark loading local
- add compatibility tests covering legacy `evonn`/`evonn2` aliases and newer `prism`/`topograph`/`stratograph`/`primordia` keys

Why safe:
- it reduces compare/export glue duplication without collapsing per-engine benchmark ownership

## Explicit Non-Targets For The Shared Branch

Keep these in the engine plans rather than pulling them into `EvoNN-Shared`:

- package-local runtime backend selectors and fallback trainers
- package-local benchmark fixes needed to make one engine complete on official lanes
- engine-local search heuristics, candidate scoring, and mutation policy
- package-local checkpoint/resume/inspect/report UX
- engine-specific telemetry or lineage fields above the shared contract minimum

## Desired End State

`EvoNN-Shared` should become:

- the clear canonical home for minimum compare/export/fairness helpers
- safer to evolve because semantics are better tested
- broad enough to remove repeated plumbing debt
- narrow enough to avoid pulling in engine/runtime logic

## Explicit Branch Targets

1. Finish the highest-value remaining duplication cleanup.
2. Strengthen semantic-compatibility tests around shared helpers.
3. Improve package conventions for new shared-helper moves.
4. Keep engine-specific behavior out of the shared layer.

## Phase 1 — Inventory and rank remaining substrate debt

**Objective:** Turn the remaining “cleanup” list into a concrete debt register.

**Work:**
1. Enumerate remaining duplicated compare/export/report helpers.
2. Rank them by value and risk.
3. Distinguish good shared candidates from package-local-by-design logic.

**Exit criteria:**
- remaining debt is named concretely rather than vaguely

## Phase 2 — Remove the highest-value remaining duplication

**Objective:** Pull the next best repeated plumbing into `evonn_shared`.

**Work:**
1. Finish remaining manifest/report/helper convergence where worthwhile.
2. Prefer small, semantics-preserving moves.
3. Update all relevant consumers in one slice per helper.

**Exit criteria:**
- one or more high-value duplication clusters disappear
- no engine-specific runtime logic is moved into Shared

## Phase 3 — Strengthen semantic safety

**Objective:** Make future shared-helper moves safer than the first wave.

**Work:**
1. Add direct unit tests around shared helpers.
2. Add consumer-side regression tests where field meaning matters.
3. Document compatibility expectations for future extractions.

**Exit criteria:**
- shared-helper regressions are caught by tests earlier

## Phase 4 — Improve packaging and usage conventions

**Objective:** Make `evonn_shared` easier to consume consistently.

**Work:**
1. Clarify what belongs in Shared and what does not.
2. Tighten docs and import conventions.
3. Reduce ad hoc re-export or compatibility shims where possible.

**Exit criteria:**
- contributors have a clearer rule set for future shared-substrate work

## Likely Execution Order

1. debt inventory
2. high-value duplication removal
3. semantic-safety strengthening
4. packaging/convention tightening

## Validation Matrix

- Shared tests
- focused consumer tests in Compare/Prism/Topograph/Contenders/Primordia/Stratograph as needed
- regression checks around summary/fairness/manifest semantics

## Merge-Back Strategy

1. debt inventory/docs
2. helper-extraction slices
3. semantic-test slices
4. convention/docs cleanup
