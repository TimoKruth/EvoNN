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
