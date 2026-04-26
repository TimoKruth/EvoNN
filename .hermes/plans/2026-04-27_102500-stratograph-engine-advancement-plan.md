# Stratograph Engine Advancement Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Stratograph` from a valid hierarchical challenger into a stronger, more complete, more portable hierarchical search engine with clearer scientific value.

**Architecture:** Keep Stratograph hierarchy-first internally, but improve it along five axes: (1) backend portability quality, (2) benchmark completeness/correctness, (3) hierarchy-search quality, (4) runtime maturity/observability, and (5) compare usefulness on the shared substrate.

**Tech Stack:** Python, MLX, existing fallback runtime path, Pydantic, DuckDB, uv workspace, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

**Scope note:** This is a dedicated Stratograph branch plan intended to move Stratograph beyond “participating challenger” status and toward a stronger, better-justified hierarchical engine.

## Current Context

- Stratograph already has the right portability direction conceptually through `mlx` vs `numpy-fallback`.
- It is no longer a greenfield package.
- It still needs quality and maturity work if hierarchy is going to be a serious search claim rather than an architectural curiosity.

## Desired End State

Stratograph should become:

- benchmark-complete on official lanes
- stronger on shared compare lanes at `64/256/1000`
- more obviously hierarchy-driven in its strengths
- operationally mature enough for repeated challenger runs
- cleaner in its MLX/fallback split than it is today

## Explicit Branch Targets

1. Harden the existing portable runtime split.
2. Keep Stratograph benchmark-complete on `smoke` and `tier1_core`.
3. Improve hierarchy-search quality under fixed budgets.
4. Strengthen status, checkpoint, report, and export maturity.
5. Make Stratograph’s hierarchy-specific value clearer in artifacts and comparisons.

## Phase 1 — Harden backend portability

**Objective:** Turn the current portability direction into a deliberate, tested runtime boundary.

**Work:**
1. Audit the MLX/fallback split for drift or hidden MLX assumptions.
2. Make backend selection and limitations explicit in config/docs/artifacts.
3. Keep fallback good enough for correctness/CI and basic compare validation.
4. Preserve contract compatibility across backends.

**Exit criteria:**
- non-MLX smoke and package validation are reliable
- backend identity and limitations are explicit

## Phase 2 — Lock benchmark completeness and official-lane correctness

**Objective:** Keep Stratograph fully valid on named lanes before deeper search work.

**Work:**
1. Verify `smoke` and `tier1_core` at `64/256/1000`.
2. Fix benchmark-specific failures and export/accounting drift.
3. Capture a clear baseline scoreboard.

**Exit criteria:**
- official lanes are benchmark-complete
- no open compare/export caveat remains for official runs

## Phase 3 — Improve hierarchy-search quality

**Objective:** Make hierarchical search materially better, not just more elaborate.

**Work:**
1. Improve parent/elite policy and hierarchy-aware mutation pressure.
2. Improve cell-library reuse and hierarchy-selection policy.
3. Surface hierarchy-specific search rationale in artifacts.
4. Keep improvements within honest budget semantics.

**Exit criteria:**
- better outcomes on at least one named lane
- hierarchy-specific improvement signals are visible in artifacts

## Phase 4 — Improve runtime/training quality

**Objective:** Raise quality per evaluation without losing control of cost.

**Work:**
1. Improve training defaults and fragile modality paths.
2. Reduce failure patterns on harder benchmarks.
3. Track wall-clock and failure-rate effects explicitly.

**Exit criteria:**
- measurable quality gains without runaway cost

## Phase 5 — Strengthen runtime maturity

**Objective:** Make Stratograph easier to trust in longer runs.

**Work:**
1. Improve status/checkpoint/resume behavior.
2. Improve inspection/report surfaces.
3. Make failure modes and partial-run state easier to reason about.

**Exit criteria:**
- live and partial runs are easy to inspect
- restart behavior is safe and boring

## Phase 6 — Clarify hierarchy-specific evidence

**Objective:** Make it easier to tell what hierarchy buys you.

**Work:**
1. Improve hierarchy-focused reporting and summary slices.
2. Add trend-friendly fields for hierarchy-level evidence where useful.
3. Keep shared-helper usage semantics-tested.

**Exit criteria:**
- Stratograph’s hierarchy-specific story is easier to evaluate from artifacts

## Likely Execution Order

1. backend portability hardening
2. benchmark correctness
3. hierarchy-search quality
4. runtime/training quality
5. runtime maturity
6. hierarchy evidence

## Validation Matrix

- package tests
- smoke on MLX
- smoke on fallback backend
- `tier1_core` at `64/256/1000`
- Compare fair-matrix reruns on official lanes

## Merge-Back Strategy

1. portability hardening
2. correctness fixes
3. search-quality improvements
4. maturity/reporting slices
