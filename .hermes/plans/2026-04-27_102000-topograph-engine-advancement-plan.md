# Topograph Engine Advancement Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Topograph` as the first serious Prism challenger by improving backend portability, search quality, budget honesty, and challenger-grade runtime maturity.

**Architecture:** Keep Topograph topology-first internally, but improve it along five axes: (1) backend portability, (2) benchmark completeness/correctness, (3) topology-search quality, (4) runtime maturity/observability, and (5) compare/fairness usefulness on the shared substrate.

**Tech Stack:** Python, MLX, fallback runtime path, Pydantic, DuckDB, uv workspace, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

**Scope note:** This is a dedicated Topograph branch plan. Its purpose is to make Topograph a stronger and more portable challenger, not just to keep it barely lane-compatible.

## Current Context

- Topograph is the first serious challenger to Prism on the current shared compare surface.
- It is already central to the repo’s comparative story.
- It is still MLX-bound today.
- Topology search and budget accounting quality matter because Topograph is the strongest non-default pressure on Prism.

## Desired End State

Topograph should become:

- runnable on MLX and a Linux-capable fallback path
- benchmark-complete on official lanes
- stronger and more stable at `64/256/1000`
- more obviously topology-first in its wins rather than just “another engine”
- operationally trustworthy enough for routine challenger reruns

## Explicit Branch Targets

1. Add a Linux-capable fallback backend without losing compare/export truth.
2. Keep Topograph benchmark-complete on `smoke` and `tier1_core`.
3. Improve topology-search quality under fixed budgets.
4. Harden budget semantics and reused-candidate accounting.
5. Strengthen operator trust, inspection, and repeatability.

## Primary Strategy

1. Backend portability first
2. Benchmark correctness and budget truth second
3. Search-quality and topology-quality improvements third
4. Runtime maturity fourth
5. Longer-horizon challenger evidence fifth

## Phase 1 — Add a Linux-capable fallback backend

**Objective:** Make Topograph executable off Apple Silicon for smoke, CI, and contract validation.

**Work:**
1. Introduce an explicit backend selector.
2. Keep MLX as the high-quality path.
3. Add a correctness-first fallback backend.
4. Preserve compare-compatible artifacts and explicit runtime metadata.

**Exit criteria:**
- non-MLX smoke runs work
- artifacts stay contract-compatible
- backend identity is explicit and honest

## Phase 2 — Lock benchmark completeness and budget truth

**Objective:** Ensure Topograph remains a fair challenger as budgets rise.

**Work:**
1. Verify `smoke` and `tier1_core` at `64/256/1000`.
2. Fix benchmark-specific failures first.
3. Keep reused-candidate and evaluation accounting explicit.
4. Strengthen export/report explanations of what counted.

**Exit criteria:**
- benchmark-complete on named lanes
- no open budget-accounting caveat for official runs

## Phase 3 — Improve topology-search quality

**Objective:** Make Topograph wins more frequent and more topology-driven.

**Work:**
1. Improve parent selection, retention, and mutation pressure.
2. Improve topology novelty vs exploit balance.
3. Surface why topologies won in artifacts.
4. Keep search honest under the shared budget envelope.

**Exit criteria:**
- quality improves on at least one named lane
- topology-specific search rationale is visible in artifacts

## Phase 4 — Improve training/runtime quality

**Objective:** Raise result quality per evaluation without breaking Topograph’s cost profile.

**Work:**
1. Tighten training defaults.
2. Improve modality-specific robustness.
3. Reduce fragile failure modes on regression/image/text tasks.
4. Track wall-clock effects explicitly.

**Exit criteria:**
- improvements are measurable
- official lanes remain benchmark-complete

## Phase 5 — Deepen runtime maturity

**Objective:** Make Topograph less fragile operationally.

**Work:**
1. Strengthen status/checkpoint/resume behavior.
2. Improve live inspection surfaces.
3. Make longer-run interruption recovery predictable.

**Exit criteria:**
- interrupted runs resume safely
- longer challenger runs are easier to monitor

## Phase 6 — Strengthen challenger evidence

**Objective:** Make Topograph easier to evaluate as a serious alternative to Prism.

**Work:**
1. Improve compare/report surfaces that explain where Topograph wins or loses.
2. Keep shared-helper integration semantics-tested.
3. Improve trend-facing data for challenger analysis.

**Exit criteria:**
- Topograph’s relative position over time is easier to read from trend artifacts

## Likely Execution Order

1. backend portability
2. correctness and accounting
3. topology-search improvements
4. training/runtime improvements
5. runtime maturity
6. challenger evidence

## Validation Matrix

- package tests
- smoke on MLX
- smoke on fallback backend
- `tier1_core` at `64/256/1000`
- Compare fair-matrix reruns on official lanes

## Merge-Back Strategy

1. backend portability
2. correctness/accounting fixes
3. search-quality improvements
4. runtime maturity slices
5. report/trend slices
