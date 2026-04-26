# Compare Trust-Lane Maturation Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Compare` from a fair-matrix orchestrator with a good emerging dashboard into the trusted evidence layer for recurring cross-engine decisions.

**Architecture:** Keep Compare focused on orchestration, fairness validation, dashboards, and trend artifacts rather than search. Improve it along four axes: (1) lane trust semantics, (2) higher-budget run reliability, (3) dashboard/trend usefulness, and (4) operator experience.

**Tech Stack:** Python, shared contracts/helpers, Typer, markdown/JSON/HTML reports, uv workspace, sibling package CLIs.

**Scope note:** This is a dedicated Compare branch plan. It is broader than “keep fair-matrix working” and aims to make Compare the routine decision surface for the whole repo.

## Current Context

- Compare already owns the shared fair-matrix flow.
- It already has trend artifacts and a dashboard.
- It still needs stronger higher-budget reliability, clearer lane semantics, and more decision-friendly summaries.

## Desired End State

Compare should become:

- the trusted evidence/orchestration layer for recurring runs
- robust at `64`, `256`, and `1000` on official lanes
- clearer about fair vs trusted-core vs trusted-extended operation
- easier to use from the CLI without asking for interpretation
- stronger as a trend and dashboard surface over time

## Explicit Branch Targets

1. Harden `fair-matrix` and named-lane execution at higher budgets.
2. Improve lane acceptance/state semantics and reporting.
3. Improve dashboards and leaderboard views.
4. Improve operator ergonomics for repeated reruns and inspection.

## Phase 1 — Harden higher-budget lane execution

**Objective:** Make official lanes boringly reliable at `64/256/1000`.

**Work:**
1. Reduce orchestration fragility across system stages.
2. Improve fallback/retry behavior where appropriate.
3. Tighten artifact-completeness and failure surfacing.
4. Keep lane acceptance artifacts explicit and auditable.

**Exit criteria:**
- repeated official reruns finish cleanly more often
- failures are explicit and actionable

## Phase 2 — Strengthen trust semantics and summaries

**Objective:** Make lane state obvious without manual interpretation.

**Work:**
1. Keep `contract-fair`, `trusted-core`, and `trusted-extended` semantics visible.
2. Improve summary output and dashboard wording so users do not need chat interpretation for basics.
3. Keep fairness, benchmark completeness, and failures clearly separated.

**Exit criteria:**
- a contributor can tell the lane state directly from generated artifacts

## Phase 3 — Improve dashboards and leaderboards

**Objective:** Make the dashboard the normal lookup surface for run results.

**Work:**
1. Expand leaderboard views.
2. Preserve both all-system and project-only benchmark-winner perspectives.
3. Improve drill-down from lane summary to benchmark-level winners/failures.
4. Keep browser-open ergonomics and regeneration flow smooth.

**Exit criteria:**
- the dashboard answers the common “who won / who failed / what changed” questions directly

## Phase 4 — Improve trend and history usefulness

**Objective:** Make longitudinal evidence more decision-friendly.

**Work:**
1. Improve trend grouping and filtering.
2. Surface budget-normalized movement and failure/fairness drift.
3. Keep append-only datasets clean across repeated reruns.

**Exit criteria:**
- it is easier to answer “did this improve anything?” from Compare outputs first

## Phase 5 — Improve operator ergonomics

**Objective:** Make recurring compare runs easier to launch and inspect.

**Work:**
1. Improve rerun helpers and presets.
2. Improve CLI guidance and output paths.
3. Reduce manual post-processing steps after routine runs.

**Exit criteria:**
- repeated lane operation becomes easier and less chat-dependent

## Likely Execution Order

1. higher-budget reliability
2. trust semantics and summaries
3. dashboards and leaderboards
4. trend usefulness
5. operator ergonomics

## Validation Matrix

- Compare tests
- repeated `smoke` and `tier1_core` reruns
- dashboard generation/regeneration
- trend-report generation

## Merge-Back Strategy

1. reliability fixes
2. summary/semantics improvements
3. dashboard/leaderboard slices
4. operator-ergonomics slices
