# Contenders Floor Hardening Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Contenders` into a stronger, more portable, more auditable baseline floor so that wins against EvoNN systems carry more scientific weight.

**Architecture:** Keep Contenders a baseline package rather than an evolutionary engine, but improve it along four axes: (1) runtime/registry structure, (2) benchmark completeness/correctness, (3) baseline strength, and (4) compare/export/budget honesty.

**Tech Stack:** Python, scikit-learn, optional torch, optional boosted extras, uv workspace, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

**Scope note:** This is a dedicated Contenders branch plan for baseline-floor quality, not just a quarter-limited reliability patch list.

## Current Context

- Contenders already gives the repo an external baseline floor.
- It is already more portable than the MLX-bound engines.
- It still has layering smell, runtime-branch complexity, and baseline-depth gaps.
- Its scientific value depends on both strength and honesty.

## Desired End State

Contenders should become:

- benchmark-complete on official shared lanes
- structurally cleaner to extend and reason about
- stronger on tabular/image/text baseline tasks
- explicit and trustworthy in budget/export semantics
- the baseline floor the engine packages have to beat honestly

## Explicit Branch Targets

1. Clean up runtime/registry layering and dispatch.
2. Keep Contenders benchmark-complete on `smoke` and `tier1_core`.
3. Strengthen the cheapest high-value baseline set first.
4. Preserve honest budget normalization and export semantics.
5. Keep Linux/macOS portability strong.

## Phase 1 — Clean up runtime and registry structure

**Objective:** Make Contenders easier to extend without fragile branching.

**Work:**
1. Separate registry metadata from builders and evaluators.
2. Make backend/task dispatch explicit.
3. Reduce string-coupled failure signaling where possible.
4. Keep configs backward-compatible where practical.

**Exit criteria:**
- adding a new contender no longer requires fragile edits across one giant path

## Phase 2 — Lock benchmark completeness and export honesty

**Objective:** Keep Contenders fully valid on the shared compare lanes.

**Work:**
1. Verify `smoke` and `tier1_core` at `64/256/1000`.
2. Fix benchmark-specific gaps, especially task-modality mismatches.
3. Keep materialize/run/export semantics explicit and resilient.
4. Preserve budget truth under cache, retry, and baseline reuse cases.

**Exit criteria:**
- official lanes are benchmark-complete
- no open export/accounting caveat remains

## Phase 3 — Strengthen the baseline floor cheaply

**Objective:** Raise the baseline floor in the highest-value, lowest-risk order.

**Work:**
1. Finish low-risk classical additions such as SVM-class baselines where missing.
2. Harden existing tree/MLP/logistic/regression coverage.
3. Add boosted extras only where operationally clean.
4. Defer bigger family jumps unless they help the core evidence story.

**Exit criteria:**
- contender floor is measurably stronger on official lanes
- baseline additions do not destabilize portability or accounting

## Phase 4 — Improve optional torch/image/text paths

**Objective:** Make optional deeper baselines more deliberate and less brittle.

**Work:**
1. Clarify CPU vs CUDA device behavior.
2. Tighten image/tensor preparation paths.
3. Improve LM contender robustness and packaging expectations.

**Exit criteria:**
- optional torch paths are easier to reason about and less failure-prone

## Phase 5 — Improve compare/report usefulness

**Objective:** Make contender evidence easier to use in routine comparisons.

**Work:**
1. Improve report clarity for baseline wins/losses/failure modes.
2. Keep shared-helper integration semantics-tested.
3. Make contender floor changes trend-visible.

**Exit criteria:**
- it is easier to tell whether engine wins are meaningful or just exploiting a weak baseline

## Likely Execution Order

1. runtime/registry cleanup
2. correctness and export honesty
3. low-risk baseline strengthening
4. optional torch/image/text hardening
5. compare/report usefulness

## Validation Matrix

- package tests
- smoke and `tier1_core` at `64/256/1000`
- materialize/run/export flows
- compare fair-matrix reruns on official lanes

## Merge-Back Strategy

1. runtime/registry cleanup
2. correctness/export fixes
3. baseline additions
4. report/trend slices
