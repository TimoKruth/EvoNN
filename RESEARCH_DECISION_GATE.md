# EvoNN Research Decision Gate

## Purpose

This document turns EvoNN research artifacts into explicit advancement
decisions.

Use it for:

- engine-advancement pull requests
- branch claims such as "this improved Prism" or "this should become the new
  default"
- compare/report changes that alter how evidence should be interpreted

The rule is simple: **judge branch claims from linked artifacts, not chat
interpretation.**

## Default Evidence Surface

The default review surface is the fair-matrix workspace, in this order:

1. `trends/fair_matrix_trends.md`
2. `trends/fair_matrix_trends.json`
3. `fair_matrix_dashboard.html`
4. `fair_matrix_dashboard.json`
5. case-local `reports/<case>/fair_matrix_summary.md`
6. case-local `reports/<case>/fair_matrix_summary.json`

Do not treat one-off markdown summaries or ad hoc terminal output as the primary
decision surface when the workspace artifacts exist.

If you are comparing a branch against earlier evidence, import the baseline into
the live workspace with `historical-baseline` and review both the current and
baseline cohorts from the same trend dataset and dashboard.

## Required Evidence Bundle

Every engine-advancement PR must include all of the following:

- workspace path used for the review
- trend report path
- dashboard path
- comparison labels reviewed, such as `current-workspace` and
  `release-2026-04-01`
- exact case IDs reviewed
- exact run IDs for the changed engine and the main comparison engines
- exact dashboard slices reviewed
- pack, budget, and seed set used for the claim
- lane operating state, accounting state, and repeatability state
- one recommended decision category from the list below

When possible, use the canonical workspace paths printed by
`evonn-compare fair-matrix`, `workspace-report`, or `historical-baseline`
instead of paraphrasing them in prose.

## Decision Categories

Use exactly one primary category in the PR summary.

### `promote`

Use when the branch shows a credible improvement on the intended lane and does
not introduce a new trust or fairness caveat.

Typical signs:

- `tier1_core` improves or holds while the target engine meaningfully improves
- no new lane-accounting, fairness, or repeatability downgrade appears
- failures or missing results do not materially worsen
- the gain is stable across repeated seeds or repeated lane runs

### `regress`

Use when the branch makes the target engine or the shared lane meaningfully
worse, even if there are isolated wins elsewhere.

Typical signs:

- lower project-only standing on the intended lane
- weaker benchmark-family position where the branch thesis was supposed to help
- more failures, missing results, or operational drift
- a win that only exists because another engine failed

### `inconclusive`

Use when the evidence is mixed and there is no clean reason to advance or
revert immediately.

Typical signs:

- some benchmarks improve while others regress without a clear thesis fit
- leaderboard movement is too small to trust
- the branch changed behavior, but the effect is not directionally stable

### `needs more seeds`

Use when the branch claim depends on too little repeated evidence.

Typical signs:

- the conclusion relies on a single seed
- seed-by-seed snapshots disagree materially
- multi-seed deltas are too noisy to support a branch decision

This is different from `inconclusive`: the problem here is insufficient
repeated evidence, not merely a mixed result.

### `Tier B-only gain`

Use when the branch improves `tier_b_core` or another benchmark-ladder Tier B
pack, but that gain is not yet reproduced on the trusted daily lane
`tier1_core`.

This is a valid research signal, but it is **not** enough to claim routine
default-lane advancement.

### `Tier 1 regression`

Use when the branch regresses `tier1_core`, especially the trusted daily lane at
the same or higher budget, even if Tier B or ad hoc experiments improved.

This category blocks promotion. Treat it as higher priority than a Tier B win.

## Decision Precedence

If multiple labels seem plausible, apply them in this order:

1. `Tier 1 regression`
2. `needs more seeds`
3. `Tier B-only gain`
4. `regress`
5. `promote`
6. `inconclusive`

This keeps the decision gate conservative when trusted-lane regressions or weak
repeat evidence exist.

## Review Workflow

### 1. Refresh the workspace artifacts

For a live run:

```bash
uv run --package evonn-compare evonn-compare fair-matrix \
  --workspace .tmp/fair-matrix-review
```

To refresh artifacts without rerunning engines:

```bash
uv run --package evonn-compare evonn-compare workspace-report \
  .tmp/fair-matrix-review
```

### 2. Import the comparison baseline when the claim is branch-relative

```bash
uv run --package evonn-compare evonn-compare historical-baseline \
  .tmp/fair-matrix-review \
  /path/to/baseline/workspace \
  --label release-2026-04-01
```

### 3. Record exact identifiers

Capture:

- case IDs from workspace state or report directories
- run IDs from the summary JSON or trend rows
- comparison labels for current versus baseline evidence

If the PR cannot point to exact IDs, it is not ready for advancement review.

### 4. Check lane health before reading winners

Review:

- `Lane Health By Budget`
- lane operating state
- budget accounting state
- repeatability state

Do not promote a branch from a degraded trust state without explicitly calling
out the downgrade.

### 5. Read the dashboard slices in a fixed order

Always review these slices:

- `Overall Leaderboard: Projects Only`
- `Aggregate Evidence: Projects Only`
- `Per-Seed Aggregate Snapshots: Projects Only`
- `Engine Rank By Benchmark Family: Projects Only`
- `Benchmark Trend View`

Use the all-systems views when contender pressure matters for the claim.

### 6. Decide whether the claimed win matches the branch thesis

Ask:

- did the intended benchmark families improve?
- did failures or missing results increase?
- did the branch improve only Tier B, or also `tier1_core`?
- is the gain visible across repeated seeds, not only one run?

### 7. Write the decision summary in the PR

Use this block:

```md
## Decision Gate Summary

- Recommended category: `promote`
- Pack / budget / seeds: `tier1_core @ 64`, seeds `42, 43, 44`
- Comparison labels: `current-workspace` vs `release-2026-04-01`
- Lane state: `trusted-core`; accounting `ok`; repeatability `ready`
- Exact case IDs: `tier1_core_eval64_seed42`, `tier1_core_eval64_seed43`, `tier1_core_eval64_seed44`
- Exact run IDs: `prism-ab12cd34`, `topograph-ef56gh78`, `contenders-ij90kl12`
- Dashboard slices reviewed: `Projects Only -> Aggregate Evidence`, `Projects Only -> Engine Rank By Benchmark Family`, `Benchmark Trend View`
- Evidence links:
  - `trends/fair_matrix_trends.md`
  - `fair_matrix_dashboard.html`
  - `reports/<case>/fair_matrix_summary.json`
- Why:
  - branch improved the intended family surface
  - no new fairness/accounting caveat appeared
  - repeated seeds support the same direction
```

## Non-Negotiable Rules

- Do not claim promotion from Tier B evidence alone.
- Do not hide `tier1_core` regressions behind broader benchmark-pack wins.
- Do not cite a dashboard without naming the exact slice that was reviewed.
- Do not cite a summary without linking the exact case ID or run IDs.
- Do not merge engine-advancement PRs without a stated decision category.
