# EvoNN Budget Accounting Policy

## Purpose

This document makes the budget contract operational.

`BUDGET_CONTRACT.md` defines the vocabulary. This file defines how exported
runs should count work when they claim fair comparison.

## Core Rule

Every compare-visible run should distinguish:

- the **declared comparable budget**
- the **actual counted work**
- the **non-comparable or separately tracked work**

The point is not to pretend all systems spend compute identically. The point is
to make the accounting legible enough that fair-vs-nonfair judgments are
auditable.

## Required Accounting Semantics

### `evaluation_count`

Meaning:
- the declared comparable budget envelope for the run

Interpretation:
- this is the number Compare should treat as the run's target budget when
  checking parity against a pack or lane policy

### `actual_evaluations`

Meaning:
- the number of evaluation attempts charged against the run in this export

Interpretation:
- failed evaluations that consumed real budget should still count here
- this field may be lower than `evaluation_count` for partial runs

### `cached_evaluations`

Meaning:
- evaluations satisfied from cache or prior persisted results rather than fresh
  work in this run

Interpretation:
- these must not silently masquerade as fresh budget spend
- if they materially change fairness, the run should be flagged or separated in
  analysis

### `failed_evaluations`

Meaning:
- evaluation attempts that consumed budget but ended failed

Interpretation:
- failures are not free and must remain visible in accounting

### `invalid_evaluations`

Meaning:
- candidate attempts rejected before full evaluation

Interpretation:
- report them explicitly
- whether they count toward the fair budget depends on package policy, but they
  must never be hidden

### `resumed_from_run_id` and `resumed_evaluations`

Meaning:
- the current export continued from a prior run and inherited prior counted work

Interpretation:
- resumed work must remain auditable instead of looking like one fresh run

### `partial_run`

Meaning:
- the export stopped before its declared comparable budget was fully completed

Interpretation:
- partial runs can still be useful, but they should not look complete

### `evaluation_semantics`

Meaning:
- one short human-readable statement explaining what one counted evaluation
  means for the exporting system

Examples:
- "one evolved candidate trained/evaluated on the requested benchmark surface"
- "one contender fit/eval pass counted per contender in the fixed pool"
- "one promoted candidate reaching the compare-counted fidelity stage"

## Policy Rules

### Failed work

If a candidate consumed real budget and then failed, it should still count in
`actual_evaluations` and also be reflected in `failed_evaluations`.

### Cached work

If a result came from cache rather than fresh work, record it in
`cached_evaluations` and keep the fairness interpretation explicit.

### Resumed work

If a run resumes from prior state, keep the immediate prior run id and the
inherited counted work visible.

### Partial runs

If a run stopped early, set `partial_run: true` rather than exporting a shape
that looks fully budget-matched.

## Compare Guidance

Compare should prefer these behaviors:

- warn when budget-accounting fields are missing
- distinguish declared budget from actual counted work
- keep resumed, cached, and partial runs explainable in trend artifacts
- avoid calling a comparison fair when accounting semantics are missing or
  obviously mismatched

## Bottom Line

The core question is not only "how big was the budget?" but also:

> **what work was actually counted, what work was reused, and what work failed?**

If an export cannot answer that clearly, it should not support a strong fair
comparison claim.
