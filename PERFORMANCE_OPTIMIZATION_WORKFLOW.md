# Performance Optimization Workflow

Status: `EVO-46` performance-discipline epic closeout.

This document is the canonical completion artifact for the `EVO-46` performance
epic. It defines how optimization branches are planned, measured, accepted, or
scrapped, and it is the reference that performance PRs must cite.

Canonical review workflow for `perf/<system-or-layer>-<optimization-name>` branches.

## Required Inputs

Every optimization branch must preserve the same:

- pack set
- budget set
- seed set
- backend class
- trust-state target

Every optimization PR and child issue must link:

- baseline artifact path
- after-change artifact path
- `perf_rows.jsonl` path for the compared runs
- exact dashboard/history slices reviewed

Optimization work is not reviewable from anecdotal timing alone.

## Review Flow

1. Generate the planned baseline artifact bundle with `evonn-compare performance-baseline`.
2. Implement one optimization family only on the branch.
3. Rerun the same matrix and store the after-change artifacts beside the branch results.
4. Load the baseline and after-change evidence into the workspace review flow with `historical-baseline` and `workspace-report`.
5. Review the canonical dashboard/history surfaces before deciding:
   - `perf_dashboard.html`
   - `trends/fair_matrix_trends.md`
   - `fair_matrix_dashboard.html`
   - exact per-case `fair_matrix_summary.json` or `baseline_summary.md`

## Optimization Child Issue Template

Use this structure for every optimization child issue:

```md
## Optimization Scope

- Branch: `perf/<system-or-layer>-<optimization-name>`
- Optimization family: `<one isolated change family>`
- Baseline artifact path: `<repo path>`
- After-change artifact path: `<repo path when available>`
- Target packs / budgets / seeds: `<pack list> @ <budget list>, seeds <seed list>`
- Backend class: `<mlx_truth | linux_fallback | ...>`

## Guardrails

- Quality expectation: `<no regression | declared tolerance>`
- Trust-state requirement: `<same-or-better>`
- Budget-accounting requirement: `<same accounting semantics>`
- Failure-rate requirement: `<no increase | declared tolerance>`

## Review Surfaces

- Planned dashboard: `perf_dashboard.html`
- Comparison history flow: `historical-baseline` + `workspace-report`
- Exact dashboard slices to review: `<fill when evidence exists>`

## Outcome Recording

- Accepted: close issue as `done` and link the baseline path, after path, dashboard slices, and exact case/run IDs used for approval.
- Rejected for revision: keep issue open and record the missing evidence or violated guardrail before rerunning.
- Scrapped: close issue as `cancelled` and preserve the baseline path, after path, and explicit scrap reason.
```

## Performance PR Template Policy

The canonical PR template lives at `.github/pull_request_template.md`.

Optimization PRs must fill the performance evidence sections and include:

- baseline artifact path
- after-change artifact path
- quality verdict
- trust-state verdict
- budget-accounting verdict
- exact dashboard/history slices reviewed
- explicit branch outcome: `accepted`, `rejected-for-revision`, or `scrapped`

## Accept Or Scrap Checklist

Accept only when at least one of these is true:

- `>=15%` median wall-clock improvement with no quality or trust regression
- `>=20%` eval/sec improvement at two or more budgets
- `>=25%` memory reduction with no runtime regression
- same quality with materially fewer actual calculations and honest metadata
- improved high-budget completion rate without worse low-budget behavior

Scrap when any of these is true:

- the gain appears only at one seed
- the gain appears only at one tiny budget
- quality or rank regresses materially
- failures increase
- backend labels become ambiguous
- budget accounting becomes harder to explain

Reject for revision when:

- required artifact paths are missing
- baseline and after-change matrices do not match
- dashboard/history slices are cited vaguely
- the branch mixes multiple optimization families
- evidence exists but does not yet meet the acceptance threshold

## Branch Outcome Recording

- `accepted`
  - PR merges with linked evidence.
  - Child issue closes as `done`.
- `rejected-for-revision`
  - PR stays open or returns to review.
  - Child issue remains active until the missing evidence is produced or the branch is scrapped.
- `scrapped`
  - PR closes without merge.
  - Child issue closes as `cancelled`.
  - Keep the artifact links so the failed branch does not get rediscovered later as a fake win.
