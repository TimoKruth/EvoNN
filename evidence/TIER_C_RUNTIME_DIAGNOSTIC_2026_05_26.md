# Tier C Runtime Diagnostic - 2026-05-26

## Scope

- Pack: `tier_c_architecture_sensitive_cumulative`
- Budgets: `154`, `264`, `374`, `484`
- Seeds: `42`, `43`, `44`
- Engine-only workspace: `.tmp/tier-c-runtime-diagnostic-engine-only`
- With-contenders workspace: `.tmp/tier-c-runtime-diagnostic-with-contenders`
- Clean rerun attempts: `.tmp/tier-c-runtime-diagnostic-with-contenders-v2`,
  `.tmp/tier-c-runtime-diagnostic-with-contenders-v3`

## Result

The engine-only diagnostic completed all 12 slices with 0 benchmark failures.
The with-contenders diagnostic completed 8 of 12 slices before runtime became
the blocking evidence item. Clean reruns were stopped because even the first
low-budget Tier C cumulative slice showed unacceptable wall-clock behavior on
this machine.

This should be interpreted as incomplete with-contenders evidence, not as a
quality failure. The important finding is that Tier C now needs runtime gates in
addition to quality gates.

## Engine-Only Summary

| System | Score | Wall sec | Evals/sec | Sec/success | Score/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prism | 99.0 | 3416.4 | 1.120 | 0.892 | 0.02898 |
| Topograph | 78.5 | 2992.5 | 1.279 | 0.782 | 0.02623 |
| Stratograph | 58.5 | 3660.1 | 1.046 | 0.956 | 0.01598 |
| Primordia | 33.5 | 4409.8 | 0.868 | 1.152 | 0.00760 |

## Partial With-Contenders Summary

| System | Full-system score | Wall sec | Evals/sec | Sec/success |
| --- | ---: | ---: | ---: | ---: |
| Contenders | 131.5 | 444.5 | 4.504 | 0.222 |
| Prism | 21.5 | 6478.1 | 0.309 | 3.236 |
| Primordia | 13.0 | 2247.8 | 0.891 | 1.123 |
| Stratograph | 9.5 | 1910.6 | 1.048 | 0.954 |
| Topograph | 4.0 | 8281.3 | 0.242 | 4.137 |

## Interpretation

- Prism remains the best Tier C engine-only generalist in this cohort.
- Topograph is runtime-efficient in the engine-only cohort, but the partial
  with-contenders run showed poor wall-clock behavior and needs repeated proof
  before any broad scaling claim.
- Stratograph is complete and measurable, but hierarchy/motif pressure needs to
  convert into better quality-per-second.
- Primordia is still useful as a specialist and seed-source engine, but it is
  the broad Tier C runtime-control risk.
- Contenders remain the quality and runtime floor on the partial external-floor
  evidence. EvoNN quality wins should be reported with runtime cost until this
  gap narrows.

## Code Changes Triggered By This Diagnostic

- Compare now carries `train_seconds` through fair-matrix trend rows.
- The dashboard exposes per-system runtime totals, evaluations/sec,
  seconds/successful candidate, family runtime allocation, and quality/runtime
  cost.
- Evidence reports now include runtime performance summaries.
- Stratograph received profile-aware motif/search pressure and family-specific
  mutation schedules.
- Primordia received stricter runtime controls for high-slot and cumulative Tier
  C work, including epoch caps and architecture clamps.

## Next Proof Gate

Before another full 3-seed, 4-budget Tier C with-contenders run, run a
runtime-first proof:

1. One seed, one budget, with contenders.
2. Confirm the dashboard runtime table stays within the agreed local wall-clock
   envelope.
3. Add the remaining seeds.
4. Add the remaining budgets.

Do not promote wide Tier C with-contenders evidence unless both quality and
runtime gates pass.
