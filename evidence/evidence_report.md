# EvoNN Evidence Registry Report

- Generated At: `2026-05-26T11:37:15.768899+00:00`
- Registry: `/Users/timokruth/Projekte/Evo Neural Nets/evidence`
- Records: `62`
- Minimum Seeds For Decision Labels: `2`

## Decision Groups

| Label | Pack | Budget | Seeds | Decision | Leader | Blockers |
| --- | --- | ---: | --- | --- | --- | --- |
| tier-c-engine-only-regression-calibrated | tier_c_architecture_sensitive_cumulative_eval132 | 132 | 42, 43, 44 | gain | prism | none |
| tier-c-engine-only-regression-calibrated | tier_c_architecture_sensitive_cumulative_eval528 | 528 | 42, 43, 44 | gain | topograph | none |
| tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 42, 43, 44 | gain | contenders | none |
| tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 42, 43, 44 | gain | contenders | none |
| tier-c-full-midbudget-3seed | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 42, 43, 44 | gain | contenders | none |
| tier-c-full-midbudget-3seed | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 42, 43, 44 | gain | contenders | none |
| tier-c-full-midbudget-3seed | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 42, 43, 44 | gain | contenders | none |
| tier-c-full-midbudget-3seed | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 42, 43, 44 | gain | contenders | none |
| tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 42, 43, 44 | gain | prism | none |
| tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 42, 43, 44 | gain | prism | none |
| tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 42, 43, 44 | gain | prism | none |
| tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 42, 43, 44 | no_material_change | prism | none |
| tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 42, 43, 44 | gain | contenders | none |
| tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 42, 43, 44 | gain | contenders | none |
| tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 42, 43 | gain | contenders | none |

## Before/After Comparisons

| Comparison | Pack | Budget | Aggregate Delta | Decision |
| --- | --- | ---: | ---: | --- |
| tier-c-full-midbudget-3seed -> tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 0.1 | likely_gain |
| tier-c-floor-gap-after -> tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 0.0 | likely_gain |
| tier-c-floor-gap-large-slice -> tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval154 | 154 | 4.25 | likely_gain |
| tier-c-runtime-diagnostic-engine-only -> tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval154 | 154 | -4.166667 | regression |
| tier-c-full-midbudget-3seed -> tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 0.033333 | likely_gain |
| tier-c-floor-gap-after -> tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 0.0 | likely_gain |
| tier-c-floor-gap-large-slice -> tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval264 | 264 | 3.875 | likely_gain |
| tier-c-runtime-diagnostic-engine-only -> tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval264 | 264 | -4.083334 | regression |
| tier-c-full-midbudget-3seed -> tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval374 | 374 | -0.033333 | likely_gain |
| tier-c-floor-gap-after -> tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 0.033333 | likely_gain |
| tier-c-floor-gap-large-slice -> tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval374 | 374 | 4.0 | likely_gain |
| tier-c-runtime-diagnostic-engine-only -> tier-c-runtime-diagnostic-with-contenders-partial | tier_c_architecture_sensitive_cumulative_eval374 | 374 | -4.0625 | regression |
| tier-c-full-midbudget-3seed -> tier-c-floor-gap-after | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 0.066667 | likely_gain |
| tier-c-floor-gap-after -> tier-c-floor-gap-large-slice | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 0.0 | likely_gain |
| tier-c-floor-gap-large-slice -> tier-c-runtime-diagnostic-engine-only | tier_c_architecture_sensitive_cumulative_eval484 | 484 | 3.625 | likely_gain |

## Engine Roles

| System | Role | Leader Groups | Family Leads |
| --- | --- | ---: | --- |
| prism | leader_candidate | 5 | synthetic |
| topograph | leader_candidate | 1 | none |
| stratograph | challenger | 0 | none |
| primordia | watch | 0 | none |
| contenders | leader_candidate | 15 | image-classification, synthetic-regression, tabular, tabular-regression |

## LM Flatline Diagnostics

| System | LM Rows | Unique Metric Values | Flatline Suspected |
| --- | ---: | ---: | --- |
| n/a | 0 | 0 | false |

## Runtime Performance

| System | Runs | Wall Seconds | Evals/sec | Sec/eval | Score/sec |
| --- | ---: | ---: | ---: | ---: | ---: |
| contenders | 12 | 616.788505 | 6.206341 | 0.161126 | 0.214012 |
| topograph | 30 | 6456.040000 | 1.492556 | 0.669992 | 0.100758 |
| prism | 30 | 7385.692748 | 1.304685 | 0.766469 | 0.084217 |
| primordia | 30 | 7890.610797 | 1.221198 | 0.818868 | 0.079271 |
| stratograph | 30 | 8419.180580 | 1.144529 | 0.873722 | 0.073047 |

## Transfer Evidence

- Seeded Trend Rows: `0`
- Seed Sources: `none`
- Proof States: `none`
- Native Transfer Claim Ready: `False`
- Native Transfer Cases: `0`
- Native Verdict Counts: `{}`
- Native Consensus: `{}`
- All Transfer Cases: `0`
- Portable Transfer Cases: `0`
- Portable Verdict Counts: `{}`
- Portable Consensus: `{}`

## Quality Diversity Evidence

- Descriptor Or Archive Rows: `0`
- Summary Archive Evidence Count: `0`
- Claim Ready: `False`

## Artifact Validation

- OK: `True`
- Issues: `none`
- Warnings: `none`
