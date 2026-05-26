# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval374

## Decision Summary

- Operating State: `contract-fair`
- Decision Grade: `fair-but-not-trusted`
- Recommended Action: `compare_with_caveats`
- Repeatability Ready: `no`
- Leading Systems: `Prism`
- Winner Counts: `prism=7; topograph=6; stratograph=2; primordia=3`
- Ties: `4`
- Fair Budgets: `374`
- Reference-Only Budgets: `none`
- Blockers: `trusted-core unmet: contenders=not-participating`
- Notes: `trusted-core unmet: contenders=not-participating`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval374`
- Operating State: `contract-fair`
- Expected Budget: `374`
- Expected Seed: `42`
- Artifact Completeness: `ok`
- Fairness Status: `ok`
- Task Coverage: `ok` (classification, regression)
- Budget Consistency: `ok`
- Seed Consistency: `ok`
- Budget Accounting: `ok`
- Core Completeness: `incomplete`
- Extended Completeness: `ok`
- System States: `primordia=benchmark-complete; prism=benchmark-complete; stratograph=benchmark-complete; topograph=benchmark-complete`
- Repeatability Ready: `no`
- Acceptance Notes: `trusted-core unmet: contenders=not-participating`

## Fair Search-Budget Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Ties |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 374 | 42 | 22 | 374 | 7 | 374 | 6 | 374 | 2 | 374 | 3 | 4 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 374 | 42 | prism vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/prism_vs_primordia.md |
| 374 | 42 | prism vs stratograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/prism_vs_stratograph.md |
| 374 | 42 | prism vs topograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/prism_vs_topograph.md |
| 374 | 42 | stratograph vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/stratograph_vs_primordia.md |
| 374 | 42 | topograph vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/topograph_vs_primordia.md |
| 374 | 42 | topograph vs stratograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-engine-only/reports/tier_c_architecture_sensitive_cumulative_eval374_seed42/topograph_vs_stratograph.md |