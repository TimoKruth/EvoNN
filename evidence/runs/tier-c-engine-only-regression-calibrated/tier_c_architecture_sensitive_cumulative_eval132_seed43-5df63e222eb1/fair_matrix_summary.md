# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval132

## Decision Summary

- Operating State: `contract-fair`
- Decision Grade: `fair-but-not-trusted`
- Recommended Action: `compare_with_caveats`
- Repeatability Ready: `no`
- Leading Systems: `Prism`
- Winner Counts: `prism=8; topograph=5; stratograph=5; primordia=2`
- Ties: `2`
- Fair Budgets: `132`
- Reference-Only Budgets: `none`
- Blockers: `trusted-core unmet: contenders=not-participating`
- Notes: `trusted-core unmet: contenders=not-participating`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval132`
- Operating State: `contract-fair`
- Expected Budget: `132`
- Expected Seed: `43`
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
| 132 | 43 | 22 | 132 | 8 | 132 | 5 | 132 | 5 | 132 | 2 | 2 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 132 | 43 | prism vs primordia | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/prism_vs_primordia.md |
| 132 | 43 | prism vs stratograph | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/prism_vs_stratograph.md |
| 132 | 43 | prism vs topograph | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/prism_vs_topograph.md |
| 132 | 43 | stratograph vs primordia | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/stratograph_vs_primordia.md |
| 132 | 43 | topograph vs primordia | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/topograph_vs_primordia.md |
| 132 | 43 | topograph vs stratograph | fair | 132 | 132 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval132_seed43/topograph_vs_stratograph.md |