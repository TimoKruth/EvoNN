# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval528

## Decision Summary

- Operating State: `contract-fair`
- Decision Grade: `fair-but-not-trusted`
- Recommended Action: `compare_with_caveats`
- Repeatability Ready: `no`
- Leading Systems: `Topograph`
- Winner Counts: `prism=2; topograph=6; stratograph=5; primordia=3`
- Ties: `6`
- Fair Budgets: `528`
- Reference-Only Budgets: `none`
- Blockers: `trusted-core unmet: contenders=not-participating`
- Notes: `trusted-core unmet: contenders=not-participating`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval528`
- Operating State: `contract-fair`
- Expected Budget: `528`
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
| 528 | 42 | 22 | 528 | 2 | 528 | 6 | 528 | 5 | 528 | 3 | 6 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 528 | 42 | prism vs primordia | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/prism_vs_primordia.md |
| 528 | 42 | prism vs stratograph | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/prism_vs_stratograph.md |
| 528 | 42 | prism vs topograph | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/prism_vs_topograph.md |
| 528 | 42 | stratograph vs primordia | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/stratograph_vs_primordia.md |
| 528 | 42 | topograph vs primordia | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/topograph_vs_primordia.md |
| 528 | 42 | topograph vs stratograph | fair | 528 | 528 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-engine-only-multiseed-multibudget/reports/tier_c_architecture_sensitive_cumulative_eval528_seed42/topograph_vs_stratograph.md |