# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval154

## Decision Summary

- Operating State: `trusted-extended`
- Decision Grade: `decision-grade`
- Recommended Action: `use_for_cross_engine_decision`
- Repeatability Ready: `yes`
- Leading Systems: `Contenders`
- Winner Counts: `prism=1; topograph=1; stratograph=1; primordia=0; contenders=16`
- Ties: `3`
- Fair Budgets: `154`
- Reference-Only Budgets: `none`
- Blockers: `none`
- Notes: `none`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval154`
- Operating State: `trusted-extended`
- Expected Budget: `154`
- Expected Seed: `42`
- Artifact Completeness: `ok`
- Fairness Status: `ok`
- Task Coverage: `ok` (classification, regression)
- Budget Consistency: `ok`
- Seed Consistency: `ok`
- Budget Accounting: `ok`
- Core Completeness: `ok`
- Extended Completeness: `ok`
- System States: `contenders=benchmark-complete; primordia=benchmark-complete; prism=benchmark-complete; stratograph=benchmark-complete; topograph=benchmark-complete`
- Repeatability Ready: `yes`
- Acceptance Notes: `none`

## Fair Search-Budget Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Contenders Evals | Contenders Wins | Ties |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 154 | 42 | 22 | 154 | 1 | 154 | 1 | 154 | 1 | 154 | 0 | 154 | 16 | 3 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Contenders Evals | Contenders Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 154 | 42 | primordia vs contenders | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/primordia_vs_contenders.md |
| 154 | 42 | prism vs contenders | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/prism_vs_contenders.md |
| 154 | 42 | prism vs primordia | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/prism_vs_primordia.md |
| 154 | 42 | prism vs stratograph | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/prism_vs_stratograph.md |
| 154 | 42 | prism vs topograph | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/prism_vs_topograph.md |
| 154 | 42 | stratograph vs contenders | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/stratograph_vs_contenders.md |
| 154 | 42 | stratograph vs primordia | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/stratograph_vs_primordia.md |
| 154 | 42 | topograph vs contenders | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/topograph_vs_contenders.md |
| 154 | 42 | topograph vs primordia | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/topograph_vs_primordia.md |
| 154 | 42 | topograph vs stratograph | fair | 154 | 154 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval154_seed42/topograph_vs_stratograph.md |