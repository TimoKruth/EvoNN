# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval264

## Decision Summary

- Operating State: `trusted-extended`
- Decision Grade: `decision-grade`
- Recommended Action: `use_for_cross_engine_decision`
- Repeatability Ready: `yes`
- Leading Systems: `Contenders`
- Winner Counts: `prism=2; topograph=0; stratograph=1; primordia=1; contenders=16`
- Ties: `2`
- Fair Budgets: `264`
- Reference-Only Budgets: `none`
- Blockers: `none`
- Notes: `none`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval264`
- Operating State: `trusted-extended`
- Expected Budget: `264`
- Expected Seed: `43`
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
| 264 | 43 | 22 | 264 | 2 | 264 | 0 | 264 | 1 | 264 | 1 | 264 | 16 | 2 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Contenders Evals | Contenders Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 264 | 43 | primordia vs contenders | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/primordia_vs_contenders.md |
| 264 | 43 | prism vs contenders | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/prism_vs_contenders.md |
| 264 | 43 | prism vs primordia | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/prism_vs_primordia.md |
| 264 | 43 | prism vs stratograph | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/prism_vs_stratograph.md |
| 264 | 43 | prism vs topograph | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/prism_vs_topograph.md |
| 264 | 43 | stratograph vs contenders | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/stratograph_vs_contenders.md |
| 264 | 43 | stratograph vs primordia | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/stratograph_vs_primordia.md |
| 264 | 43 | topograph vs contenders | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/topograph_vs_contenders.md |
| 264 | 43 | topograph vs primordia | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/topograph_vs_primordia.md |
| 264 | 43 | topograph vs stratograph | fair | 264 | 264 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-runtime-diagnostic-with-contenders/reports/tier_c_architecture_sensitive_cumulative_eval264_seed43/topograph_vs_stratograph.md |