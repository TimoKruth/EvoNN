# Fair Matrix: tier_c_architecture_sensitive_cumulative_eval374

## Decision Summary

- Operating State: `trusted-extended`
- Decision Grade: `decision-grade`
- Recommended Action: `use_for_cross_engine_decision`
- Repeatability Ready: `yes`
- Leading Systems: `Contenders`
- Winner Counts: `prism=2; topograph=2; stratograph=0; primordia=1; contenders=14`
- Ties: `3`
- Fair Budgets: `374`
- Reference-Only Budgets: `none`
- Blockers: `none`
- Notes: `none`

## Lane Metadata

- Preset: `tier_c_local_cumulative`
- Pack: `tier_c_architecture_sensitive_cumulative_eval374`
- Operating State: `trusted-extended`
- Expected Budget: `374`
- Expected Seed: `44`
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
| 374 | 44 | 22 | 374 | 2 | 374 | 2 | 374 | 0 | 374 | 1 | 374 | 14 | 3 |

## Reference Baseline Results

| Budget | Seed | Benchmarks | Prism Evals | Prism Wins | Topograph Evals | Topograph Wins | Stratograph Evals | Stratograph Wins | Primordia Evals | Primordia Wins | Contenders Evals | Contenders Wins | Ties | Note |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
_None_

## Parity/Validity Check

| Budget | Seed | Pair | Status | Left Evals | Right Evals | Left Policy | Right Policy | Data Sig Match | Reason | Report |
|---:|---:|---|---|---:|---:|---|---|---|---|---|
| 374 | 44 | primordia vs contenders | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/primordia_vs_contenders.md |
| 374 | 44 | prism vs contenders | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/prism_vs_contenders.md |
| 374 | 44 | prism vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/prism_vs_primordia.md |
| 374 | 44 | prism vs stratograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/prism_vs_stratograph.md |
| 374 | 44 | prism vs topograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/prism_vs_topograph.md |
| 374 | 44 | stratograph vs contenders | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/stratograph_vs_contenders.md |
| 374 | 44 | stratograph vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/stratograph_vs_primordia.md |
| 374 | 44 | topograph vs contenders | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/topograph_vs_contenders.md |
| 374 | 44 | topograph vs primordia | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/topograph_vs_primordia.md |
| 374 | 44 | topograph vs stratograph | fair | 374 | 374 | prototype_equal_budget | prototype_equal_budget | yes | --- | /Users/timokruth/Projekte/Evo Neural Nets/.tmp/tier-c-floor-gap-large-slice/reports/tier_c_architecture_sensitive_cumulative_eval374_seed44/topograph_vs_stratograph.md |