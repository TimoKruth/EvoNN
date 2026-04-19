# Stratograph Ablation Report

- Two-Level Shared Wins Overall: `False`
- Shared Better Than Unshared: `False`
- Shared Better Than Flat: `False`
- Shared Better Than No-Clone: `True`
- Shared Better Than No-Motif-Bias: `True`

## Variant Summary

| Variant | OK Benchmarks | Mean Params | Mean Reuse | Mean Macro Depth | Mean Cell Depth |
|---|---:|---:|---:|---:|---:|
| flat_macro | 1 | 65792.0 | 0.0000 | 3.00 | 2.00 |
| two_level_unshared | 1 | 78272.0 | 0.0000 | 3.00 | 3.50 |
| two_level_shared | 1 | 203776.0 | 0.2000 | 6.00 | 3.50 |
| two_level_shared_no_clone | 1 | 203776.0 | 0.2000 | 6.00 | 3.50 |
| two_level_shared_no_motif_bias | 1 | 42272.0 | 0.3333 | 4.00 | 4.50 |

## Pairwise

| Left | Right | Wins | Losses | Ties | Mean Delta | Mean Param Saving | Efficiency Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| two_level_shared | flat_macro | 0 | 1 | 0 | -11.477763 | -137984.0 | 0 |
| two_level_shared | two_level_unshared | 0 | 1 | 0 | -4.468945 | -125504.0 | 0 |
| two_level_shared | two_level_shared_no_clone | 0 | 0 | 1 | 0.000000 | 0.0 | 0 |
| two_level_shared | two_level_shared_no_motif_bias | 1 | 0 | 0 | 1.362038 | -161504.0 | 0 |

## Benchmark Winners

| Benchmark | Winner | Variants |
|---|---|---|
| tiny_lm_synthetic | flat_macro | flat_macro=248.8939556342799 params=65792 reuse=0.00, two_level_unshared=255.90277352487354 params=78272 reuse=0.00, two_level_shared=260.3717187084 params=203776 reuse=0.20, two_level_shared_no_clone=260.3717187084 params=203776 reuse=0.20, two_level_shared_no_motif_bias=261.73375688959635 params=42272 reuse=0.33 |
| tinystories_lm_smoke | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
| wikitext2_lm_smoke | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
