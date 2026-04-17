# Stratograph Ablation Report

- Two-Level Shared Wins Overall: `False`
- Shared Better Than Unshared: `True`
- Shared Better Than Flat: `False`
- Shared Better Than No-Clone: `True`
- Shared Better Than No-Motif-Bias: `True`

## Variant Summary

| Variant | OK Benchmarks | Mean Params | Mean Reuse | Mean Macro Depth | Mean Cell Depth |
|---|---:|---:|---:|---:|---:|
| flat_macro | 3 | 58986.7 | 0.0000 | 4.33 | 2.00 |
| two_level_unshared | 3 | 69616.0 | 0.0000 | 3.67 | 3.75 |
| two_level_shared | 3 | 27824.0 | 0.5833 | 3.67 | 3.67 |
| two_level_shared_no_clone | 3 | 27824.0 | 0.5833 | 3.67 | 3.67 |
| two_level_shared_no_motif_bias | 3 | 13520.0 | 0.4167 | 3.67 | 3.22 |

## Pairwise

| Left | Right | Wins | Losses | Ties | Mean Delta | Mean Param Saving | Efficiency Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| two_level_shared | flat_macro | 0 | 3 | 0 | -0.163954 | 31162.7 | 0 |
| two_level_shared | two_level_unshared | 2 | 1 | 0 | 0.029962 | 41792.0 | 2 |
| two_level_shared | two_level_shared_no_clone | 0 | 0 | 3 | 0.000000 | 0.0 | 0 |
| two_level_shared | two_level_shared_no_motif_bias | 2 | 1 | 0 | 0.028443 | -14304.0 | 1 |

## Benchmark Winners

| Benchmark | Winner | Variants |
|---|---|---|
| digits | flat_macro | flat_macro=0.9611111111111111 params=35840 reuse=0.00, two_level_unshared=0.7416666666666667 params=73280 reuse=0.00, two_level_shared=0.7944444444444444 params=23360 reuse=0.75, two_level_shared_no_clone=0.7944444444444444 params=23360 reuse=0.75, two_level_shared_no_motif_bias=0.6583333333333333 params=28736 reuse=0.25 |
| fashion_mnist | flat_macro | flat_macro=0.775390625 params=68608 reuse=0.00, two_level_unshared=0.6279296875 params=118144 reuse=0.00, two_level_shared=0.7080078125 params=52096 reuse=0.50, two_level_shared_no_clone=0.7080078125 params=52096 reuse=0.50, two_level_shared_no_motif_bias=0.57421875 params=3808 reuse=0.50 |
| mnist | flat_macro | flat_macro=0.8173828125 params=72512 reuse=0.00, two_level_unshared=0.6025390625 params=17424 reuse=0.00, two_level_shared=0.5595703125 params=8016 reuse=0.50, two_level_shared_no_clone=0.5595703125 params=8016 reuse=0.50, two_level_shared_no_motif_bias=0.744140625 params=8016 reuse=0.50 |
