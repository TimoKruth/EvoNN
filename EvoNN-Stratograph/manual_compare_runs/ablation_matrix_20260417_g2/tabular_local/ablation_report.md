# Stratograph Ablation Report

- Two-Level Shared Wins Overall: `False`
- Shared Better Than Unshared: `True`
- Shared Better Than Flat: `True`
- Shared Better Than No-Clone: `False`
- Shared Better Than No-Motif-Bias: `False`

## Variant Summary

| Variant | OK Benchmarks | Mean Params | Mean Reuse | Mean Macro Depth | Mean Cell Depth |
|---|---:|---:|---:|---:|---:|
| flat_macro | 6 | 1724.2 | 0.0000 | 3.67 | 2.00 |
| two_level_unshared | 6 | 1913.2 | 0.0000 | 3.50 | 3.60 |
| two_level_shared | 6 | 831.7 | 0.4722 | 3.17 | 3.25 |
| two_level_shared_no_clone | 6 | 769.0 | 0.5000 | 3.00 | 3.17 |
| two_level_shared_no_motif_bias | 6 | 1426.7 | 0.4583 | 3.67 | 3.50 |

## Pairwise

| Left | Right | Wins | Losses | Ties | Mean Delta | Mean Param Saving | Efficiency Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| two_level_shared | flat_macro | 2 | 2 | 2 | 0.056150 | 892.5 | 1 |
| two_level_shared | two_level_unshared | 3 | 2 | 1 | 0.010443 | 1081.5 | 3 |
| two_level_shared | two_level_shared_no_clone | 0 | 1 | 5 | -0.003333 | -62.7 | 0 |
| two_level_shared | two_level_shared_no_motif_bias | 1 | 3 | 2 | -0.009001 | 595.0 | 1 |

## Benchmark Winners

| Benchmark | Winner | Variants |
|---|---|---|
| blobs_f2_c2 | flat_macro | flat_macro=1.0 params=320 reuse=0.00, two_level_unshared=1.0 params=536 reuse=0.00, two_level_shared=1.0 params=248 reuse=0.50, two_level_shared_no_clone=1.0 params=248 reuse=0.50, two_level_shared_no_motif_bias=1.0 params=248 reuse=0.50 |
| breast_cancer | flat_macro | flat_macro=0.9649122807017544 params=7680 reuse=0.00, two_level_unshared=0.9385964912280702 params=6630 reuse=0.00, two_level_shared=0.9473684210526315 params=2910 reuse=0.50, two_level_shared_no_clone=0.9473684210526315 params=2910 reuse=0.50, two_level_shared_no_motif_bias=0.9385964912280702 params=6392 reuse=0.25 |
| circles | two_level_unshared | flat_macro=0.485 params=480 reuse=0.00, two_level_unshared=0.875 params=1288 reuse=0.00, two_level_shared=0.855 params=696 reuse=0.33, two_level_shared_no_clone=0.875 params=320 reuse=0.50, two_level_shared_no_motif_bias=0.86 params=784 reuse=0.50 |
| iris | flat_macro | flat_macro=1.0 params=336 reuse=0.00, two_level_unshared=0.9666666666666667 params=552 reuse=0.00, two_level_shared=1.0 params=264 reuse=0.50, two_level_shared_no_clone=1.0 params=264 reuse=0.50, two_level_shared_no_motif_bias=1.0 params=264 reuse=0.50 |
| moons | two_level_shared_no_motif_bias | flat_macro=0.91 params=320 reuse=0.00, two_level_unshared=0.965 params=536 reuse=0.00, two_level_shared=0.95 params=248 reuse=0.50, two_level_shared_no_clone=0.95 params=248 reuse=0.50, two_level_shared_no_motif_bias=0.98 params=248 reuse=0.50 |
| wine | flat_macro | flat_macro=1.0 params=1209 reuse=0.00, two_level_unshared=0.8888888888888888 params=1937 reuse=0.00, two_level_shared=0.9444444444444444 params=624 reuse=0.50, two_level_shared_no_clone=0.9444444444444444 params=624 reuse=0.50, two_level_shared_no_motif_bias=0.9722222222222222 params=624 reuse=0.50 |
