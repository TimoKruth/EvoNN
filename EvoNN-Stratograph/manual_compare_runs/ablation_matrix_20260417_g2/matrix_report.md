# Stratograph Ablation Matrix

## Global Pairwise

| Left | Right | Wins | Losses | Ties |
|---|---|---:|---:|---:|
| two_level_shared | flat_macro | 8 | 29 | 7 |
| two_level_shared | two_level_unshared | 26 | 11 | 7 |
| two_level_shared | two_level_shared_no_clone | 5 | 4 | 35 |
| two_level_shared | two_level_shared_no_motif_bias | 21 | 13 | 10 |

## Packs

| Pack | Benchmarks | Shared>Unshared | Shared>Flat | Shared>NoClone | Shared>NoMotifBias |
|---|---:|---|---|---|---|
| tabular_local | 6 | True | True | False | False |
| image_smoke | 3 | True | False | True | True |
| lm_smoke | 3 | False | False | True | True |
| mixed_38_smoke | 38 | True | False | True | True |
