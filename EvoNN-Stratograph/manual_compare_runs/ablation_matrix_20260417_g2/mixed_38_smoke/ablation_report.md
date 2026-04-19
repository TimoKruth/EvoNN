# Stratograph Ablation Report

- Two-Level Shared Wins Overall: `False`
- Shared Better Than Unshared: `True`
- Shared Better Than Flat: `False`
- Shared Better Than No-Clone: `True`
- Shared Better Than No-Motif-Bias: `True`

## Variant Summary

| Variant | OK Benchmarks | Mean Params | Mean Reuse | Mean Macro Depth | Mean Cell Depth |
|---|---:|---:|---:|---:|---:|
| flat_macro | 34 | 19116.9 | 0.0000 | 3.85 | 2.00 |
| two_level_unshared | 34 | 26307.5 | 0.0000 | 3.59 | 3.73 |
| two_level_shared | 34 | 15947.6 | 0.4848 | 3.71 | 3.55 |
| two_level_shared_no_clone | 34 | 13581.8 | 0.4922 | 3.71 | 3.61 |
| two_level_shared_no_motif_bias | 34 | 9166.9 | 0.4755 | 3.59 | 3.59 |

## Pairwise

| Left | Right | Wins | Losses | Ties | Mean Delta | Mean Param Saving | Efficiency Wins |
|---|---|---:|---:|---:|---:|---:|---:|
| two_level_shared | flat_macro | 6 | 23 | 5 | -0.344290 | 3169.3 | 5 |
| two_level_shared | two_level_unshared | 21 | 7 | 6 | -0.106987 | 10359.9 | 18 |
| two_level_shared | two_level_shared_no_clone | 5 | 3 | 26 | -0.000605 | -2365.8 | 3 |
| two_level_shared | two_level_shared_no_motif_bias | 17 | 9 | 8 | 0.054345 | -6780.6 | 13 |

## Benchmark Winners

| Benchmark | Winner | Variants |
|---|---|---|
| adult | flat_macro | flat_macro=0.8173828125 params=896 reuse=0.00, two_level_unshared=0.8046875 params=2604 reuse=0.00, two_level_shared=0.8046875 params=1974 reuse=0.33, two_level_shared_no_clone=0.8046875 params=1974 reuse=0.33, two_level_shared_no_motif_bias=0.8037109375 params=1974 reuse=0.33 |
| bank_marketing | flat_macro | flat_macro=0.890625 params=1152 reuse=0.00, two_level_unshared=0.8828125 params=1968 reuse=0.00, two_level_shared=0.8876953125 params=5600 reuse=0.20, two_level_shared_no_clone=0.8876953125 params=5600 reuse=0.20, two_level_shared_no_motif_bias=0.888671875 params=880 reuse=0.50 |
| blobs_f2_c2 | flat_macro | flat_macro=1.0 params=320 reuse=0.00, two_level_unshared=1.0 params=536 reuse=0.00, two_level_shared=1.0 params=248 reuse=0.50, two_level_shared_no_clone=1.0 params=248 reuse=0.50, two_level_shared_no_motif_bias=1.0 params=248 reuse=0.50 |
| blood_transfusion | flat_macro | flat_macro=0.7733333333333333 params=320 reuse=0.00, two_level_unshared=0.7733333333333333 params=536 reuse=0.00, two_level_shared=0.7733333333333333 params=248 reuse=0.50, two_level_shared_no_clone=0.7733333333333333 params=248 reuse=0.50, two_level_shared_no_motif_bias=0.7666666666666667 params=248 reuse=0.50 |
| breast_cancer | flat_macro | flat_macro=0.9649122807017544 params=7680 reuse=0.00, two_level_unshared=0.9385964912280702 params=6630 reuse=0.00, two_level_shared=0.956140350877193 params=2910 reuse=0.50, two_level_shared_no_clone=0.9473684210526315 params=2910 reuse=0.50, two_level_shared_no_motif_bias=0.9210526315789473 params=2970 reuse=0.67 |
| circles | two_level_shared_no_clone | flat_macro=0.485 params=480 reuse=0.00, two_level_unshared=0.875 params=1288 reuse=0.00, two_level_shared=0.955 params=248 reuse=0.50, two_level_shared_no_clone=0.995 params=248 reuse=0.50, two_level_shared_no_motif_bias=0.895 params=696 reuse=0.33 |
| circles_n02_f3 | two_level_shared | flat_macro=0.48 params=320 reuse=0.00, two_level_unshared=0.92 params=1288 reuse=0.00, two_level_shared=1.0 params=248 reuse=0.50, two_level_shared_no_clone=1.0 params=248 reuse=0.50, two_level_shared_no_motif_bias=1.0 params=248 reuse=0.50 |
| credit_g | flat_macro | flat_macro=0.81 params=1760 reuse=0.00, two_level_unshared=0.72 params=7300 reuse=0.00, two_level_shared=0.735 params=1340 reuse=0.50, two_level_shared_no_clone=0.735 params=1340 reuse=0.50, two_level_shared_no_motif_bias=0.74 params=1340 reuse=0.50 |
| digits | flat_macro | flat_macro=0.9611111111111111 params=35840 reuse=0.00, two_level_unshared=0.7416666666666667 params=73280 reuse=0.00, two_level_shared=0.7944444444444444 params=23360 reuse=0.75, two_level_shared_no_clone=0.7944444444444444 params=23360 reuse=0.75, two_level_shared_no_motif_bias=0.6583333333333333 params=28736 reuse=0.25 |
| electricity | flat_macro | flat_macro=0.673828125 params=480 reuse=0.00, two_level_unshared=0.5986328125 params=536 reuse=0.00, two_level_shared=0.62890625 params=1824 reuse=0.17, two_level_shared_no_clone=0.62890625 params=1824 reuse=0.17, two_level_shared_no_motif_bias=0.6259765625 params=248 reuse=0.50 |
| fashion_mnist | flat_macro | flat_macro=0.775390625 params=68608 reuse=0.00, two_level_unshared=0.6279296875 params=118144 reuse=0.00, two_level_shared=0.7080078125 params=52096 reuse=0.50, two_level_shared_no_clone=0.7080078125 params=52096 reuse=0.50, two_level_shared_no_motif_bias=0.57421875 params=3808 reuse=0.50 |
| gas_sensor | two_level_unshared | flat_macro=0.94140625 params=135168 reuse=0.00, two_level_unshared=0.95703125 params=113472 reuse=0.00, two_level_shared=0.953125 params=168192 reuse=0.50, two_level_shared_no_clone=0.947265625 params=85632 reuse=0.75, two_level_shared_no_motif_bias=0.9375 params=85536 reuse=0.33 |
| gesture_phase | two_level_shared | flat_macro=0.462890625 params=9088 reuse=0.00, two_level_unshared=0.46875 params=22800 reuse=0.00, two_level_shared=0.48046875 params=3488 reuse=0.50, two_level_shared_no_clone=0.46484375 params=3488 reuse=0.50, two_level_shared_no_motif_bias=0.466796875 params=5920 reuse=0.75 |
| heart_disease | flat_macro | flat_macro=0.8703703703703703 params=780 reuse=0.00, two_level_unshared=0.7962962962962963 params=1326 reuse=0.00, two_level_shared=0.8518518518518519 params=598 reuse=0.50, two_level_shared_no_clone=0.8518518518518519 params=598 reuse=0.50, two_level_shared_no_motif_bias=0.7962962962962963 params=598 reuse=0.50 |
| ilpd | flat_macro | flat_macro=0.7094017094017094 params=480 reuse=0.00, two_level_unshared=0.7094017094017094 params=810 reuse=0.00, two_level_shared=0.7094017094017094 params=370 reuse=0.50, two_level_shared_no_clone=0.7094017094017094 params=370 reuse=0.50, two_level_shared_no_motif_bias=0.7094017094017094 params=370 reuse=0.50 |
| iris | flat_macro | flat_macro=1.0 params=336 reuse=0.00, two_level_unshared=0.9666666666666667 params=552 reuse=0.00, two_level_shared=1.0 params=264 reuse=0.50, two_level_shared_no_clone=1.0 params=264 reuse=0.50, two_level_shared_no_motif_bias=1.0 params=264 reuse=0.50 |
| jungle_chess | flat_macro | flat_macro=0.6787109375 params=504 reuse=0.00, two_level_unshared=0.6279296875 params=936 reuse=0.00, two_level_shared=0.6298828125 params=816 reuse=0.50, two_level_shared_no_clone=0.6298828125 params=816 reuse=0.50, two_level_shared_no_motif_bias=0.6337890625 params=264 reuse=0.50 |
| kc1 | flat_macro | flat_macro=0.8554502369668247 params=1728 reuse=0.00, two_level_unshared=0.8459715639810427 params=3318 reuse=0.00, two_level_shared=0.8459715639810427 params=2544 reuse=0.33, two_level_shared_no_clone=0.8459715639810427 params=2544 reuse=0.33, two_level_shared_no_motif_bias=0.8554502369668247 params=2544 reuse=0.33 |
| letter | flat_macro | flat_macro=0.6962890625 params=6240 reuse=0.00, two_level_unshared=0.3466796875 params=14638 reuse=0.00, two_level_shared=0.4892578125 params=3458 reuse=0.50, two_level_shared_no_clone=0.4892578125 params=3458 reuse=0.50, two_level_shared_no_motif_bias=0.408203125 params=3458 reuse=0.50 |
| mfeat_factors | flat_macro | flat_macro=0.9725 params=68608 reuse=0.00, two_level_unshared=0.8825 params=118144 reuse=0.00, two_level_shared=0.9325 params=3808 reuse=0.50, two_level_shared_no_clone=0.9325 params=3808 reuse=0.50, two_level_shared_no_motif_bias=0.895 params=3808 reuse=0.50 |
| mnist | flat_macro | flat_macro=0.8173828125 params=72512 reuse=0.00, two_level_unshared=0.6025390625 params=17424 reuse=0.00, two_level_shared=0.5595703125 params=8016 reuse=0.50, two_level_shared_no_clone=0.5595703125 params=8016 reuse=0.50, two_level_shared_no_motif_bias=0.744140625 params=8016 reuse=0.50 |
| moons | two_level_shared_no_motif_bias | flat_macro=0.91 params=320 reuse=0.00, two_level_unshared=0.955 params=536 reuse=0.00, two_level_shared=0.965 params=264 reuse=0.67, two_level_shared_no_clone=0.965 params=264 reuse=0.67, two_level_shared_no_motif_bias=0.98 params=248 reuse=0.50 |
| nomao | flat_macro | flat_macro=0.935546875 params=70848 reuse=0.00, two_level_unshared=0.900390625 params=159072 reuse=0.00, two_level_shared=0.908203125 params=7248 reuse=0.50, two_level_shared_no_clone=0.908203125 params=7248 reuse=0.50, two_level_shared_no_motif_bias=0.9072265625 params=9600 reuse=0.50 |
| ozone_level | flat_macro | flat_macro=0.9467455621301775 params=47808 reuse=0.00, two_level_unshared=0.9428007889546351 params=89928 reuse=0.00, two_level_shared=0.9388560157790927 params=26856 reuse=0.75, two_level_shared_no_clone=0.9388560157790927 params=26856 reuse=0.75, two_level_shared_no_motif_bias=0.9388560157790927 params=84816 reuse=0.17 |
| phoneme | two_level_unshared | flat_macro=0.73046875 params=320 reuse=0.00, two_level_unshared=0.7548828125 params=1352 reuse=0.00, two_level_shared=0.732421875 params=248 reuse=0.50, two_level_shared_no_clone=0.748046875 params=248 reuse=0.50, two_level_shared_no_motif_bias=0.73828125 params=696 reuse=0.33 |
| qsar_biodeg | flat_macro | flat_macro=0.8436018957345972 params=17304 reuse=0.00, two_level_unshared=0.7914691943127962 params=7344 reuse=0.00, two_level_shared=0.7677725118483413 params=8938 reuse=0.75, two_level_shared_no_clone=0.7677725118483413 params=8938 reuse=0.75, two_level_shared_no_motif_bias=0.7772511848341233 params=5544 reuse=0.33 |
| segment | flat_macro | flat_macro=0.9025974025974026 params=1786 reuse=0.00, two_level_unshared=0.7813852813852814 params=6558 reuse=0.00, two_level_shared=0.7835497835497836 params=1406 reuse=0.50, two_level_shared_no_clone=0.7835497835497836 params=1406 reuse=0.50, two_level_shared_no_motif_bias=0.8116883116883117 params=4158 reuse=0.50 |
| speed_dating | flat_macro | flat_macro=0.841796875 params=16896 reuse=0.00, two_level_unshared=0.841796875 params=29376 reuse=0.00, two_level_shared=0.8408203125 params=2544 reuse=0.33, two_level_shared_no_clone=0.8408203125 params=2544 reuse=0.33, two_level_shared_no_motif_bias=0.8408203125 params=1488 reuse=0.75 |
| steel_plates_fault | flat_macro | flat_macro=0.5449871465295629 params=3402 reuse=0.00, two_level_unshared=0.46786632390745503 params=5670 reuse=0.00, two_level_shared=0.519280205655527 params=2646 reuse=0.50, two_level_shared_no_clone=0.480719794344473 params=1584 reuse=0.50, two_level_shared_no_motif_bias=0.4832904884318766 params=2646 reuse=0.50 |
| tiny_lm_synthetic | flat_macro | flat_macro=248.8939556342799 params=65792 reuse=0.00, two_level_unshared=255.90277352487354 params=78272 reuse=0.00, two_level_shared=260.3717187084 params=203776 reuse=0.20, two_level_shared_no_clone=260.3717187084 params=203776 reuse=0.20, two_level_shared_no_motif_bias=261.73375688959635 params=42272 reuse=0.33 |
| tinystories_lm | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
| tinystories_lm_smoke | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
| vehicle | flat_macro | flat_macro=0.7823529411764706 params=4028 reuse=0.00, two_level_unshared=0.6529411764705882 params=2538 reuse=0.00, two_level_shared=0.7352941176470589 params=3778 reuse=0.50, two_level_shared_no_clone=0.7352941176470589 params=3778 reuse=0.50, two_level_shared_no_motif_bias=0.6470588235294118 params=3778 reuse=0.50 |
| wall_robot | two_level_shared_no_clone | flat_macro=0.642578125 params=6480 reuse=0.00, two_level_unshared=0.62109375 params=4392 reuse=0.00, two_level_shared=0.6298828125 params=1992 reuse=0.50, two_level_shared_no_clone=0.69140625 params=5176 reuse=0.50, two_level_shared_no_motif_bias=0.5859375 params=3384 reuse=0.75 |
| wikitext2_lm | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
| wikitext2_lm_smoke | none | flat_macro=failed, two_level_unshared=failed, two_level_shared=failed, two_level_shared_no_clone=failed, two_level_shared_no_motif_bias=failed |
| wilt | flat_macro | flat_macro=0.9462809917355371 params=480 reuse=0.00, two_level_unshared=0.9462809917355371 params=536 reuse=0.00, two_level_shared=0.9462809917355371 params=248 reuse=0.50, two_level_shared_no_clone=0.9462809917355371 params=248 reuse=0.50, two_level_shared_no_motif_bias=0.9462809917355371 params=248 reuse=0.50 |
| wine | flat_macro | flat_macro=1.0 params=1209 reuse=0.00, two_level_unshared=0.8611111111111112 params=1352 reuse=0.00, two_level_shared=0.9722222222222222 params=624 reuse=0.50, two_level_shared_no_clone=0.9444444444444444 params=624 reuse=0.50, two_level_shared_no_motif_bias=0.9722222222222222 params=624 reuse=0.50 |
