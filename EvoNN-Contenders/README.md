# EvoNN-Contenders

Fixed contender zoo for EvoNN benchmark packs.

Purpose:

- run strong non-evolutionary baselines on current 38-benchmark pack
- keep contender sets editable in config
- export `manifest.json` + `results.json` for `evonn-compare`

Current contender groups:

- `tabular`: tree ensembles, MLPs, logistic, SVM, optional `xgboost` / `lightgbm` / `catboost`
- `synthetic`: strong subset of tabular contenders plus SVM / optional boosted trees
- `image`: flat-feature MLP / tree baselines plus optional `cnn_small`
- `language_modeling`: n-gram baselines plus optional `transformer_lm_tiny`

Quick start:

```bash
cd ../EvoNN-Contenders
uv run --extra boosted --extra torch evonn-contenders run --config configs/working_33_plus_5_lm_contenders.yaml
uv run evonn-contenders symbiosis export \
  --run-dir runs/working_33_plus_5_lm_contenders_seed42 \
  --pack-path ../EvoNN-Compare/manual_compare_runs/20260417_budget608_seed42_broad_w2_retry/packs/working_33_plus_5_lm_compare_broad_608_w2_r2_eval608.yaml
```

Compare example:

```bash
cd ../EvoNN-Compare
uv run evonn-compare compare \
  ../EvoNN-Topograph/runs/working_33_plus_5_lm_compare_broad_608_w2_r2_eval608_seed42 \
  ../EvoNN-Contenders/runs/working_33_plus_5_lm_contenders_seed42 \
  --pack ../EvoNN-Compare/manual_compare_runs/20260417_budget608_seed42_broad_w2_retry/packs/working_33_plus_5_lm_compare_broad_608_w2_r2_eval608.yaml
```

Notes:

- benchmark loading is reused from local `EvoNN-Stratograph`
- budget in exports means contender evaluations, not evolutionary evaluations
- adjust pools in YAML before larger sweeps
- optional contenders are skipped by default when required extras are not installed
- refreshed fair configs now assume `--extra boosted --extra torch` for full pool coverage
