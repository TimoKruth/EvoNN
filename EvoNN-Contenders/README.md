# EvoNN-Contenders

Fixed contender zoo for EvoNN benchmark packs.

Purpose:

- run strong non-evolutionary baselines on shared EvoNN benchmark packs
- keep contender sets editable in config
- export `manifest.json` + `results.json` for `evonn-compare`

Current contender groups:

- `tabular`: tree ensembles, MLPs, logistic, SVM, optional `xgboost` / `lightgbm` / `catboost`
- `synthetic`: strong subset of tabular contenders plus SVM / optional boosted trees
- `image`: flat-feature MLP / tree baselines plus optional `cnn_small`
- `language_modeling`: n-gram baselines plus optional `transformer_lm_tiny`

Quick start from the monorepo root:

```bash
uv run --package evonn-contenders evonn-contenders --help
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/smoke.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_smoke_seed42 \
  --pack-path EvoNN-Compare/parity_packs/tier1_core_smoke.yaml
```

Notes:

- benchmark loading is reused from local `EvoNN-Stratograph`
- budget in exports means contender evaluations, not evolutionary evaluations
- adjust pools in YAML before larger sweeps
- optional contenders are skipped by default when required extras are not installed
- refreshed fair configs now assume `--extra boosted --extra torch` for full pool coverage

Official lanes:

```bash
# smoke / eval16
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/smoke.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_smoke_seed42 \
  --pack-path EvoNN-Compare/parity_packs/tier1_core_smoke.yaml

# tier1_core / eval64
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/tier1_core_eval64.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_tier1_core_eval64_seed42 \
  --pack-path EvoNN-Compare/parity_packs/tier1_core.yaml

# tier1_core / eval256
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/tier1_core_eval256.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_tier1_core_eval256_seed42 \
  --pack-path EvoNN-Compare/parity_packs/tier1_core.yaml

# tier1_core / eval1000
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/tier1_core_eval1000.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_tier1_core_eval1000_seed42 \
  --pack-path EvoNN-Compare/parity_packs/tier1_core.yaml

# tier_b_core / eval256
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/tier_b_core_eval256.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_tier_b_core_eval256_seed42 \
  --pack-path shared-benchmarks/suites/parity/tier_b_core.yaml

# tier_b_core / eval1000
uv run --package evonn-contenders evonn-contenders run \
  --config EvoNN-Contenders/configs/official_lanes/tier_b_core_eval1000.yaml
uv run --package evonn-contenders evonn-contenders symbiosis export \
  --run-dir EvoNN-Contenders/runs/official_tier_b_core_eval1000_seed42 \
  --pack-path shared-benchmarks/suites/parity/tier_b_core.yaml
```

Policy note:

- official lanes use `benchmark_pack.pack_name` resolution rather than hard-coded benchmark lists
- missing optional boosted/torch contenders do not block benchmark-complete status by policy, but exports now record those skips so Compare can surface them in lane trust summaries
