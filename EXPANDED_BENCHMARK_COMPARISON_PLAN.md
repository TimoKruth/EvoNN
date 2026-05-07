# Expanded EvoNN Benchmark Comparison Plan

## Summary

`main` is clean and tracking `origin/main`, so the previous work is pushed.

The comparison expansion should be staged, not a single large leap. The goal is to grow from the current trusted `tier1_core` lane into a benchmark ladder with increasing complexity and diversity, while requiring at least one reliable contender baseline per benchmark before that benchmark is allowed into recurring decision-grade comparisons.

Chosen defaults:

- Use a staged ladder, not one giant immediate suite.
- Required contender floor is reliable core only: sklearn-backed contenders for tabular/image and n-gram contenders for language modeling.
- Torch CNN/Transformer and external boosted-tree libraries remain optional enhanced pressure and must be reported explicitly when skipped.
- A benchmark is not promoted unless Compare can prove every included system emits complete artifacts and Contenders has at least one valid required contender result.

## Current Facts From Repo

- Existing trusted lane: `EvoNN-Compare/parity_packs/tier1_core.yaml`, 8 benchmarks.
- Existing Tier B pack: `shared-benchmarks/suites/parity/tier_b_core.yaml`, 4 benchmarks including tabular classification, tabular regression, image, and LM smoke.
- Shared catalog contains 152 benchmark YAMLs:
  - 113 classification
  - 34 regression
  - 5 language modeling
- Generated all-shared pack exists with 41 benchmarks:
  - 33 classification
  - 8 regression
  - 3 image benchmarks included within classification
- Compare currently requires total budget to be divisible by benchmark count.
- Contenders already support:
  - tabular/synthetic: gradient boosting, extra trees, random forest, MLP, logistic, SVM, optional XGBoost/LightGBM/CatBoost
  - image: MLP, logistic, trees, optional Torch CNN
  - language modeling: unigram/bigram n-gram, optional Torch Transformer

## Target Outcome

Create a benchmark ladder where EvoNN can answer:

- Does a search engine only work on tiny sanity tasks?
- Does it stay competitive on harder real local tasks?
- Does it generalize across modality, dimensionality, class count, noise, nonlinearity, and regression?
- Does it beat or tie a reasonable contender floor?
- Are wins meaningful, or only caused by weak baselines?

The expanded comparison surface should produce:

- fair-matrix summaries
- trend rows
- dashboard leaderboards
- contender-floor reports
- per-benchmark admission status
- explicit saturation/tie handling for metrics with known maximum possible score

## Public Interfaces And Schema Changes

### 1. Extend parity pack metadata

Update `EvoNN-Compare/src/evonn_compare/contracts/parity.py`.

Add optional fields to `ParityBenchmark`:

```python
benchmark_group: Literal["tabular", "synthetic", "image", "language_modeling"] | None
domain: str | None
difficulty: Literal["smoke", "core", "hard", "stress"] | None
runtime_class: Literal["ci", "local", "overnight", "weekend", "special"] | None
minimum_required_contenders: tuple[str, ...] = ()
enhanced_optional_contenders: tuple[str, ...] = ()
score_ceiling: float | None = None
tie_tolerance_abs: float = 1e-12
tie_tolerance_rel: float = 1e-12
admission_notes: str = ""
```

Add optional fields to `ParityPack`:

```python
ladder_tier: Literal["A", "B", "C", "D", "E"] | None
usage_classification: dict[str, Any] | None
promotion_requirements: dict[str, Any] | None
```

Compatibility:

- Existing packs remain valid.
- Missing metadata is allowed but packs without contender-floor metadata cannot be promoted beyond exploratory status.

### 2. Add contender-floor validation

Add Compare CLI:

```bash
uv run --package evonn-compare evonn-compare benchmark-audit --pack <pack> --output <path>
```

Required output files:

- `benchmark_audit.json`
- `benchmark_audit.md`

Audit must validate:

- benchmark resolves for Prism, Topograph, Stratograph, Primordia, and Contenders
- metric name and direction are declared
- total budget presets are divisible by benchmark count
- at least one required contender is configured and runnable per benchmark
- optional contenders are listed separately and can be skipped without failing required completeness
- score ceiling exists for bounded metrics like accuracy
- no benchmark is admitted into recurring lanes if required contender floor is missing

### 3. Add contender-floor report into fair-matrix artifacts

Every fair-matrix run should emit:

```text
reports/<case>/contender_floor_report.json
reports/<case>/contender_floor_report.md
```

Fields:

- `benchmark_id`
- `required_contenders`
- `required_contenders_ran`
- `required_contenders_ok`
- `best_required_contender`
- `best_required_contender_metric`
- `enhanced_optional_contenders`
- `enhanced_optional_skips`
- `floor_status`: `passed | weak | missing | failed`
- `admission_status`: `decision_grade | exploratory_only | blocked`

Fair-matrix lane state must downgrade if any benchmark has `floor_status=missing` or `failed`.

### 4. Add normalized possible-points semantics

For metrics with a natural ceiling:

- `accuracy`, `f1`, `auc`: ceiling is `1.0`
- if all systems and contenders are at ceiling within tolerance, report a ceiling tie
- ceiling ties should not be counted as evidence of one engine beating another

For metrics without a natural ceiling:

- `mse`, `rmse`, `mae`, `perplexity`, `loss`: no 100 percent point ceiling
- compare by metric direction and tolerance
- report best-observed tie only, not 100 percent possible points

Dashboard labels:

- `ceiling_tie`
- `best_observed_win`
- `best_observed_tie`
- `contender_floor_win`
- `engine_beats_floor`
- `engine_below_floor`

## Benchmark Ladder To Build

### Tier A: `tier_a_contract`

Purpose: fast smoke and contract validation.

Pack size: 8 benchmarks.

Source: keep current `tier1_core_smoke` semantics, but normalize metadata.

Benchmarks:

- `iris_classification`
- `wine_classification`
- `breast_cancer`
- `moons_classification`
- `digits_image`
- `diabetes_regression`
- `friedman1_regression`
- `credit_g_classification`

Budgets:

- 16
- 64

Admission:

- required contender floor must pass
- all five systems must produce L3 measurable output
- no benchmark may have hidden failure collapse

### Tier B: `tier_b_core_v2`

Purpose: default local research lane with real diversity.

Pack size: 12 benchmarks.

Benchmarks:

- `iris_classification`
- `breast_cancer`
- `credit_g_classification`
- `openml_bank_marketing`
- `openml_letter`
- `openml_gas_sensor`
- `moons_classification`
- `circles_classification`
- `digits_image`
- `fashionmnist_image`
- `diabetes_regression`
- `openml_cpu_activity`

Reasoning:

- keeps known anchors
- adds larger real tabular classification
- adds multiclass/tabular shape recognition
- adds harder image
- includes both easy and harder regression
- remains local/overnight feasible

Budgets:

- 96
- 384
- 768
- 1536

All are divisible by 12.

Required contenders:

- tabular classification: `hist_gb`, `extra_trees`, `linear_svc`
- synthetic classification: `hist_gb`, `extra_trees`, `svm_nystroem_rbf`
- image classification: `mlp_wide`, `extra_trees`
- regression: `hist_gb`, `extra_trees`, `ridge_or_linear`, `svr_or_nystroem_svr`

Implementation note:

If regression contender names are currently classification-shaped, formalize regression-specific registry specs rather than relying on implicit runtime branching.

### Tier C: `tier_c_architecture_sensitive`

Purpose: stress topology, nonlinearity, dimensionality, noise, multiclass scale, and modality.

Pack size: 16 benchmarks.

Benchmarks:

- `openml_gas_sensor`
- `openml_nomao`
- `openml_mfeat_factors`
- `openml_segment`
- `openml_phoneme`
- `openml_steel_plates_fault`
- `openml_qsar_biodeg`
- `openml_wilt`
- `mnist_image`
- `fashionmnist_image`
- `friedman1_regression`
- `openml_concrete`
- `openml_energy_efficiency`
- `openml_airfoil`
- `tinystories_lm_smoke`
- `wikitext2_lm_smoke`

Budgets:

- 128
- 512
- 1024
- 2048

All are divisible by 16.

Required contenders:

- tabular classification: `hist_gb`, `extra_trees`, `linear_svc`
- image: `mlp_wide`, `extra_trees`
- regression: `hist_gb`, `extra_trees`, `ridge_or_linear`
- LM: `bigram_lm_a01`, `bigram_lm_a20`

Optional enhanced contenders:

- image: `cnn_small`
- LM: `transformer_lm_tiny`
- tabular: `xgb_small`, `lgbm_small`, `catboost_small`

Promotion rule:

Tier C starts as exploratory only. It becomes decision-grade only after two clean runs at `512` and one clean run at `1024`.

### Tier D: `tier_d_broad_shared`

Purpose: broad benchmark diversity across the existing all-shared surface.

Pack size: start from the audited admitted subset of `all_shared.yaml`.
The implemented admitted subset contains 26 benchmarks promoted from clean Tier
B/C evidence; broaden it only as additional candidates pass `benchmark-audit`,
required contender-floor checks, and at least one clean special-lane proof run.

Budgets:

- 208
- 416
- 832
- 1664

All are divisible by the current 26-benchmark admitted subset and keep Prism's
mixed-family population factorable. If the admitted set grows, revise these
presets rather than carrying forward incompatible per-benchmark prime budgets.

Status: decision-grade as a separate Tier D leaderboard surface after three
clean `tier_d_local` proof runs at budget `208`.

Admission:

- each benchmark must pass `benchmark-audit`
- if any benchmark lacks required contender coverage, either add the contender or exclude the benchmark into `tier_d_blocked_candidates.yaml`
- no benchmark enters `tier_d_broad_shared` just because it exists in the catalog

Dashboard:

Tier D results must stay visible as a separate broad-lane leaderboard so they do
not distort Tier A/B/C trend claims. The three-run gate is satisfied for the
current 26-benchmark admitted pack.

## Contender Policy

Required floor:

- Must be dependency-light and reliable.
- Must run on the user's normal local environment without needing Torch, XGBoost, LightGBM, or CatBoost.
- Must provide at least one strong-ish baseline per benchmark, preferably more than one.

Enhanced floor:

- Optional dependencies can improve pressure but cannot be required for benchmark completeness.
- If optional contenders are missing, reports must say exactly what was skipped.
- Enhanced results should be displayed separately from required-floor results.

Per-benchmark pass condition:

- At least one required contender completes with `status=ok`.
- Best required contender metric is recorded.
- EvoNN engines are evaluated against the best required contender, not against a weak arbitrary first contender.

Contender strength labels:

- `floor`: required reliable contender set
- `enhanced`: optional stronger contender set
- `missing`: no valid required contender result
- `weak`: contender ran but only trivial/linear baseline was available for a benchmark that requires nonlinear pressure

## Implementation Steps

### Step 1: Add metadata support

Files:

- `EvoNN-Compare/src/evonn_compare/contracts/parity.py`
- `EvoNN-Shared/src/evonn_shared/benchmarks.py`

Work:

- add optional fields listed above
- keep existing YAML packs valid
- add unit tests for loading old and new pack shapes

Acceptance:

- all existing packs load unchanged
- new metadata round-trips through pydantic models
- no existing fair-matrix tests regress

### Step 2: Add benchmark audit CLI

Files:

- `EvoNN-Compare/src/evonn_compare/cli/...`
- `EvoNN-Compare/src/evonn_compare/orchestration/benchmark_resolution.py`
- `EvoNN-Compare/tests/...`

Work:

- create `benchmark-audit`
- resolve every benchmark against every system
- validate required contender floor
- validate budget divisibility for declared presets
- emit JSON and markdown

Acceptance:

- `benchmark-audit --pack tier1_core` passes
- `benchmark-audit --pack tier_b_core` identifies missing metadata but does not crash
- bad pack with unsupported benchmark fails with actionable details

### Step 3: Formalize regression contender specs

Files:

- `EvoNN-Contenders/src/evonn_contenders/contenders/registry.py`
- `EvoNN-Contenders/src/evonn_contenders/contenders/runtime.py`
- `EvoNN-Contenders/tests/test_contender_backends.py`

Work:

- add explicit regression contender specs
- ensure `hist_gb`, `extra_trees`, `random_forest`, `ridge`, `mlp`, and guarded `svr` are available for regression
- keep config backward compatible

Acceptance:

- tabular regression benchmarks have at least two valid required contender options
- diabetes, friedman1, cpu_activity, concrete, airfoil, energy_efficiency all run through Contenders
- failed/guarded SVMs are reported as guarded, not generic failures

### Step 4: Create new staged packs

Files:

- `shared-benchmarks/suites/parity/tier_a_contract.yaml`
- `shared-benchmarks/suites/parity/tier_b_core_v2.yaml`
- `shared-benchmarks/suites/parity/tier_c_architecture_sensitive.yaml`
- `shared-benchmarks/suites/parity/tier_d_broad_shared.yaml`
- `shared-benchmarks/suites/parity/tier_d_blocked_candidates.yaml`

Work:

- add explicit metadata for every benchmark
- add required and optional contenders per benchmark
- add score ceiling where valid
- add runtime class and promotion requirements

Acceptance:

- all packs pass YAML validation
- `benchmark-audit` passes for Tier A and Tier B
- Tier C may start exploratory but must list exact blockers if any
- Tier D must not include unaudited benchmarks

### Step 5: Add lane presets

Files:

- `EvoNN-Compare/src/evonn_compare/orchestration/lane_presets.py`
- `MONOREPO.md`
- `README.md`

Add presets:

- `tier_a_contract`: pack `tier_a_contract`, budget `64`
- `tier_b_local_v2`: pack `tier_b_core_v2`, budget `96`
- `tier_b_overnight_v2`: pack `tier_b_core_v2`, budget `384`
- `tier_b_weekend_v2`: pack `tier_b_core_v2`, budget `1536`
- `tier_c_local`: pack `tier_c_architecture_sensitive`, budget `128`
- `tier_c_overnight`: pack `tier_c_architecture_sensitive`, budget `512`
- `tier_d_broad`: pack `tier_d_broad_shared`, budget `416`

Acceptance:

- dry-run emits configs for every preset
- invalid budgets are rejected before execution
- docs explain when each lane should be used

### Step 6: Integrate contender-floor report into fair-matrix

Files:

- `EvoNN-Compare/src/evonn_compare/orchestration/fair_matrix.py`
- `EvoNN-Compare/src/evonn_compare/reporting/fair_matrix_dashboard.py`
- `EvoNN-Compare/src/evonn_compare/reporting/fair_matrix_md.py`

Work:

- compute best required contender per benchmark
- compute engine-vs-floor status
- compute ceiling ties
- add report artifacts
- expose in dashboard

Acceptance:

- dashboard can show:
  - full-system leaderboard
  - EvoNN-only leaderboard
  - contender-floor table
  - ceiling-tie table
  - benchmarks where EvoNN is below required floor
- fair-matrix state downgrades when required floor is missing

### Step 7: Run promotion sequence

Initial validation commands:

```bash
uv run --package evonn-compare evonn-compare benchmark-audit --pack tier_a_contract --output .tmp/benchmark-audit/tier_a_contract.md
uv run --package evonn-compare evonn-compare benchmark-audit --pack tier_b_core_v2 --output .tmp/benchmark-audit/tier_b_core_v2.md
uv run --package evonn-compare evonn-compare benchmark-audit --pack tier_c_architecture_sensitive --output .tmp/benchmark-audit/tier_c_architecture_sensitive.md
```

Tier A run:

```bash
uv run --package evonn-compare evonn-compare fair-matrix --preset tier_a_contract --workspace .tmp/comparison-ladder/tier_a --reset-workspace --open
```

Tier B validation:

```bash
uv run --package evonn-compare evonn-compare fair-matrix --preset tier_b_local_v2 --workspace .tmp/comparison-ladder/tier_b --reset-workspace --open
uv run --package evonn-compare evonn-compare fair-matrix --preset tier_b_overnight_v2 --workspace .tmp/comparison-ladder/tier_b --resume --open
```

Tier C exploratory:

```bash
uv run --package evonn-compare evonn-compare fair-matrix --preset tier_c_local --workspace .tmp/comparison-ladder/tier_c --reset-workspace --open
```

Promotion rule:

- Tier A promoted when one clean run passes.
- Tier B promoted when two clean runs pass at `96` and one clean run passes at `384`.
- Tier C remains exploratory until two clean `512` runs and one clean `1024` run exist.
- Tier D is decision-grade for its admitted 26-benchmark pack after three clean `208` runs; remaining blocked candidates require separate admission proof.

## Testing Plan

Unit tests:

- old parity packs still load
- new metadata validates
- bad contender-floor metadata fails audit
- required vs optional contender distinction works
- score-ceiling tie logic works
- regression contender registry resolves correct backends

Integration tests:

- `benchmark-audit --pack tier1_core`
- `benchmark-audit --pack tier_b_core_v2`
- fair-matrix dry-run for all new presets
- contender-floor report generated for a small fixture
- dashboard renders contender floor and ceiling ties

Smoke tests:

- Tier A `64`
- Tier B `96`
- Tier C `128` dry-run first, then execute only after audit passes

Acceptance criteria:

- no benchmark can silently enter decision-grade comparison without contender floor
- no optional dependency skip fails required completeness
- all missing optional dependencies are visible
- all engines and contenders emit L3 measurable artifacts for promoted packs
- dashboard clearly separates engine wins from ceiling ties and contender-floor failures

## Risks And Mitigations

Risk: Tier C/D packs expose unsupported benchmark IDs across engines.

Mitigation: `benchmark-audit` blocks promotion and writes exact unsupported benchmark/system pairs.

Risk: Image and LM contender floors may be too weak without Torch.

Mitigation: Required reliable floor remains honest, but reports label enhanced Torch pressure as missing when unavailable. Claims against image/LM should distinguish `required-floor` from `enhanced-floor`.

Risk: Large packs become too slow for routine use.

Mitigation: Keep Tier D special-only and keep Tier B as the main recurring expanded research lane.

Risk: Regression contender semantics stay implicit.

Mitigation: Formalize regression-specific contender specs before promoting expanded regression packs.

Risk: Accuracy benchmarks saturate and inflate win counts.

Mitigation: Add ceiling-tie handling and exclude ceiling ties from meaningful beat counts.

## Definition Of Done

This expansion is complete when:

- Tier A, Tier B, and Tier C packs exist with explicit metadata.
- `benchmark-audit` exists and blocks unaudited benchmark promotion.
- Every promoted benchmark has at least one required contender result.
- Fair-matrix emits contender-floor reports.
- Dashboard shows contender-floor status and ceiling ties.
- Tier B expanded comparison can run cleanly at `96`, `384`, and `1536`.
- Tier C can run at least exploratory at `128`.
- Tier D broad suite exists, includes 26 admitted benchmarks, and has three clean repeated `208` proof runs.
- Documentation explains which lanes are trusted, exploratory, overnight, or special-only.
