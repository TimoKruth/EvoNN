# Contender Expansion Plan

Source-first plan. This document is based on current code, not old docs.

## Current Source Truth

Current `EvoNN-Contenders` is a small fixed-baseline runner:

- dependencies are only `numpy`, `scikit-learn`, `openml`, `duckdb`, `pydantic`, `pyyaml`, `rich`, `typer` in [EvoNN-Contenders/pyproject.toml](EvoNN-Contenders/pyproject.toml)
- contender registry is hard-coded in [EvoNN-Contenders/src/evonn_contenders/contender_pool.py](EvoNN-Contenders/src/evonn_contenders/contender_pool.py)
- supported families today are:
  - tree ensembles / gradient boosting
  - MLP
  - logistic regression
  - unigram / bigram language models
- image datasets are currently loaded as flat feature vectors, not structured tensors, in [EvoNN-Contenders/src/evonn_contenders/benchmarks/registry.py](EvoNN-Contenders/src/evonn_contenders/benchmarks/registry.py)
- contender execution path is one generic loop in [EvoNN-Contenders/src/evonn_contenders/pipeline.py](EvoNN-Contenders/src/evonn_contenders/pipeline.py)
- language modeling path assumes tiny local models with `fit(...)` + `perplexity(...)` protocol in [EvoNN-Contenders/src/evonn_contenders/contender_pool.py](EvoNN-Contenders/src/evonn_contenders/contender_pool.py)

Implication:

- `SVM` is easy to add with current stack
- `XGBoost/LightGBM/CatBoost` need new optional deps
- `CNN` needs new tensor-aware image path
- `Transformer` needs new training/runtime path; current LM protocol is too narrow
- evolutionary contenders should be adapters to sibling EvoNN projects, not reimplemented here

## 90-day alignment

This plan stays valid long-term, but the next 90 days should optimize for the
trusted daily compare lane rather than for maximum family count.

Near-term contender priority order:

1. harden contender reliability, export semantics, and budget truth on the
   default compare lanes
2. clean up runtime/registry structure where it blocks reliable daily-lane use
3. add the cheapest high-value baseline expansion, starting with SVM-class
   additions
4. only then consider boosted extras if they improve the daily lane cleanly
5. defer CNN, transformer, and evolutionary-adapter expansion unless they are
   directly needed for the primary quarter claim

## Recommended Order

Best order:

1. `SVM`
2. `XGBoost/LightGBM/CatBoost`
3. `CNN`
4. `Transformer`
5. evolutionary contender adapters

Reason:

- first two expand classical baselines fast
- CNN gives immediate value on current image benchmarks
- transformer is bigger runtime/design jump
- evolutionary contenders need budget normalization and cross-project adapter work

## Phase 0: Refactor Before New Families

Do this first. Current single-file registry is already near edge.

### Goals

- separate contender metadata from contender builders
- support task-specific runtimes without giant `if` ladder
- keep old configs working

### Changes

1. Split `contender_pool.py` into:
   - `registry.py` for named specs
   - `builders/classical.py`
   - `builders/lm.py`
   - later `builders/cnn.py`, `builders/transformer.py`, `builders/evolutionary.py`
2. Extend `ContenderSpec` with fields like:
   - `backend`
   - `task_kind`
   - `supports_groups`
   - `optional_dependency`
   - `budget_mode`
3. Add runtime dispatch:
   - `evaluate_classifier_contender(...)`
   - `evaluate_lm_contender(...)`
   - later `evaluate_torch_classifier_contender(...)`
   - later `evaluate_transformer_lm_contender(...)`
4. Keep current config schema shape from [EvoNN-Contenders/src/evonn_contenders/config.py](EvoNN-Contenders/src/evonn_contenders/config.py), but allow more names in each pool.

## Phase 1: Add SVM

This is lowest-risk because `scikit-learn` already exists.

### Add

- `linear_svc`
- `rbf_svc`
- `poly_svc`
- `linear_svc_balanced`
- `svm_nystroem_rbf` for larger tabular sets

### Scope

- `tabular`: yes
- `synthetic`: yes
- `image`: maybe only `linear_svc` first
- `language_modeling`: no

### Guardrails

- many current datasets are medium/large; kernel SVM can explode
- add config limits:
  - `max_train_samples_for_kernel_svm`
  - `skip_if_num_rows_gt`
  - `skip_if_input_dim_gt`
- for big sets use `LinearSVC` or Nystroem approximation, not raw RBF SVC

### Deliverable

First safe slice:

- `linear_svc`
- `rbf_svc_small`
- `linear_svc_balanced`

## Phase 2: Add XGBoost / LightGBM / CatBoost

These are strong baselines for tabular data. They should be optional extras, not hard required deps.

### Dependency Plan

Add optional dependency groups in `pyproject.toml`:

- `boosted = ["xgboost>=...", "lightgbm>=...", "catboost>=..."]`

### Add

- `xgb_default`
- `xgb_small`
- `lgbm_default`
- `lgbm_small`
- `catboost_default`
- `catboost_small`

### Scope

- `tabular`: yes
- `synthetic`: yes
- `image`: no by default
- `language_modeling`: no

### Design Rules

- CPU-first defaults
- fixed seeds
- moderate tree counts
- silent logging
- hard fail with clear missing-dependency message if optional extra not installed

### Why after SVM

- stronger engineering cost than SVM
- best value mostly on tabular, where repo already has many benchmarks

## Phase 3: Add CNN

Current blocker: image datasets load as flat vectors. CNN should not train on flattened fake-image tensors unless shape metadata is explicit.

### Source Gaps To Fix First

1. Add image shape metadata to benchmark specs/catalog:
   - `image_height`
   - `image_width`
   - `image_channels`
2. Extend image loader in [EvoNN-Contenders/src/evonn_contenders/benchmarks/registry.py](EvoNN-Contenders/src/evonn_contenders/benchmarks/registry.py) to optionally return:
   - flat arrays for sklearn baselines
   - shaped tensors for CNN baselines
3. Add optional PyTorch dependency group:
   - `torch = ["torch>=..."]`

### Add

- `cnn_small`
- `cnn_medium`
- `cnn_regularized`

### Scope

- `image`: yes
- everything else: no

### Architecture Rules

- small conv stacks only
- no giant training loops
- fixed epoch budget
- early stopping on validation
- CPU-safe baseline first; GPU optional

### First Slice

Only support:

- `digits`
- `mnist`
- `fashion_mnist`

Do not start with image transformers. CNN first is simpler and more useful here.

## Phase 4: Add Transformer

Best first transformer target is language modeling, not image.

Reason:

- repo already has LM benchmarks
- current LM contenders are extremely weak n-gram baselines
- transformer gives meaningful frontier jump there

### Source Gaps To Fix First

1. Introduce torch-based LM runtime separate from `LanguageModel` protocol
2. Add config knobs for:
   - `max_steps`
   - `batch_size`
   - `learning_rate`
   - `device`
   - `context_length_override`
3. Ensure LM datasets expose vocab/context metadata cleanly from benchmark spec

### Add

- `transformer_lm_tiny`
- `transformer_lm_small`
- maybe `gru_lm_small` as intermediate sanity baseline

### Scope

- `language_modeling`: yes
- `image`: no in first pass
- `tabular`: no

### Delivery Rule

Do not start with a full general training framework. Start with one tiny decoder-only transformer that can train on current cached `.npz` LM sets.

## Phase 5: Add Evolutionary Contender Adapters

Do not rebuild Prism / Topograph / Stratograph logic inside `EvoNN-Contenders`.

Best path: adapter contenders that call existing sibling project entry points with fixed mini-budgets and read back result metrics.

### Why Adapter, Not Reimplementation

- current evolution logic already lives elsewhere
- reimplementing inside contenders would fork source truth
- adapter preserves one owner per system

### Add

- `prism_mini`
- `topograph_mini`
- `stratograph_mini`
- maybe later `hybrid_mini`

### Required Work

1. Define adapter contract:
   - benchmark id in
   - seed in
   - budget in
   - result JSON out
2. Normalize budgets across systems:
   - evaluation count
   - wall-clock cap
   - or both
3. Make failure reasons explicit:
   - missing project
   - env missing
   - run failed
   - timeout

### Scope

- only benchmarks actually supported by each sibling project
- configs must allow opt-in; these should not be default contenders

## Config Plan

Extend config with optional family-specific sections:

- `svm`
- `boosted_trees`
- `torch`
- `evolutionary`

Each section should carry caps and safety defaults, for example:

- sample caps
- timeout seconds
- epoch / step caps
- device preference
- allow_optional_missing

Keep old simple contender lists working.

## Testing Plan

Need tests before adding big families.

### Add

- registry resolution tests for every new contender name
- smoke eval tests per family
- missing optional dependency tests
- image shape tests for CNN path
- LM smoke tests for transformer path
- adapter tests with mocked result files for evolutionary contenders

### Keep Runtime Cheap

- use `iris`, `moons`, `digits`, `tiny_lm_synthetic`
- no heavy OpenML in unit tests

## Recommended Milestones

### Milestone 1

- refactor registry/runtime
- add `SVM`
- add tests

### Milestone 2

- add optional boosted-tree libraries
- add config + missing-dependency handling
- add tests

### Milestone 3

- add tensor-aware image path
- add `CNN`
- add tests

### Milestone 4

- add torch LM runtime
- add tiny transformer LM
- add tests

### Milestone 5

- add evolutionary adapters
- normalize budget contract
- add export/report fields for adapter provenance

## Final Recommendation

Best immediate implementation order:

1. refactor contender runtime
2. add `SVM` first
3. add `XGBoost/LightGBM/CatBoost`
4. then add `CNN`
5. then add LM transformer
6. only then add evolutionary adapters

If only one next step happens now, it should be `SVM`. Lowest cost, no new heavy dependency, strongest signal about whether contender zoo expansion is worth continuing.
