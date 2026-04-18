# Stratograph Research Notes

## 2026-04-17 Hierarchy Runtime Upgrade

Work completed:

- Prism-aligned `uv` project extras in `pyproject.toml`
- LM cache resolver aligned with shared-benchmark layout
- `warm-cache` CLI command added
- lightweight runtime maturity added:
  - stronger classifier head search
  - image-specific heads
  - trigram plus feature-bucket LM scorer
  - SGD head inheritance across generations
- task-aware motif priors
- clone mutation strengthened
- wider ablation matrix and sharper pack configs

## Commands Run

```bash
uv run stratograph warm-cache --config configs/ablation_matrix_smoke.yaml
uv run stratograph ablate-matrix --config configs/ablation_matrix_smoke.yaml --workspace manual_compare_runs/ablation_matrix_20260417_full_runtime
uv run stratograph ablate --config configs/ablation_image_hard.yaml --workspace manual_compare_runs/ablation_image_hard_20260417
uv run stratograph ablate --config configs/ablation_openml_structured.yaml --workspace manual_compare_runs/ablation_openml_structured_20260417
uv run stratograph ablate --config configs/ablation_lm_full.yaml --workspace manual_compare_runs/ablation_lm_full_20260417
uv run --extra dev pytest -q
```

## Main Findings

Full matrix:

- shared vs flat: `12` wins, `31` losses, `7` ties
- shared vs unshared: `28` wins, `15` losses, `7` ties
- shared vs no-clone: `7` wins, `7` losses, `36` ties
- shared vs no-motif-bias: `24` wins, `18` losses, `8` ties

Interpretation:

- hierarchy with sharing still **helps** versus equally deep unshared hierarchy
- motif bias now **helps overall**
- clone mutation now has **real but still weak** impact
- flat macro baseline still strongest overall

Pack read:

- `tabular_local`: shared beats all other variants
- `image_hard`: shared beats unshared and no-motif-bias, but flat still dominates
- `openml_structured`: shared strongly beats unshared, mixed against flat
- `lm_full`: flat dominates; LM remains biggest weak spot

## Current Bottlenecks

- image tasks: shared structures still underperform flat macro on absolute score
- LM tasks: hierarchy costs too many params and current LM scorer still favors flat
- clone benefit present but not yet large enough to justify complexity

## 2026-04-18 Neural Head + Macro DAG Upgrade

Work completed:

- macro seeds/candidates now build real DAGs with skips, fan-in, and multi-sink outputs
- crossover now preserves inherited parent macro edges instead of collapsing to plain chains
- mutation now includes macro rewiring and skip-edge insertion
- seed genomes now start as real two-level graphs, not just stacked chains
- evaluator no longer relies on sklearn model selection as primary path
- classification now trains a neural GELU head with warm-start inheritance
- language modeling now trains a neural vocab projection head with softmax updates
- parameter counts now include trainable head params
- new focused configs added:
  - `configs/ablation_core_graph_runtime.yaml`
  - `configs/ablation_lm_smoke_runtime.yaml`

## Commands Run

```bash
uv run --extra dev pytest -q tests/test_config_genome_compile.py tests/test_search_ops.py tests/test_lm_cache_runtime.py
uv run --extra dev pytest -q tests/test_benchmarks.py
uv run --extra dev pytest -q tests/test_ablation_motifs_runtime.py::test_ablation_suite_outputs_report
uv run stratograph ablate --config configs/ablation_core_graph_runtime.yaml --workspace manual_compare_runs/ablation_core_graph_runtime_20260418
uv run stratograph ablate --config configs/ablation_image_hard.yaml --workspace manual_compare_runs/ablation_image_hard_20260418_runtime_graph
```

LM follow-up note:

- attempted:
  - `uv run stratograph ablate --config configs/ablation_lm_full.yaml --workspace manual_compare_runs/ablation_lm_full_20260418_runtime_graph`
  - `uv run stratograph ablate --config configs/ablation_lm_smoke_runtime.yaml --workspace manual_compare_runs/ablation_lm_smoke_runtime_20260418`
- both were stopped after confirming LM runtime cost increased sharply; they did not finish in-turn

## Main Findings

Image hard:

- shared vs flat: `0` wins, `4` losses
- shared vs unshared: `2` wins, `2` losses
- shared vs no-clone: `2` wins, `1` loss, `1` tie
- shared vs no-motif-bias: `4` wins, `0` losses

Core graph/runtime pack (`moons`, `digits`, `tiny_lm_synthetic`):

- shared vs flat: `1` win, `2` losses
- shared vs unshared: `1` win, `2` losses
- shared vs no-clone: `2` wins, `0` losses, `1` tie
- shared vs no-motif-bias: `1` win, `2` losses

Direct read:

- new code really produces graphier winners
  - shared core winners show branch factor `1.83` to `2.25`
  - image shared winners show branch factor `1.67` to `2.25`
- but graphier plus more trainable runtime did **not** create broad dominance
- shared hierarchy still helps against weaker hierarchy variants in image
- shared hierarchy still loses absolute image score to flat macro
- simple LM signal from `tiny_lm_synthetic` still favors unshared in this run
- new neural runtime is more realistic, but also much more expensive on LM

## Updated Bottlenecks

- macro DAG is now real, but cell internals still do not beat flat image backbone in absolute score
- trainable head improved realism more than raw win rate
- LM runtime maturity increased compute cost before it produced better hierarchy results
- next LM work must improve both efficiency and quality, not only realism

## Next Best Work

1. image-specific compiler/runtime changes, not only better heads
2. real LM head instead of count-based surrogate
3. stronger clone-specialize scheduling
4. motif library learned from winning runs instead of fixed priors
