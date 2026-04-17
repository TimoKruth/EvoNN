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

## Next Best Work

1. image-specific compiler/runtime changes, not only better heads
2. real LM head instead of count-based surrogate
3. stronger clone-specialize scheduling
4. motif library learned from winning runs instead of fixed priors
