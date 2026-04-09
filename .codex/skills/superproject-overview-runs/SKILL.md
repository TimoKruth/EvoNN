---
name: superproject-overview-runs
description: Use when the task is to run or extend the cross-project overview workflows in the Evo Neural Nets superproject, especially the shared-pack three-way runs and contender comparisons that summarize EvoNN, EvoNN-2, Hybrid, and EvoNN contenders together.
---

# Superproject Overview Runs

Use this skill when the goal is a standardized shared-universe comparison from the superproject.

## Repo roots

- Superproject: `/Users/timokruth/Projekte/Evo Neural Nets`
- Symbiosis orchestrator: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis`

The orchestration entrypoint is Symbiosis.

## Standard overview sequence

1. Refresh the shared benchmark matrix:

```bash
cd /Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis
uv run symbiosis benchmark-matrix
```

2. Run the three-way shared-pack campaign:

```bash
uv run symbiosis symbiosis-campaign \
  --pack parity_packs/generated/all_shared.yaml \
  --seeds <seed-list> \
  --budget <budget> \
  --workspace campaigns/<three-way-workspace>
```

3. Run the contender comparison on the shared tabular subset:

```bash
uv run symbiosis contender-campaign \
  --pack parity_packs/generated/all_shared_tabular.yaml \
  --trial-budget <n> \
  --seed <n> \
  --epochs <n> \
  --batch-size <n> \
  --workspace campaigns/<contender-workspace>
```

## What this currently means

- `all_shared.yaml` is the current three-system shared universe.
- `all_shared_tabular.yaml` is the current contender-compatible shared tabular universe.
- For `EvoNN-2` and `Hybrid`, the standardized comparison path is still pack-based or singleton-pack based through Symbiosis.
- Do not assume a native “train on everything once, then compare every benchmark” mode exists yet for EvoNN-2 or Hybrid.

## Large-run policy

- Let Symbiosis generate the project-specific configs.
- Let Symbiosis warm EvoNN caches before large shared runs.
- Use one workspace per major run.
- Validate completion from report files, not just process exit.

Expected final artifacts:

- Three-way:
  - `reports/three_way_summary.md`
  - pairwise comparison reports
- Contenders:
  - `reports/contender_summary.md`

## Extension policy

When the user asks for a broader overview:

- first refresh the matrix
- then prefer generated packs
- then scale pack by pack
- only add new benchmarks to the overview after they are enabled and exportable in all relevant systems

## Current canonical packs

- `parity_packs/generated/all_shared.yaml`
- `parity_packs/generated/all_shared_tabular.yaml`
- `parity_packs/generated/all_shared_image.yaml`
- `parity_packs/generated/all_shared_classification.yaml`
- `parity_packs/generated/all_shared_regression.yaml`
