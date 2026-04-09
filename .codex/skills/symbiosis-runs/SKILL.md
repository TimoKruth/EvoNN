---
name: symbiosis-runs
description: Use when the task is to run EvoNN-Symbiosis workflows from the superproject, including matrix refresh, generated pack creation, solo runs, three-way campaigns, contender campaigns, and hybrid-only runs.
---

# Symbiosis Runs

Use this skill for the standardized orchestration layer.

## Repo root

- Symbiosis root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis`

Run these commands with `cwd` set to the Symbiosis repo root.

## Matrix and generated packs

Refresh the canonical benchmark matrix, derived sets, and generated runnable packs:

```bash
uv run symbiosis benchmark-matrix
```

Current generated shared packs live in:

- `parity_packs/generated/all_shared.yaml`
- `parity_packs/generated/all_shared_tabular.yaml`
- `parity_packs/generated/all_shared_image.yaml`
- `parity_packs/generated/all_shared_classification.yaml`
- `parity_packs/generated/all_shared_regression.yaml`

## Solo runs

Run one system across a pack and seed list:

```bash
uv run symbiosis solo \
  --system <evonn|evonn2|hybrid> \
  --pack <pack-name-or-yaml-path> \
  --seeds 42,43,44 \
  --budget 128 \
  --workspace campaigns/<name> \
  --root <project-root-if-needed>
```

Use `solo` for per-benchmark or per-pack fallback runs when native pack-wide comparison logic is not yet standardized in the target project.

## Three-way campaign

Run EvoNN, EvoNN-2, and Hybrid together:

```bash
uv run symbiosis symbiosis-campaign \
  --pack <pack-name-or-yaml-path> \
  --seeds 42,43,44 \
  --budget 128 \
  --workspace campaigns/<name>
```

This is the standard three-system parity workflow.

## Contender campaign

Run EvoNN contenders against a shared parity pack:

```bash
uv run symbiosis contender-campaign \
  --pack <pack-name-or-yaml-path> \
  --trial-budget <n> \
  --seed <n> \
  --epochs <n> \
  --batch-size <n> \
  --workspace campaigns/<name>
```

Optional backends that are not installed should be treated as skipped, not as structural workflow failures.

## Hybrid-only runs

Run Hybrid directly:

```bash
uv run symbiosis hybrid \
  --pack <pack-name-or-yaml-path> \
  --seed <n> \
  --population <n> \
  --generations <n> \
  --epochs <n> \
  --output campaigns/<name>/runs/<run-name>
```

Resume Hybrid:

```bash
uv run symbiosis hybrid-resume --run-dir <run-dir>
```

## Outputs to verify

- `reports/campaign_summary.md`
- `reports/three_way_summary.md`
- `reports/contender_summary.md`
- pairwise `.md` and `.json` comparison reports
- exported `manifest.json` and `results.json` in run directories

## Default policy

- Refresh the benchmark matrix before broad new overview runs.
- Prefer generated packs over ad hoc benchmark lists.
- Use Symbiosis as the standard place for cross-project and contender-aware runs.
