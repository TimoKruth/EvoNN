---
name: evonn-runs
description: Use when the task is to run EvoNN directly from the superproject, including cache prep, direct evolution runs, contender comparisons, or Symbiosis export of EvoNN results.
---

# EvoNN Runs

Use this skill when the user wants a direct `EvoNN` run rather than a Symbiosis-managed wrapper.

## Repo root

- Superproject root: `/Users/timokruth/Projekte/Evo Neural Nets`
- EvoNN root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN`

Always run EvoNN commands with `cwd` set to the EvoNN repo root.

## Canonical commands

Warm cache for a parity or generated pack before larger runs:

```bash
uv run evonn benchmarks warm-cache --pack <pack-name-or-yaml-path>
```

Run one evolution job from a prepared config:

```bash
uv run evonn evolve run --config <config.yaml>
```

Export a completed EvoNN run into the Symbiosis contract:

```bash
uv run evonn symbiosis export <run_id> --pack <pack-name-or-yaml-path>
```

Run contender comparisons directly from EvoNN:

```bash
uv run evonn compare <ray-tune|nni|autogluon|tabpfn|xgboost|lightgbm|flaml> \
  --pack <pack-name-or-yaml-path> \
  --trial-budget <n> \
  --seed <n> \
  --epochs <n> \
  --batch-size <n>
```

## Standard workflow

1. Warm cache for the exact pack first.
2. Run `evolve run` with a generated or hand-written config.
3. Resolve the run ID from stdout or from `runs/*/config.json`.
4. Export with `evonn symbiosis export` if the run will be compared in Symbiosis.

## Outputs to expect

- Run directory: `EvoNN/runs/<run_id>`
- Structured export files after export:
  - `manifest.json`
  - `results.json`
  - `dataset_manifest.json`
  - `model_summary.json`
  - `genome_summary.json`

## Practical notes

- Generated pack YAML paths from `EvoNN-Symbiosis/parity_packs/generated/*.yaml` are valid `--pack` inputs.
- For large bridge packs, always warm cache first. This is now part of the standardized workflow.
- For contender work across shared packs, prefer the Symbiosis contender runner unless the user explicitly wants a single EvoNN-only compare run.
