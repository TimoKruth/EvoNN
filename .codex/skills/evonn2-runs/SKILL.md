---
name: evonn2-runs
description: Use when the task is to run EvoNN-2 directly from the superproject, including evolve runs, resume flows, transfer checks, or Symbiosis export of EvoNN-2 results.
---

# EvoNN-2 Runs

Use this skill when the user wants a direct `EvoNN-2` run rather than a Symbiosis wrapper.

## Repo root

- Superproject root: `/Users/timokruth/Projekte/Evo Neural Nets`
- EvoNN-2 root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-2`

Always run EvoNN-2 commands with `cwd` set to the EvoNN-2 repo root.

## Canonical commands

Run one evolution job:

```bash
uv run evonn2 evolve --config <config.yaml> --run-dir <run-dir>
```

Resume a stopped run:

```bash
uv run evonn2 evolve --config <config.yaml> --run-dir <run-dir> --resume
```

Export a completed run into the Symbiosis contract:

```bash
uv run evonn2 symbiosis export <run-dir> --pack <pack-name-or-yaml-path>
```

Transfer the best genome from one run to a new benchmark:

```bash
uv run evonn2 transfer --from <run-dir> --benchmark <benchmark-id> \
  --epochs <n> --lr <float> --batch-size <n>
```

## Standard workflow

1. Prefer generated configs from Symbiosis for parity runs.
2. Use explicit `--run-dir` paths so resumability and exports are deterministic.
3. Export with `evonn2 symbiosis export` before any cross-system comparison.

## Outputs to expect

- Run directory: chosen `--run-dir`
- During run:
  - `config.yaml`
  - `metrics.duckdb`
- After export:
  - `manifest.json`
  - `results.json`
  - compact artifact summaries

## Current comparison policy

- For shared-pack comparisons, prefer Symbiosis to generate the pack-aware EvoNN-2 config.
- Until a native “train once across all benchmarks, then compare each benchmark” workflow is formalized, the standardized comparison path is still per-pack or per-benchmark via Symbiosis orchestration.
