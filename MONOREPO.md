# EvoNN Monorepo

This directory is the uv workspace root for the EvoNN umbrella.

Current workspace members:

- `EvoNN-Compare`
- `EvoNN-Contenders`
- `EvoNN-Primordia`
- `EvoNN-Prism`
- `EvoNN-Shared`
- `EvoNN-Stratograph`
- `EvoNN-Topograph`

## Structure

The monorepo is intentionally an umbrella research stack, not a single merged
runtime.

Foundation packages:
- `EvoNN-Compare`
- `EvoNN-Contenders`
- `EvoNN-Shared`
- `shared-benchmarks/`

Shared-infrastructure candidates that may deserve first-class umbrella modules
over time:
- benchmark/parity-pack resolution helpers
- export/manifest/summary helpers
- report-rendering helpers
- run-storage/schema helpers
- telemetry/budget/seeding metadata models
- CLI support helpers

The first shared substrate package now lives in `EvoNN-Shared/` as `evonn_shared`.
Its current role is intentionally narrow: shared compare/export contracts, minimal
benchmark descriptors, budget metadata, and run identity helpers.

Search packages:
- `EvoNN-Primordia`
- `EvoNN-Prism`
- `EvoNN-Topograph`
- `EvoNN-Stratograph`

Root-level strategy docs:
- `VISION.md`
- `ROADMAP.md`
- `BENCHMARK_LADDER.md`
- `BUDGET_CONTRACT.md`
- `TELEMETRY_SPEC.md`
- `SEEDING_LADDERS_IMPLEMENTATION_PLAN.md`

## Structural Unification Policy

The monorepo is allowed to unify shared research infrastructure, but it should
not erase the scientific distinctness of the search systems.

Good shared-substrate candidates:
- benchmark catalog and parity-pack resolution
- compare-facing export helpers
- common report sections and rendering utilities
- run metadata / evaluation storage primitives
- telemetry, budget, and seeding validation models
- recurring CLI helper patterns

Keep package-local unless strong evidence suggests otherwise:
- genome definitions
- mutation and crossover logic
- compiler/runtime implementations
- search-loop coordinators
- abstraction-specific telemetry that sits above the umbrella minimum contract

In short:
- unify infrastructure for trust, parity, and maintenance
- preserve search-core differences for science

## Commands

Install package dev dependencies from root:

```bash
uv sync --package evonn-compare --extra dev
uv sync --package evonn-contenders --extra dev
uv sync --package evonn-primordia --extra dev
uv sync --package prism --extra dev
uv sync --package stratograph --extra dev
uv sync --package topograph --extra dev
```

Run package CLIs from root:

```bash
uv run --package evonn-compare python -m evonn_compare --help
uv run --package evonn-contenders evonn-contenders --help
uv run --package evonn-primordia primordia --help
uv run --package prism prism --help
uv run --package stratograph stratograph --help
uv run --package topograph topograph --help
```

Run package tests from root where implemented:

```bash
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
uv run --package evonn-contenders --extra dev pytest -q EvoNN-Contenders/tests
uv run --package prism --extra dev pytest -q EvoNN-Prism/tests
uv run --package stratograph --extra dev pytest -q EvoNN-Stratograph/tests
uv run --package topograph --extra dev pytest -q EvoNN-Topograph/tests
```

## Adding More Packages

When another package is ready for monorepo ownership:

1. move it under this root
2. keep its package-local `pyproject.toml`
3. add its folder path to `[tool.uv.workspace].members`
4. validate with `uv sync --package <name>`
5. update `VISION.md` and `MONOREPO.md` if it changes the umbrella structure
