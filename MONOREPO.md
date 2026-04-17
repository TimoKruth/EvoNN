# EvoNN Monorepo

This directory is the uv workspace root for EvoNN packages.

Current workspace members:

- `EvoNN-Compare`
- `EvoNN-Topograph`
- `EvoNN-Prism`

## Structure

- monorepo root owns workspace-level sync and lock
- package metadata stays inside each package `pyproject.toml`
- package-local `uv.lock` files should be removed after workspace adoption
- package folder names stay stable

## Commands

Install Compare dev deps from root:

```bash
uv sync --package evonn-compare --extra dev
```

Run Compare tests from root:

```bash
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
```

Run Compare CLI from root:

```bash
uv run --package evonn-compare python -m evonn_compare --help
```

Install Prism dev deps from root:

```bash
uv sync --package prism --extra dev
```

Run Prism tests from root:

```bash
uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
```

Run Prism CLI from root:

```bash
uv run --package prism prism --help
```

## Adding More Packages

When another package is ready for monorepo ownership:

1. move it under this root
2. keep its package-local `pyproject.toml`
3. add its folder path to `[tool.uv.workspace].members`
4. delete package-local `uv.lock`
5. validate with `uv sync --package <name>`
