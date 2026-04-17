# EvoNN Monorepo

This directory is the uv workspace root for EvoNN packages.

Current workspace members:

- `EvoNN-Prism`

## Structure

- monorepo root owns workspace-level sync and lock
- package metadata stays inside each package `pyproject.toml`
- package folder names stay stable

## Commands

Install Prism dev deps from root:

```bash
uv sync --package prism --extra dev
```

Run Prism tests from root:

```bash
uv run --package prism pytest --cov=prism --cov-report=term-missing
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
4. validate with `uv sync --package <name>`
