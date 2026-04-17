# EvoNN-Compare

Protocol-first comparison layer for Prism, Topograph, Stratograph, Hybrid, and contender exports.

Prefer root-level `uv` commands from monorepo root:

```bash
uv sync --package evonn-compare --extra dev
uv run --package evonn-compare --extra dev pytest -q EvoNN-Compare/tests
uv run --package evonn-compare python -m evonn_compare --help
```

Package metadata stays in [pyproject.toml](./pyproject.toml). Workspace lock lives at monorepo root.
