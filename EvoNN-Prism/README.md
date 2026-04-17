# Prism

Family-based evolutionary neural architecture search for Apple Silicon.

Prism searches across multiple model families instead of tuning one template. It runs local-first with MLX, stores lineage and metrics in DuckDB, and can export Symbiosis-compatible artifacts for cross-system comparison.

## Scope

- Search across families such as `mlp`, `conv2d`, `attention`, `sparse_attention`
- Evolve on shared benchmark packs
- Track evaluations, lineage, archives, and reports per run
- Export `manifest.json`, `results.json`, and `summary.json` for Symbiosis flows

## Inputs

Prism expects shared assets from superproject:

- benchmark catalog: `../shared-benchmarks/catalog/`
- parity/suite packs: `../shared-benchmarks/suites/`
- LM cache fixtures: `../shared-benchmarks/lm_cache/`

Overrides:

- `EVONN_SHARED_BENCHMARKS_DIR`
- `PRISM_CATALOG_DIR`
- `PRISM_PARITY_PACK_DIRS`
- `PRISM_LM_CACHE_DIR`

## Run

From monorepo root:

```bash
uv sync --package prism --extra dev
uv run --package prism prism evolve -c EvoNN-Prism/configs/tiny_smoke/config.yaml --run-dir runs/prism-tiny
uv run --package prism prism inspect runs/prism-tiny
uv run --package prism prism report runs/prism-tiny
```

From package dir:

```bash
uv run prism evolve -c configs/tiny_smoke/config.yaml --run-dir runs/prism-tiny
uv run prism inspect runs/prism-tiny
uv run prism report runs/prism-tiny
```

## Outputs

Each run writes:

- `config.yaml` or `config.json`
- `metrics.duckdb`
- `summary.json`
- `report.md`
- `checkpoints/`

Symbiosis export writes:

- `manifest.json`
- `results.json`
- `summary.json`

## Quality

Package checks:

```bash
uv run --package prism --extra dev ruff check EvoNN-Prism
uv run --package prism --extra dev pytest EvoNN-Prism/tests --cov=prism --cov-report=term-missing
```

Tiny end-to-end smoke lives in `tests/test_cli_smoke_e2e.py` and `configs/tiny_smoke/config.yaml`.
