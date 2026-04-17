# Prism

Prism is a family-based evolutionary neural architecture search system for Apple Silicon.

It searches across multiple model families, tracks lineage and archives, and exports comparable run artifacts for the wider EvoNN ecosystem.

Inside the EvoNN monorepo, prefer running `uv` commands from the repository root with `--package prism`.

## Status

This repo now vendors a minimal self-contained benchmark catalog and parity pack set for local smoke runs and CI:

- `benchmarks/catalog/`
- `parity_packs/`

Broader benchmark suites can still be used, but they must be configured explicitly through environment variables instead of relying on sibling repos.

## Requirements

- Python `3.13`
- Apple Silicon / macOS for MLX-backed training
- `uv`

## Install

Base install:

```bash
uv sync
```

Dev tools:

```bash
uv sync --extra dev
```

Optional dataset extras for OpenML-backed benchmarks:

```bash
uv sync --extra data
```

## Data And Pack Paths

Prism now resolves assets from local repo paths first, then explicit env vars.

- benchmark catalog: `benchmarks/catalog/` or `PRISM_CATALOG_DIR`
- parity packs: `parity_packs/` or `PRISM_PARITY_PACK_DIRS`
- LM cache files: `benchmarks/lm_cache/`, `~/.prism/datasets/`, or `PRISM_LM_CACHE_DIR`

If a catalog, pack, or LM cache is missing, Prism should fail with a direct error telling you which path was checked.

## Tiny Smoke Run

This is the fastest self-contained run in the repo. It uses the local `moons` benchmark and a single-generation search budget.

```bash
uv run prism evolve -c configs/tiny_smoke/config.yaml --run-dir runs/tiny_smoke
```

Inspect and report:

```bash
uv run prism inspect runs/tiny_smoke
uv run prism report runs/tiny_smoke
```

## LM Example

The bundled LM config includes one synthetic benchmark plus optional cached LM datasets:

```bash
uv run prism evolve -c configs/lm5_direct_tiny/config.yaml --run-dir runs/lm5_direct_tiny
```

`tinystories_lm` and `wikitext2_lm` require cached `.npz` windows. Point Prism at them with `PRISM_LM_CACHE_DIR` if they are not stored in `benchmarks/lm_cache/`.

## Development

Lint:

```bash
uv run ruff check .
```

Tests with coverage:

```bash
uv run pytest --cov=prism --cov-report=term-missing
```

## Repo Policy

- generated run outputs stay out of git
- local smoke assets live in this repo
- external catalogs and packs must be explicit, not implicit sibling-repo lookups
