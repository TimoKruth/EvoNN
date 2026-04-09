# Evo Neural Nets — Superproject Instructions

## Project Structure

This is a git superproject with three submodules:

- `EvoNN/` — Track A: family-based macro NAS (17 model families, 102 benchmarks)
- `EvoNN-2/` — Track B: NEAT topology evolution (speciation, QD, mixed-precision)
- `EvoNN-Symbiosis/` — Comparison layer + Hybrid engine + Observatory dashboard

All three use Python 3.13+, MLX, uv, DuckDB, Pydantic 2.

## Key Paths

- Superproject root: `/Users/timokruth/Projekte/Evo Neural Nets`
- EvoNN root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN`
- EvoNN-2 root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-2`
- Symbiosis root: `/Users/timokruth/Projekte/Evo Neural Nets/EvoNN-Symbiosis`

## Running EvoNN

Always `cd` to the EvoNN root first. Key commands:

```bash
uv run evonn benchmarks warm-cache --pack <pack>
uv run evonn evolve run --config <config.yaml>
uv run evonn symbiosis export <run_id> --pack <pack>
uv run evonn compare <contender> --pack <pack> --trial-budget <n>
```

Outputs go to `EvoNN/runs/<run_id>/`. Per-run DuckDB is default for new runs.

## Running EvoNN-2

Always `cd` to the EvoNN-2 root. Key commands:

```bash
uv run evonn2 evolve --config <config.yaml> --run-dir <dir>
uv run evonn2 evolve --config <config.yaml> --run-dir <dir> --resume
uv run evonn2 symbiosis export <run-dir> --pack <pack>
```

Each run gets its own `metrics.duckdb` in the run directory.

## Running Symbiosis

Always `cd` to the Symbiosis root. Key commands:

```bash
# Comparison campaign (EvoNN vs EvoNN-2)
uv run symbiosis campaign --pack <pack> --seeds 42,43,44 --budgets 128 --execute

# Solo campaign (one system)
uv run symbiosis solo --system <evonn|evonn2|hybrid> --pack <pack> --seeds 42,43 --budget 128

# Three-way symbiosis (all three systems)
uv run symbiosis symbiosis-campaign --pack <pack> --seeds 42,43 --budget 128

# Hybrid only
uv run symbiosis hybrid --pack <pack> --seed 42 --population 8 --generations 5 --epochs 20
uv run symbiosis hybrid-resume --run-dir <dir>

# Observatory dashboard
uv run symbiosis observatory --evonn-root ../EvoNN --evonn2-root ../EvoNN-2 --port 8417

# System info
uv run symbiosis info
```

## DuckDB Lock Policy

- EvoNN uses per-run DuckDB (`per_run_db: true`) — parallel runs are safe
- EvoNN-2 always uses per-run DuckDB — parallel runs are safe
- Hybrid uses per-run DuckDB via HybridRunStore — parallel runs are safe
- The old global `EvoNN/runs/metrics.duckdb` still exists for historical reads
- Never run two processes writing to the same DuckDB file

## Testing

```bash
# EvoNN
cd EvoNN && uv run python -m pytest -q

# EvoNN-2 (exclude known MoE segfault)
cd EvoNN-2 && uv run python -m pytest --deselect tests/test_moe.py::test_moe_gradient_flows -q

# Symbiosis
cd EvoNN-Symbiosis && uv run pytest -q
```

## Parity Packs

Shared benchmark definitions live in `EvoNN-Symbiosis/parity_packs/`:
- `tier1_core.yaml` — 8 symmetric benchmarks (main comparison)
- `tier2_evonn_leaning.yaml` — 6 image/tabular (EvoNN advantage)
- `tier3_evonn2_leaning.yaml` — 4 topology (EvoNN-2 advantage)

Generated packs (if benchmark-matrix has been run):
- `parity_packs/generated/all_shared.yaml`
- `parity_packs/generated/all_shared_tabular.yaml`

## Canonical Benchmark IDs

Both parents use different native IDs. The symbiosis export maps to canonical:
- `iris` (EvoNN-2) → `iris_classification` (canonical)
- `friedman_regression` (EvoNN) → `friedman1_regression` (canonical)

See `CANONICAL_BENCHMARK_IDS` in each project's symbiosis export module.

## Commits

When committing across projects, commit each submodule separately, then update the superproject:

```bash
cd EvoNN && git add ... && git commit -m "..."
cd EvoNN-2 && git add ... && git commit -m "..."
cd EvoNN-Symbiosis && git add ... && git commit -m "..."
cd /Users/timokruth/Projekte/Evo\ Neural\ Nets && git add -A && git commit -m "Update submodule refs: ..."
```
