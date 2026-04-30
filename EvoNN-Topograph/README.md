# EvoNN-Topograph

Topology-first evolutionary neural architecture search.

Topograph is the primary challenger to Prism on the shared Compare surface. It
searches graph structure directly, tracks topology and benchmark telemetry, and
exports compare-ready `manifest.json`, `results.json`, and `summary.json`
artifacts.

## Scope

- Evolve topology-first neural architectures on shared benchmark packs
- Track genome structure, fitness, benchmark outcomes, runtime metadata, and
  failure patterns
- Support report and inspect surfaces for completed or in-progress runs
- Export Symbiosis/Compare-compatible artifacts
- Carry canonical unseeded or seeded transfer metadata in compare manifests

## Run

From the monorepo root:

```bash
uv sync --package topograph --extra dev
uv run --package topograph topograph evolve \
  --config EvoNN-Topograph/configs/tiny_smoke/config.yaml \
  --run-dir EvoNN-Topograph/runs/tiny-smoke
uv run --package topograph topograph inspect EvoNN-Topograph/runs/tiny-smoke
uv run --package topograph topograph report EvoNN-Topograph/runs/tiny-smoke
```

Export a run for Compare:

```bash
uv run --package topograph topograph symbiosis export \
  EvoNN-Topograph/runs/tiny-smoke \
  --pack EvoNN-Topograph/parity_packs/tiny_smoke.yaml
```

## Useful Commands

```bash
uv run --package topograph topograph --help
uv run --package topograph topograph benchmarks
uv run --package topograph topograph suite --help
```

## Configs

Small local configs live under `configs/`:

- `tiny_smoke/config.yaml`
- `tiny_pool_smoke/config.yaml`
- `tiny_lm_smoke/config.yaml`
- `tiny_lm_synthetic_smoke.yaml`
- `tinystories_lm_smoke.yaml`
- `wikitext2_lm_smoke.yaml`

For recurring cross-project comparison, prefer the monorepo-level
`evonn-compare fair-matrix` presets instead of running Topograph by hand.

## Docs

- `VISION.md`
- `ARCHITECTURE_RULES.md`
- `../EVONN_90_DAY_PLAN.md`
- `../.hermes/plans/README.md`

## Quality

Package checks:

```bash
uv run --package topograph --extra dev ruff check EvoNN-Topograph
uv run --package topograph --extra dev pytest EvoNN-Topograph/tests --cov=topograph --cov-report=term-missing
```

The monorepo wrapper is:

```bash
bash scripts/ci/topograph-checks.sh all
```
