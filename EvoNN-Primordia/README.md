# EvoNN-Primordia

Primitive-first evolutionary search for EvoNN.

Primordia is the missing bottom-up layer in the EvoNN umbrella. It sits beneath
Prism, Topograph, and Stratograph and asks a lower-level question:

which tiny computational motifs deserve to exist before they are assembled into
larger neural architectures?

## Scope

Primordia focuses on cheap, local-first discovery of:
- single-neuron or tiny-group operator variants
- gated microcircuits
- tiny sparse motifs
- activation and merge patterns
- reusable low-level building blocks that may later seed higher-level search

## Role In The Umbrella

Primordia is not meant to replace the architecture-scale systems.

Its purpose is to:
- discover low-level reusable motifs cheaply
- build primitive priors and motif libraries
- export artifacts that later systems can optionally consume
- test whether bottom-up search improves later search efficiency

## Planned Outputs

Long run, Primordia should export:
- motif manifests
- motif bank summaries
- benchmark-conditioned motif reports
- candidate primitive libraries suitable for later seeding experiments

## Current State

Primordia now includes a cheap, runnable primitive-search lane aimed at fair
comparison against the rest of the umbrella.

Current deliverables:
- self-contained Primordia-local runtime boundary with MLX as the reference backend
- sklearn-backed `numpy-fallback` execution path for non-MLX local parity lanes
- Primordia-local benchmark and parity loaders
- MLX-backed primitive candidate search
- runtime metadata carried through run/export artifacts
- primitive usage, benchmark-group coverage, and failure telemetry carried through reports/exports
- primitive bank summary artifact emitted alongside run artifacts and compare exports for later seeding-style analysis
- benchmark-conditioned seed candidate artifact emitted for downstream family/topology/hierarchy seeding experiments
- richer markdown reports that include primitive-bank winners and representative genomes
- richer CLI inspection that summarizes runtime, usage, wins, and best benchmark outcomes from run artifacts
- budget-matched per-benchmark evaluation scheduling
- per-benchmark best primitive selection
- compare-ready manifest/results export
- markdown + JSON run artifacts
- workspace integration

This is intentionally still a small primitive lane, not the final motif-bank or
upstream seeding system. It is also intentionally implemented as a self-contained
package rather than a thin import wrapper around sibling EvoNN projects.

## CLI

From monorepo root:

```bash
uv run --package evonn-primordia primordia --help
uv run --package evonn-primordia primordia run --config path/to/config.yaml
uv run --package evonn-primordia primordia inspect --run-dir path/to/run
uv run --package evonn-primordia primordia report --run-dir path/to/run
uv run --package evonn-primordia primordia seed export --run-dir path/to/run
uv run --package evonn-primordia primordia symbiosis export --run-dir path/to/run --pack-path path/to/pack.yaml
```

Checked-in official lane configs now live under `configs/`:

```bash
uv run --package evonn-primordia primordia run --config EvoNN-Primordia/configs/smoke.yaml
uv run --package evonn-primordia primordia symbiosis export --run-dir path/to/run --pack-path EvoNN-Compare/parity_packs/tier1_core_smoke.yaml

uv run --package evonn-primordia primordia run --config EvoNN-Primordia/configs/tier1_core_eval64.yaml
uv run --package evonn-primordia primordia symbiosis export --run-dir path/to/run --pack-path EvoNN-Compare/parity_packs/tier1_core.yaml
```

The package-level official lane set is:
- `configs/smoke.yaml` for the `tier1_core_smoke` compare lane at 16 evaluations
- `configs/tier1_core_eval64.yaml` for the default local `tier1_core` lane
- `configs/tier1_core_eval256.yaml` for the overnight `tier1_core` lane
- `configs/tier1_core_eval1000.yaml` for the weekend `tier1_core` lane

`primordia report` now refreshes `report.md` from the current run artifacts when
`summary.json` is present, so stale markdown snapshots can be rebuilt after
export or telemetry updates without deleting the old report first.

`primordia inspect` now surfaces benchmark-group coverage, primitive-bank leaders,
representative primitive architectures, and recent failure reasons directly from
`summary.json` and `trial_records.json`, rebuilding the primitive-bank view from
`best_results.json`/`trial_records.json` when the bank artifact itself is
missing, so Primordia run introspection stays closer to Prism/Topograph-style
operator inspection.

## Fair Comparison Role

Primordia currently acts as a primitive-first search system with MLX as its
reference execution runtime and an explicitly budget-matched evaluation count.
The package now also installs cleanly on non-Darwin hosts by treating MLX as a
platform-specific dependency, while run/export artifacts still record the actual
runtime backend used. On hosts where MLX is unavailable, `primordia run` now
falls back to a clearly labeled `numpy-fallback` runtime for local
classification, regression, and image parity lanes such as
`tier1_core_eval64`/`tier1_core_eval256`.

That fallback keeps the compare/export contract intact for local fair-matrix
reruns without pretending to be the native MLX family compiler. MLX remains the
reference backend for native Primordia family execution, and it is still
required for text/language-modeling validation or any run where architectural
fidelity to the MLX families matters.

## Core Docs

- `VISION.md`
- `IMPLEMENTATION_PLAN.md` (archived bootstrap record only)
- `../EVONN_90_DAY_PLAN.md`
- `../.hermes/plans/README.md`
- `ARCHITECTURE_RULES.md`
- `CHANGELOG.md`
