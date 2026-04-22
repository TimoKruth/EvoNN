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
- Primordia-local benchmark and parity loaders
- MLX-backed primitive candidate search
- runtime metadata carried through run/export artifacts
- primitive usage, benchmark-group coverage, and failure telemetry carried through reports/exports
- primitive bank summary artifact emitted alongside run artifacts and compare exports for later seeding-style analysis
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
uv run --package evonn-primordia primordia symbiosis export --run-dir path/to/run --pack-path path/to/pack.yaml
```

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
runtime backend used. That makes it possible to include Primordia in fair
EvoNN-Compare matrix runs alongside Prism, Topograph, Stratograph, and the
Contenders baseline.

## Core Docs

- `VISION.md`
- `IMPLEMENTATION_PLAN.md`
- `ARCHITECTURE_RULES.md`
- `CHANGELOG.md`
