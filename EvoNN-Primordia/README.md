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
- budget-matched primitive candidate search
- per-benchmark best primitive selection
- compare-ready manifest/results export
- markdown + JSON run artifacts
- workspace integration

This is intentionally still a small primitive lane, not the final motif-bank or
upstream seeding system.

## CLI

From monorepo root:

```bash
uv run --package evonn-primordia primordia --help
uv run --package evonn-primordia primordia run --config path/to/config.yaml
uv run --package evonn-primordia primordia symbiosis export --run-dir path/to/run --pack-path path/to/pack.yaml
```

## Fair Comparison Role

Primordia currently acts as a primitive-first search system with an explicitly
budget-matched evaluation count. That makes it possible to include Primordia in
fair EvoNN-Compare matrix runs alongside Prism, Topograph, Stratograph, and the
Contenders baseline.

## Core Docs

- `VISION.md`
- `IMPLEMENTATION_PLAN.md`
