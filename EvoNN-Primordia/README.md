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

## Initial State

This package is being introduced as the Horizon 2 umbrella completion step.
The initial implementation intentionally starts small:
- package scaffold
- CLI placeholder
- explicit vision and implementation plan
- workspace integration

That is enough to give EvoNN a concrete home for primitive-first research
without prematurely locking in one runtime design.

## CLI

From monorepo root:

```bash
uv run --package evonn-primordia primordia --help
```

## Core Docs

- `VISION.md`
- `IMPLEMENTATION_PLAN.md`
