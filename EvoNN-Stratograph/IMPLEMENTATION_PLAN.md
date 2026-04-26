# Stratograph Implementation Plan

## Status

This file is now an archived bootstrap record.

Its original purpose was to define how Stratograph would be built as a new
greenfield hierarchical sibling project. That greenfield stage is complete:
Stratograph now exists as a real package with genome/compiler/runtime,
compare-compatible exports, and tests.

## Active Planning Surface

Current Stratograph advancement work now lives here:

- [.hermes/plans/README.md](../.hermes/plans/README.md)
- [.hermes/plans/2026-04-27_102500-stratograph-engine-advancement-plan.md](../.hermes/plans/2026-04-27_102500-stratograph-engine-advancement-plan.md)

Use those files for current branch execution.

## What This File Still Means

Keep this document only as historical context for:

- the original greenfield design intent
- the early hierarchy-first package boundary
- the fact that Stratograph began as a build-out plan rather than as a mature
  challenger

It should not be treated as a current execution plan anymore.

1. `moons_classification`
2. `digits_image`
3. `tiny_lm_synthetic`
4. LM-only smoke
5. full 38 at smoke budget
6. full 38 at 76
7. full 38 at 152
8. full 38 at 304
9. full 38 at current broad 608 lane

Exit criteria:
- all 38 benchmarks exported
- budget parity accepted
- pairwise comparison reports generated

## What This Turn Implements

- project folder and docs
- fresh package scaffold
- local benchmark boundary
- hierarchical genome models and codec
- hierarchical compiler
- hierarchy-aware mutation/crossover search loop
- fast evaluator for classification/image/LM tasks
- compare-compatible run/export surface
- execution ladder generation and runs
- tests for config, benchmark loading, compiler, search, pipeline, export

## What This Turn Does Not Finish

- full long-horizon hierarchy-specialized trainer optimization
- long-horizon weight inheritance across training runs
- advanced novelty/QD scheduling beyond current lightweight archive
- compare-side orchestration automation
- final merged-system work across Prism/Topograph/Stratograph

Those remain next slices.
