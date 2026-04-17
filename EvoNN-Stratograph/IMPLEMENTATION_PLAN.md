# Stratograph Implementation Plan

## Goal

Build `EvoNN-Stratograph` as a greenfield sibling project that can later run on
the same 38-benchmark comparison lane as Prism and Topograph while preserving a
truly distinct core design.

## Source Of Truth For Comparison

Use the current broad 38-benchmark comparison lane as gold reference, not the
older smoke-only `eval38` lane.

Required compatibility surface:
- same 38 canonical benchmark IDs
- same task kinds
- same metric names and directions
- same parity-pack based budget semantics
- same `manifest.json` / `results.json` export contract

## Architecture Plan

### Phase 0: Design Lock

- Finalize hierarchical genome models
- Finalize compiler interfaces and invariants
- Finalize hierarchy-specific telemetry
- Finalize export compatibility rules

Exit criteria:
- one written design spec
- one worked example genome -> compiled summary -> export summary

### Phase 1: Project Foundation

- Create new repo root and package
- Add CLI, config models, storage, reporting, tests
- Add smoke config for the 38-benchmark set

Exit criteria:
- package installs
- CLI responds
- run directory layout stable

### Phase 2: Benchmark Boundary

- Implement local benchmark loading
- Reuse shared benchmark catalogs only at boundary
- Support all 38 tasks including 5 LM tasks
- Implement parity-pack loading and native-id resolution

Exit criteria:
- all 38 benchmarks load
- native/canonical mappings resolve

### Phase 3: Hierarchical Genome

- Implement macro graph genes
- Implement cell library genes
- Implement micro-graph genes
- Implement serialization and invariant validation

Exit criteria:
- random genomes generate
- genomes serialize/deserialize
- invariants enforced

### Phase 4: Hierarchical Compiler

- Compile cells independently
- Compile macro graph over cell instances
- Support shared cells and cloned cells
- Support tabular, image, and LM forward shapes

Exit criteria:
- deterministic compile
- forward passes for classification and LM inputs
- parameter estimate and architecture summary available

### Phase 5: Prototype Runtime

- Create prototype evolution loop
- Persist genomes and benchmark records
- Produce exportable runs even before full trainer exists

Exit criteria:
- `evolve` creates run dir
- `inspect`, `report`, `symbiosis export` work

### Phase 6: Full Trainer

- Replace prototype skipped benchmark results with real training
- Implement classification and LM evaluation
- Add hierarchical weight transfer

Exit criteria:
- real `ok` benchmark records
- single-benchmark quality better than random

### Phase 7: Search

- Add hierarchy-aware mutations
- Add hierarchy-aware crossover
- Add optional novelty/QD descriptors

Exit criteria:
- search improves over seed on at least one classification and one LM benchmark

### Phase 8: Compare Integration

- Extend compare contract to accept `stratograph`
- Add labels and slots
- Add pairwise compare workflows
- Later add campaign generation for Stratograph

Exit criteria:
- `Stratograph vs Prism` compare works
- `Stratograph vs Topograph` compare works

### Phase 9: 38-Benchmark Ladder

Run progression:

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

- full MLX trainer
- weight inheritance across training runs
- advanced novelty/QD scheduling beyond current lightweight archive
- compare-side orchestration automation
- final merged-system work across Prism/Topograph/Stratograph

Those remain next slices.
