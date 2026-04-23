# Changelog

## 0.1.1 - 2026-04-21

Runtime and boundary maturity update.

Added / changed:
- MLX-backed hierarchical compiler
- MLX-backed trainable classification and LM heads
- shared-benchmarks-driven parity-pack resolution
- ladder generation that no longer depends on EvoNN-Compare manual run artifacts
- runtime backend/version metadata now recorded in run budget metadata and reused by compare exports
- architecture rules document capturing the distinct-project and shared-boundary rules from the vision docs

Known limits:
- no long-horizon hierarchy-specialized weight inheritance yet
- no compare-side orchestration automation in scope here
- no final merged-system work across Prism/Topograph/Stratograph

## 0.1.0-alpha - 2026-04-17

Initial Stratograph foundation.

Added:
- fresh sibling project `EvoNN-Stratograph`
- hierarchy-first genome with macro graph + reusable cell library
- deterministic hierarchical compiler
- benchmark boundary for 38 compare benchmarks
- fast evaluator for classification, image, and LM tasks
- hierarchy-aware mutation and crossover
- novelty/QD-lite telemetry
- compare-capable startup/export surface
- execution ladder generation and completed ladder artifacts

Artifacts:
- `manifest.json`
- `results.json`
- `report.md`
- `config.yaml`
- `genome_summary.json`
- `model_summary.json`
- `dataset_manifest.json`

Known limits:
- no MLX trainer yet
- no long-horizon weight inheritance yet
- no compare-side orchestration automation in scope here
