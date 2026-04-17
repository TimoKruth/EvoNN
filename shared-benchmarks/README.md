# Shared Benchmarks

This folder is the shared benchmark source of truth for the Evo Neural Nets superproject.

Layout:

- `catalog/`: benchmark YAML definitions
- `suites/`: benchmark suite YAMLs
- `migration/`: notes for normalization and cutover work

Resolution rule:

- if a project has a local fallback and this folder disagrees, this folder should win

Environment:

- `EVONN_SHARED_BENCHMARKS_DIR` may point at this folder

Current baseline:

- catalog and suites were initialized from `EvoNN-Topograph/benchmarks/`
