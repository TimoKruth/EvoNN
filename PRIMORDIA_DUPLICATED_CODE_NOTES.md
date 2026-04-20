# Primordia Duplicated Code Notes

This repository intentionally duplicates some code inside `EvoNN-Primordia` so that Primordia can run independently of the other EvoNN workspace packages.

## Why this document exists

The user explicitly requested that Primordia become completely independent from the other projects, even if that requires duplication.

That means Primordia should not import runtime, benchmark, model, genome, or export logic from sibling packages such as:
- `EvoNN-Prism`
- `EvoNN-Contenders`
- `EvoNN-Compare`
- `EvoNN-Topograph`
- `EvoNN-Stratograph`

## What was duplicated into Primordia

The following code families were copied into `EvoNN-Primordia/src/evonn_primordia/` and adapted to use Primordia-local imports:

### MLX runtime / model search core
- `genome.py`
- `families/models.py`
- `families/compiler.py`
- `runtime/training.py`

These were derived from the MLX-oriented search/runtime approach that previously lived in Prism.

### Benchmark and parity support
- `benchmarks/spec.py`
- `benchmarks/datasets.py`
- `benchmarks/lm.py`
- `benchmarks/parity.py`
- `benchmarks/__init__.py`

These were duplicated so Primordia can load benchmark metadata and parity packs without importing sibling packages.

## Why duplication was chosen

This duplication is deliberate, not accidental.

Benefits:
- Primordia can be installed and run on its own.
- Refactors in sibling packages do not break Primordia imports.
- Primordia can evolve its own benchmark/runtime/model logic independently.
- Local Apple Silicon MLX runs can be treated as Primordia-native rather than delegated through Prism.

Costs:
- Bug fixes may need to be applied in more than one package.
- Shared logic can drift over time.
- Maintenance burden is higher until a future stable shared library exists.

## Maintenance guidance

If a bug is fixed in duplicated logic elsewhere, check whether the corresponding Primordia-local copy also needs the fix.

If future shared abstractions are introduced, do **not** automatically re-couple Primordia to sibling packages unless independence is no longer a requirement.

## Current intent

At this point, Primordia is intended to be a self-contained MLX-capable package within the monorepo, not a thin wrapper around Prism or Contenders.
