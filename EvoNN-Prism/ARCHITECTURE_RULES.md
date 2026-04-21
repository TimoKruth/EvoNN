# Prism Architecture Rules

This file records the operational rules implied by the EvoNN umbrella vision and
`EvoNN-Prism/VISION.md`.

## 1. Prism must stay a distinct family-first search system

Prism exists to test family-based neural architecture search as its own research
claim.

That means Prism should keep its own:

- family-aware genome and compatibility logic
- family compiler/runtime choices
- family-aware search, archive, and reproduction policies
- family-level telemetry and reporting
- family-specific artifact surface

It should remain comparable to sibling systems. It should not become a thin
wrapper around them.

## 2. Shared boundary is intentional; shared core is not

The following cross-project sharing is intentional and should stay common where
possible:

- shared benchmark catalogs
- shared parity packs / suite definitions
- shared LM cache datasets
- shared export / compare contract expectations
- shared benchmark identities and budget semantics

The following should remain Prism-local unless a future stable umbrella library
can preserve project independence:

- family search/runtime behavior
- family mutation, crossover, and archive policy
- family-specific selection pressure
- family telemetry and reporting logic

## 3. Extract general parts when they are not unique

If a component is not unique to Prism and the project could still run standalone
with explicit benchmark data or documented inputs, it should live at a common
boundary instead of being coupled to a sibling package's private internals.

Good examples of shared/common substrate:

- benchmark metadata
- parity pack definitions
- reusable datasets and caches
- compare-facing manifests and result schemas

Bad examples of accidental coupling:

- importing a sibling runtime/compiler to execute Prism candidates
- depending on another project's private run artifacts to resolve packs
- hiding family-first logic inside a supposedly shared helper

## 4. Family diversity is the primary search object

Prism should make family diversity explicit in:

- genomes and constraints
- archive structure
- reproduction policy
- benchmark coverage signals
- inspection/reporting
- export summaries

If family diversity gets reduced to a single-template tuning loop, Prism stops
answering its intended research question.

## 5. Breadth and evidence matter as much as winners

Prism should preserve evidence about:

- which families survive on which benchmarks
- where transfer helps or fails
- which mutations improve families consistently
- how breadth changes under benchmark pressure
- why elites survive across packs

This evidence matters alongside top-line scores.

## 6. Comparable outside, different inside

Externally Prism should line up with Primordia, Topograph, Stratograph, and
Compare on:

- canonical benchmark IDs
- parity-pack compatibility
- export contract shape
- budget semantics
- fair-compare expectations

Internally it should still teach something different.

## 7. Local-first MLX execution is part of the umbrella standard

Prism should continue to meet the umbrella's Apple-Silicon / MLX local-first
standard while keeping its family-first runtime distinct.

## 8. Standalone should remain possible

Prism should be runnable on its own when given explicit benchmark data, shared
benchmark definitions, or documented pack inputs. Shared substrate is fine.
Hidden dependence on sibling package internals is not.
