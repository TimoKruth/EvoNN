# Stratograph Architecture Rules

This file records the operational rules implied by the EvoNN umbrella vision and
`EvoNN-Stratograph/VISION.md`.

## 1. Stratograph must stay a distinct project

Stratograph exists to test hierarchy-first search as its own scientific claim.
That means it should keep its own:

- genome assumptions
- compiler/runtime
- mutation and crossover logic
- training choices
- telemetry and failure modes

It is allowed to be comparable to sibling systems. It should not become a thin
wrapper around them.

## 2. Shared boundary is intentional; shared core is not

The following cross-project sharing is intentional and should stay common where
possible:

- shared benchmark catalogs
- shared parity packs / suite definitions
- shared LM cache datasets
- shared export / compare contract expectations
- shared benchmark identities and budget semantics

The following should remain Stratograph-local unless there is a future stable
umbrella library that does not destroy project independence:

- hierarchical runtime/compiler
- hierarchy-aware training loop
- hierarchy-aware search operators
- hierarchy-specific telemetry and reporting logic

## 3. Extract general parts when they are not unique

If a component is not unique to Stratograph and the project could still work
standalone with explicit training data or documented inputs, it should live at a
common boundary instead of being coupled to a sibling package's private internals.

Good examples of shared/common substrate:

- benchmark metadata
- parity pack definitions
- reusable datasets and caches
- compare-facing manifests and result schemas

Bad examples of accidental coupling:

- importing another project's runtime/compiler to execute Stratograph candidates
- depending on another project's manual run artifacts to resolve packs
- inheriting another project's search implementation instead of keeping a
  hierarchy-first core

## 4. Hierarchy is the primary search object

Stratograph should make hierarchy explicit in:

- genome
- compiler
- training/runtime
- telemetry
- inspection/reporting
- export summaries

If hierarchy gets hidden as a convenience feature inside a flat system, the
project stops answering its intended research question.

## 5. Reuse and specialization must both remain visible

Stratograph should preserve evidence about:

- shared-cell reuse
- clone and specialization events
- motif recurrence
- where specialization helps or hurts

This evidence matters as much as a top-line score.

## 6. Comparable outside, different inside

Externally Stratograph should line up with Prism, Topograph, Primordia, and
Compare on:

- canonical benchmark IDs
- parity-pack compatibility
- export contract shape
- budget semantics
- fair-compare expectations

Internally it should still teach something different.

## 7. Local-first MLX execution is part of the umbrella standard

Prism and Topograph already establish MLX / Apple Silicon as the local-first
reference point. Stratograph should meet that bar while keeping its hierarchy-
first runtime distinct.

## 8. Standalone should remain possible

Stratograph should be runnable on its own when given explicit benchmark data,
shared benchmark definitions, or documented pack inputs. Shared substrate is
fine. Hidden dependence on sibling package internals is not.
