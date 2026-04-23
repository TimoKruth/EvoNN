# Topograph Architecture Rules

This file records the operational rules implied by the EvoNN umbrella vision and
`EvoNN-Topograph/VISION.md`.

## 1. Topograph must stay a distinct topology-first search system

Topograph exists to test topology-first neural architecture search as its own
scientific claim.

That means Topograph should keep its own:

- graph/genome assumptions
- topology compiler/runtime choices
- topology mutation, crossover, and archive logic
- topology-specific telemetry and reporting
- topology-specific artifact surface

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

The following should remain Topograph-local unless a future stable umbrella
library can preserve project independence:

- topology runtime/compiler behavior
- topology mutation, crossover, and scheduling logic
- hardware-aware search heuristics
- topology telemetry and reporting logic

## 3. Extract general parts when they are not unique

If a component is not unique to Topograph and the project could still work
standalone with explicit benchmark data or documented inputs, it should live at
a common boundary instead of being coupled to a sibling package's private
internals.

Good examples of shared/common substrate:

- benchmark metadata
- parity pack definitions
- reusable datasets and caches
- compare-facing manifests and result schemas

Bad examples of accidental coupling:

- importing a sibling runtime/compiler to execute Topograph candidates
- depending on another project's private run artifacts to resolve packs
- moving topology-first logic into a shared helper that erases the project's
  distinct search model

## 4. Topology is the primary search object

Topograph should make topology explicit in:

- genome structure
- compiler
- search operators
- hardware-aware scheduling
- telemetry
- export summaries

If graph structure gets reduced to shallow parameter tuning, Topograph stops
answering its intended research question.

## 5. Hardware reality must stay in the loop

Topograph should preserve evidence about:

- latency and memory tradeoffs
- parameter and model-byte constraints
- where hardware-aware pressure changes the frontier
- which motifs survive under device limits
- what weight inheritance or reuse actually saves

Device-aware evidence matters alongside benchmark quality.

## 6. Comparable outside, different inside

Externally Topograph should line up with Prism, Primordia, Stratograph, and
Compare on:

- canonical benchmark IDs
- parity-pack compatibility
- export contract shape
- budget semantics
- fair-compare expectations

Internally it should still teach something different.

## 7. Local-first MLX execution is part of the umbrella standard

Topograph should continue to meet the umbrella's Apple-Silicon / MLX local-first
standard while keeping its topology-first runtime distinct.

## 8. Standalone should remain possible

Topograph should be runnable on its own when given explicit benchmark data,
shared benchmark definitions, or documented pack inputs. Shared substrate is
fine. Hidden dependence on sibling package internals is not.
