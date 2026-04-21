# Primordia Architecture Rules

This file records the operational rules implied by the EvoNN umbrella vision and
`EvoNN-Primordia/VISION.md`.

## 1. Primordia must stay a distinct primitive-first layer

Primordia exists to test whether useful low-level computational motifs can be
found profitably below the level of whole families, flat topologies, and
hierarchical cell systems.

That means Primordia should keep its own:

- primitive-search assumptions
- primitive genome and mutation logic
- primitive-family runtime/training path
- primitive-specific telemetry and reporting
- primitive-specific artifact surface

It may seed higher-level systems later, but it should not quietly disappear into
those systems now.

## 2. Shared boundary is intentional; shared core is not

The following cross-project sharing is intentional and should stay common where
possible:

- shared benchmark catalogs
- shared parity packs / suite definitions
- shared LM cache datasets
- shared compare/export contracts
- shared benchmark identities and budget semantics

The following should remain Primordia-local unless a future stable umbrella
library can preserve the scientific distinction of the primitive-first layer:

- primitive runtime/compiler choices
- primitive mutation/search loop
- primitive-family selection policy
- primitive-specific telemetry and reports

## 3. Extract general parts when they are not unique

If a component is not unique to Primordia and the project can still run
standalone with explicit data or documented inputs, it should live at a common
boundary rather than coupling Primordia to a sibling package's private internals.

Good shared/common substrate examples:

- benchmark metadata
- parity pack definitions
- reusable datasets and caches
- compare-facing manifests and result schemas

Bad accidental coupling examples:

- importing a sibling search/runtime core to evaluate Primordia candidates
- depending on another project's private manual run artifacts for pack resolution
- hiding primitive-first logic inside a higher-level search package

## 4. Cheap-first is part of the design, not a temporary shortcut

Primordia only makes strategic sense if it remains cheaper than full
architecture-scale search. If the primitive lane grows so expensive that it no
longer provides a bounded scouting function, it loses much of its value.

## 5. Exportability is mandatory

Discovered primitive motifs and search results should remain exportable and
versioned. Otherwise Primordia cannot contribute to the umbrella's cumulative
memory.

## 6. Comparable outside, different inside

Externally Primordia should line up with Prism, Topograph, Stratograph, and
Compare on:

- canonical benchmark IDs
- parity-pack compatibility
- export contract shape
- budget semantics
- fair-compare expectations

Internally it should still answer a distinct primitive-first research question.

## 7. Local-first MLX execution is the current reference path

Primordia should meet the umbrella's Apple-Silicon / MLX local-first standard
while keeping its own primitive-first scope. Shared benchmarks are welcome;
shared runtime dependence on sibling packages is not.

## 8. Standalone should remain possible

Primordia should be runnable on its own when given explicit benchmark data,
shared benchmark definitions, or documented pack inputs. Shared substrate is
fine. Hidden dependence on sibling package internals is not.
