# EvoNN Roadmap

## Purpose

This document turns the umbrella EvoNN vision into an operating roadmap.

The roadmap is intentionally organized around horizons instead of package-local
feature lists. The point is to grow the whole program in an order that preserves
fair comparison, local survivability, and scientific clarity.

Package-specific branch execution plans now live under `.hermes/plans/README.md`.
This roadmap remains the long-horizon umbrella sequence rather than the active
branch task list.

## North Star

EvoNN should become a local-first, benchmark-disciplined research program for
discovering reusable neural structure across multiple levels of abstraction:

- primitive computational motifs
- reusable cells and motifs
- model families
- flat graph topologies
- hierarchical graph systems
- later, broader task-system behaviors under constrained budgets

The project should be judged less by isolated leaderboard wins and more by its
ability to produce fair, repeatable evidence about which abstractions matter.

## Horizon 1: Trustworthy Umbrella

Goal:
make the comparative substrate trustworthy on local hardware.

Why first:
without this layer, higher-level search wins are hard to trust and easy to
misinterpret.

Deliverables:
- normalized benchmark ladder
- normalized budget contract
- normalized telemetry/reporting surface
- first-class inventory of which structural layers should become shared
  umbrella infrastructure versus remain package-local
- clear package roles and workspace coherence
- stronger root-level documentation of how the umbrella fits together

Primary artifacts:
- `VISION.md`
- `BENCHMARK_LADDER.md`
- `BUDGET_CONTRACT.md`
- `TELEMETRY_SPEC.md`
- `MONOREPO.md`

Success criteria:
- every major package can point to the same benchmark taxonomy
- budget terms mean the same thing across systems
- exported telemetry has a common minimum surface
- recurring infrastructure duplication is identified and ranked instead of being
  allowed to drift silently
- monorepo docs reflect the actual umbrella rather than a partial subset

### Horizon 1A: Structural Unification Of Shared Infrastructure

Goal:
reduce duplicated research plumbing across packages without collapsing distinct
search bets into one runtime.

Why here:
shared substrate work belongs with the trustworthy-umbrella horizon because it
directly affects comparability, maintenance cost, and parity velocity.

Primary candidates for unification:
- benchmark/parity-pack resolution helpers beneath package-local benchmark APIs
- Symbiosis export-core helpers and compare-facing summary assembly
- report-generation primitives such as markdown escaping and failure-pattern
  rendering
- run-storage primitives and common metadata tables
- typed telemetry/budget/seeding metadata models and validators
- CLI helper conventions for recurring inspect/export surfaces

Keep separate by design:
- genomes and candidate representations
- mutation/crossover logic
- compiler/runtime internals
- package-local search coordinators
- abstraction-specific telemetry above the shared minimum contract

Near-term ranking:
1. shared benchmark/parity loader layer
2. shared export/manifest/summary helpers
3. shared report rendering helpers
4. shared storage substrate
5. shared telemetry/budget/seeding models
6. shared CLI support helpers
7. shared package scaffolding conventions

Success criteria:
- the repo names the best shared-infrastructure candidates explicitly
- future umbrella work can target shared substrate intentionally rather than by
  ad hoc copy-porting
- package identity is preserved while duplicated compare-facing plumbing starts
  shrinking

## Horizon 2: Primitive-First Search

Goal:
add the missing bottom-up search layer beneath architecture-scale systems.

Why second:
EvoNN currently has architecture-scale bets, but not a clean primitive-first
search project that connects single-neuron or microcircuit evolution to the
rest of the stack.

Deliverables:
- a dedicated primitive-first package
- explicit primitive and microcircuit search thesis
- cheap benchmark lane for low-level motif discovery
- motif export format and upstream integration plan

Primary artifacts:
- `EvoNN-Primordia/README.md`
- `EvoNN-Primordia/VISION.md`
- `.hermes/plans/2026-04-26_211820-primordia-quality-parity-plan.md`

Success criteria:
- primitive-first search is a first-class umbrella concept, not an afterthought
- the repo has a named place for low-level motif research
- the role of primitive priors relative to Prism, Topograph, and Stratograph is
  explicit

## Horizon 3: Transfer And Cumulative Search

Goal:
stop starting from near-zero every time.

Key ideas:
- motif memory
- archive reuse
- benchmark-family priors
- transfer-aware seeding
- cross-system evidence reuse when fair

Likely artifacts:
- motif bank format
- archive compatibility rules
- seed provenance schema
- transfer audit reporting

Success criteria:
- repeated runs demonstrably benefit from prior evidence
- reused structure is tracked and auditable
- transfer improves search efficiency without contaminating fair comparison

## Horizon 4: Harder Benchmark Classes

Goal:
move toward increasingly difficult benchmark families without sacrificing local
survivability or comparison discipline.

Examples of target benchmark classes:
- harder multi-domain packs
- longer-horizon sequence tasks
- code-like tasks
- system-like tasks with staged evaluation
- frontier-style tasks in reduced or carefully budgeted forms

Important constraint:
this horizon is not "become an LLM project." The target is broader: any hard
benchmark class where neural systems may still discover surprising, competitive,
or unconventional solutions.

Success criteria:
- benchmark progression remains staged and honest
- local hardware can still run meaningful reduced experiments
- hard-task results are reported with explicit limits and budget disclosures

## Cross-Horizon Operating Rules

### 1. Local-first is a design constraint

The M1 Max / Apple Silicon use case is not incidental. It shapes what counts as
a good search system:
- bounded memory
- bounded worker count
- staged fidelity
- resumable execution
- overnight-safe run profiles
- compact, inspectable artifacts

### 2. Hard benchmarks are a north star, not an excuse for chaos

EvoNN should aim high, but the project should grow through a difficulty ladder
rather than leap straight to extremely expensive or noisy evaluations.

### 3. Distinct systems stay distinct until evidence justifies merger

Prism, Topograph, Stratograph, and Primordia should remain scientifically useful
as separate bets. Merger is an outcome of evidence, not a default coding style.

### 4. Fair comparison outranks wishful thinking

If a system loses under shared packs and normalized budgets, that is useful
knowledge.

### 5. Reuse must be auditable

When lower-level discoveries seed later runs, that transfer should be visible in
artifacts and reports.

## Near-Term Priorities

For the current 90-day operating window, use `EVONN_90_DAY_PLAN.md` as the
execution document beneath this roadmap.

Priority 1:
make `tier1_core` the trusted daily research lane, with repeatable compare
artifacts and explicit operating status at budgets `64`, `256`, and `1000`,
using the state terms defined in `EVONN_90_DAY_PLAN.md` rather than a binary
"trusted/not trusted" label.

Priority 2:
define and enforce project-wide budget/accounting semantics so fair comparison
does not depend on per-engine interpretation of evaluations, reuse, caching,
failures, or resumed runs.

Priority 3:
make structured trend artifacts the default decision surface for whether a
change improved anything.

Priority 4:
harden the trust layer in CI, especially `EvoNN-Shared`, `EvoNN-Compare`,
`EvoNN-Contenders`, `EvoNN-Primordia`, and `EvoNN-Stratograph`, while keeping
macOS-native MLX validation explicit for Prism and Topograph.

Priority 5:
strengthen the contender floor and execute one auditable transfer/seeding loop
so the umbrella starts compounding evidence rather than only accumulating
infrastructure.

## Long-Term End State

If the roadmap succeeds, EvoNN becomes:
- a comparative research discipline for architecture search
- a local-first discovery stack rather than a pile of experiments
- a source of trustworthy benchmark artifacts and fair comparisons
- a cumulative system where lower-level discoveries can improve higher-level
  search without destroying scientific honesty

That is the intended umbrella shape.
