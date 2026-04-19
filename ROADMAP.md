# EvoNN Roadmap

## Purpose

This document turns the umbrella EvoNN vision into an operating roadmap.

The roadmap is intentionally organized around horizons instead of package-local
feature lists. The point is to grow the whole program in an order that preserves
fair comparison, local survivability, and scientific clarity.

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
- monorepo docs reflect the actual umbrella rather than a partial subset

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
- `EvoNN-Primordia/IMPLEMENTATION_PLAN.md`

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

Priority 1:
finish and enforce Horizon 1 documents across the repo.

Priority 2:
establish Primordia as the missing Horizon 2 package.

Priority 3:
thread the benchmark ladder and budget contract into package docs and CLIs.

Priority 4:
start defining how primitive motifs are exported, versioned, and consumed by
higher-level systems.

## Long-Term End State

If the roadmap succeeds, EvoNN becomes:
- a comparative research discipline for architecture search
- a local-first discovery stack rather than a pile of experiments
- a source of trustworthy benchmark artifacts and fair comparisons
- a cumulative system where lower-level discoveries can improve higher-level
  search without destroying scientific honesty

That is the intended umbrella shape.