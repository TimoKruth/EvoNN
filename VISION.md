# EvoNN Vision

## Purpose

EvoNN exists to answer one broad question:

how should neural architectures be discovered, compared, improved, and carried
forward when we care about real search abstractions, real benchmark breadth,
real budget discipline, and real local iteration?

The whole Evo Neural Nets idea is bigger than any one package in this
repository.

It is an umbrella research program that combines:

- distinct architecture-search systems
- a primitive-first search layer beneath architecture-scale systems
- strong fixed baselines
- shared benchmark contracts
- common export and comparison formats
- local-first execution assumptions, especially on Apple Silicon
- evidence loops that decide which ideas deserve to survive

So EvoNN is not just one search engine and not just one leaderboard effort. It
is a coordinated research stack with three base layers and four search bets.

Foundation layers:

- `shared-benchmarks`: common benchmark identities, packs, and task metadata
- `Compare`: the protocol-first comparison layer that makes results from distinct systems fair, reproducible, and inspectable
- `Contenders`: the strong fixed-baseline layer that prevents evolutionary systems from winning by default against weak references

Search layers:

- `Primordia`: primitive-first search over low-level computational motifs
- `Prism`: family-first search
- `Topograph`: flat-topology-first search
- `Stratograph`: hierarchy-first search

The long-run goal is not to keep collecting disconnected experiments. The goal
is to learn which search abstractions actually matter, under fair budgets,
shared benchmark contracts, and bounded local execution, and then use that
evidence to shape stronger future systems.

## The Whole EvoNN Thesis

EvoNN is organized around a few simple claims:

1. Neural architecture search should be evaluated across shared benchmark packs,
   not one-off demos.
2. Different search abstractions should stay genuinely distinct long enough to
   teach us something real.
3. Baselines must be strong enough that evolutionary wins mean something.
4. Exports, manifests, reports, telemetry, and budget accounting matter as much
   as raw scores.
5. Local and bounded experimentation is strategically valuable because it makes
   iteration, reproduction, and comparison practical.
6. Structure discovered at lower levels should be allowed to seed higher levels
   rather than being discarded between experiments.
7. Hard benchmarks are an important north star, but EvoNN is not restricted to
   language models or coding tasks; the real target is any benchmark surface
   where neural systems might still discover surprising, competitive solutions.

Put differently:

EvoNN is trying to turn architecture search from a collection of isolated model
hunts into a comparative discipline with memory.

That means the project should tell us not only what scored highest, but also:

- what kind of search abstraction produced the result
- what budget it consumed
- what benchmarks it really covered
- what telemetry and artifacts survive outside the source run
- whether the result still matters against strong non-evolutionary baselines
- whether the result can seed stronger future searches instead of dying as an
  isolated checkpoint

## Shared Substrate

The whole program depends on a common substrate that sits beneath the named
search systems.

### Shared Benchmarks

`shared-benchmarks/` is the benchmark source of truth for the umbrella project.
It provides the common catalog and suite definitions that make cross-system
comparison meaningful in the first place.

Without shared benchmark identities and pack semantics, EvoNN collapses into
parallel repos talking past each other.

See:
- [shared-benchmarks/README.md](./shared-benchmarks/README.md)
- [BENCHMARK_LADDER.md](./BENCHMARK_LADDER.md)

### Local-First Research Loops

EvoNN consistently leans toward local, bounded, iteration-friendly research
loops rather than assuming cluster-first experimentation.

That bias shows up strongly in Prism and Topograph's Apple Silicon / MLX
orientation, and it should shape the rest of the umbrella as well. The point is
not only to search more. The point is to learn faster under constraints, with
evidence that can be rerun and compared.

### Budget And Telemetry Discipline

A fair umbrella project needs normalized budget semantics and a common telemetry
surface. Runs should declare their constraints and report them in comparable
ways.

See:
- [BUDGET_CONTRACT.md](./BUDGET_CONTRACT.md)
- [TELEMETRY_SPEC.md](./TELEMETRY_SPEC.md)

## The Foundation: Compare

`EvoNN-Compare` is the trust layer of the project.

Its job is to ingest exports from Primordia, Prism, Topograph, Stratograph,
future merged systems, and fixed contenders, then turn them into fair
side-by-side comparisons using shared packs, shared benchmark identities,
shared budget semantics, and common reporting artifacts.

In practice, Compare should become the place where EvoNN answers questions like:

- which system actually won under the same benchmark pack and budget?
- how broad was that win across task types?
- what evidence survives outside the source repo of the system that produced it?
- which results are robust enough to trust, reproduce, and challenge?
- what lower-level priors actually improved later systems enough to matter?

Without Compare, EvoNN is just a collection of interesting codebases. With
Compare, it becomes a comparative research program with memory.

See:
- [EvoNN-Compare/README.md](./EvoNN-Compare/README.md)

## The Foundation: Contenders

`EvoNN-Contenders` is the baseline discipline layer.

Its role is to maintain a configurable zoo of strong non-evolutionary models and
export their results in the same comparison-friendly shape used elsewhere. That
keeps the project grounded: if an evolutionary system cannot beat serious fixed
baselines on shared packs, the result is informative even when it is
disappointing.

Contenders should grow into a broad but controlled benchmark opponent set:

- classical tabular baselines
- boosted-tree baselines
- image baselines
- lightweight language-model baselines
- sequence baselines
- future task-specific baselines for harder benchmark classes
- eventually adapters that let sibling EvoNN systems participate as normalized
  contender-style references under fixed mini-budgets

Its purpose is not to become a second general framework. Its purpose is to make
the rest of EvoNN honest, especially when evolutionary systems are tempted to
compare themselves only against weak references or incomparable prior runs.

See:
- [EvoNN-Contenders/README.md](./EvoNN-Contenders/README.md)
- [CONTENDER_EXPANSION_PLAN.md](./CONTENDER_EXPANSION_PLAN.md)

## The Four Search Bets

### Primordia

`Primordia` is the primitive-first system.

It asks: which low-level computational motifs deserve to exist before they are
assembled into families, topologies, or hierarchical cell systems? Primordia is
the bridge between single-neuron or tiny-circuit evolution and the larger
search systems above it.

Its job is not to become a giant end-to-end benchmark monster immediately. Its
job is to discover reusable low-level structure cheaply enough that the rest of
EvoNN can benefit from it.

Read more:
- [EvoNN-Primordia/README.md](./EvoNN-Primordia/README.md)
- [EvoNN-Primordia/VISION.md](./EvoNN-Primordia/VISION.md)

### Prism

`Prism` is the family-first system.

It asks: which model family should solve this task, and how should that family
be parameterized? Prism searches across curated architectural families rather
than inventing arbitrary graphs from scratch. That makes it the cleanest test of
whether family choice plus disciplined family-level evolution is already enough
to produce strong, general, local-first results.

Prism is the most direct path to broad family-aware search on Apple Silicon, and
it sets the reference point for model-family discovery inside EvoNN.

Read more:
- [EvoNN-Prism/README.md](./EvoNN-Prism/README.md)
- [EvoNN-Prism/VISION.md](./EvoNN-Prism/VISION.md)

### Topograph

`Topograph` is the flat-topology-first system.

It asks: which explicit graph structure should exist under real hardware and
deployment constraints? Topograph treats topology as the primary search object,
not a side effect of family choice. It is the test bed for graph evolution,
operator-level search, quality-diversity archives, and hardware-aware scoring.

Topograph is where EvoNN explores whether graph structure itself can become a
deployable, device-aware source of advantage.

Read more:
- [EvoNN-Topograph/VISION.md](./EvoNN-Topograph/VISION.md)

### Stratograph

`Stratograph` is the hierarchy-first system.

It asks: which reusable structures should repeat, where should they specialize,
and how should a macro graph coordinate them? Instead of evolving one flat DAG,
Stratograph evolves a macro graph over a library of reusable cells. It is the
bet that hierarchy, motif reuse, and controlled specialization can produce
better search efficiency, transfer, and interpretability than flat search
alone.

Stratograph exists as a separate project because hierarchy should be tested as a
distinct scientific claim, not hidden as an extension of Topograph.

Read more:
- [EvoNN-Stratograph/README.md](./EvoNN-Stratograph/README.md)
- [EvoNN-Stratograph/VISION.md](./EvoNN-Stratograph/VISION.md)

## How The Pieces Fit Together

The intended stack looks like this:

1. `shared-benchmarks` defines common benchmark identities and packs.
2. `Contenders` establishes strong fixed baselines.
3. `Primordia` discovers cheap low-level motifs and primitive priors.
4. `Prism`, `Topograph`, and `Stratograph` run distinct architecture-scale
   search programs.
5. Each system exports comparable artifacts.
6. `Compare` evaluates them under shared packs and fair budget contracts.
7. The project learns which ideas deserve to survive into later merged systems.

## Seeding Ladders As Research Objects

EvoNN should treat transfer and seeding policy as a first-class research topic,
not only as a convenience feature.

There are at least two legitimate inheritance ladders the umbrella should test.

### Ladder A: Direct Primitive Seeding

- `Primordia -> Stratograph`
- `Primordia -> Topograph`
- `Primordia -> Prism`

This ladder tests whether low-level primitive priors are already rich enough to
improve all higher systems directly.

Questions this ladder answers:
- do primitive motifs help every higher abstraction immediately?
- which packages benefit from direct low-level priors and which do not?
- does direct seeding accelerate search enough to justify the extra coupling?

### Ladder B: Staged Seeding

- `Primordia -> Stratograph`
- `Stratograph -> Topograph`
- `Topograph -> Prism`

This ladder tests whether discoveries should be translated upward through each
successive abstraction instead of being injected everywhere directly.

Questions this ladder answers:
- does hierarchy form the right first consumer of primitive motifs?
- does topology benefit more from hierarchical priors than from raw primitive
  priors?
- does Prism become stronger when it inherits already-structured topology-level
  knowledge instead of primitive-level knowledge?

### Umbrella Policy

EvoNN should not assume one ladder is correct in advance.

Instead, the project should:
- implement both ladders explicitly
- label runs with the seeding ladder they use
- keep non-seeded baselines available
- compare direct, staged, and no-seeding regimes under shared benchmark packs
  and budget contracts
- preserve enough package distinctness that gains can still be attributed to the
  target architecture rather than hidden coupling

This is important scientifically. If direct primitive seeding helps everything,
that is evidence. If staged inheritance wins, that is also evidence. The project
should measure the difference instead of settling the question rhetorically.

That means the umbrella project is not centered on one repo. It is centered on
an evaluation loop:

- define a fair benchmark surface
- define a bounded budget contract
- propose a search abstraction
- run it on shared packs
- export reproducible evidence
- compare it against strong peers and contenders
- keep what survives honest comparison
- carry reusable structure upward when the evidence justifies it

That loop is the whole EvoNN idea in operational form.

## Horizon Structure

EvoNN should grow in ordered horizons rather than by uncontrolled sprawl.

### Horizon 1: Trustworthy umbrella

Goal:
make the comparative substrate trustworthy on local hardware.

Focus:
- shared benchmark ladder
- normalized budget contracts
- common telemetry/reporting surface
- workspace coherence across umbrella packages
- stronger documentation of roles and boundaries
- a repeatable automation loop that can pull latest work, run real smoke tests after bounded improvements, and record whether tiny budget runs improve or regress over time
- a Linux-capable execution path that can run full compare-grade validation instead of leaving reproducibility tied to MLX-only environments

Near-future expectation:
EvoNN should become stable enough that the routine architecture-parity automation can do more than inspect docs and make tiny edits. It should be able to pull the latest branch state, run package-relevant smoke tests after each bounded improvement, and where practical run small matched-budget validation studies that show whether recent changes actually help, hurt, or leave results unchanged.

That means the umbrella should steadily move toward:
- smoke-test coverage that is real, fast, and runnable on the automation host
- tiny-budget benchmark configs that are intentionally designed for trend detection rather than headline performance
- package docs and vision files that explicitly state which verification path is expected after small improvements
- compare/report surfaces that can accumulate these small-budget validation results over time instead of treating every improvement as anecdotal
- backend/runtime separation strong enough that a Linux host can execute full compare-style validation runs even when MLX is unavailable
- portability goals defined in terms of shared search semantics and comparable evidence, not fantasy promises of bitwise-identical floating-point results across platforms

### Horizon 2: Primitive-first search

Goal:
add the missing bottom-up search layer beneath architecture-scale systems.

Focus:
- primitive and microcircuit evolution
- cheap benchmark packs for motif discovery
- motif export formats that can seed later systems
- explicit boundary between low-level discovery and higher-level search

### Horizon 3: Transfer and cumulative search

Goal:
stop starting from near-zero every time.

Focus:
- motif memory
- archive reuse
- benchmark-family priors
- transfer-aware seeding across systems
- explicit comparison of direct primitive seeding versus staged seeding ladders
- run labeling so transfer policy remains auditable in Compare

### Horizon 3.5: Portable compare execution

Goal:
make EvoNN capable of running serious compare-style validation on Linux as well as Apple-first local hardware.

Focus:
- backend/runtime separation so search semantics survive beyond MLX-only execution
- Linux-capable smoke, regression, and small-budget compare runs
- explicit runtime metadata that distinguishes backend, device class, precision mode, and worker topology
- portable evaluation paths that can validate the same benchmark packs and export contracts on Linux even when performance differs
- reproducibility defined as comparable evidence under shared budgets, not strict numeric identity across runtimes

### Horizon 4: Harder benchmark classes

Goal:
move toward increasingly difficult and surprising benchmark families without
sacrificing fairness or local survivability.

Focus:
- stronger multi-domain packs
- harder sequence and code-like tasks
- system-level evaluation surfaces
- eventually frontier-style tasks under reduced or staged evaluation budgets

See:
- [ROADMAP.md](./ROADMAP.md)

## Strategic Direction

Long run, EvoNN should become:

- a local-first architecture-discovery program, not a pile of disconnected
  experiments
- a benchmark-disciplined ecosystem for architecture search research
- a place where primitive structure, family choice, topology, and hierarchy are
  tested as separate explanatory axes
- a source of reproducible artifacts rather than isolated claims
- a bridge between evolutionary systems and strong non-evolutionary baselines
- a foundation for future merged systems that inherit only proven advantages

The merged future matters, but only later.

First, the current systems need to stay distinct enough to answer real
questions:

- when do primitive motifs matter?
- when does family choice dominate?
- when does flat topology dominate?
- when does hierarchy dominate?
- what survives comparison against strong contenders?
- what remains true under shared benchmark packs and normalized budgets?
- what discovered structure transfers across tasks instead of overfitting one
  run?

## What Success Looks Like

EvoNN succeeds if it produces:

- fair comparisons that people can actually trust
- strong baseline coverage that rules out easy self-deception
- clear evidence about the strengths and limits of Primordia, Prism,
  Topograph, and Stratograph
- reusable exports, reports, benchmark contracts, and telemetry surfaces
- a path toward a later combined system built from proven wins instead of
  intuition alone
- evidence that lower-level search can produce structure worth carrying upward

## Final Statement

EvoNN is not one NAS thesis and not one implementation strategy.

It is a full research idea:

- shared benchmarks to define the playing field
- contenders to keep the field honest
- compare to make claims testable
- Primordia to test primitive-first search
- Prism to test family-first search
- Topograph to test flat-topology-first search
- Stratograph to test hierarchy-first search
- local-first execution so iteration stays fast, reproducible, and accessible

The point of the whole project is to discover which architecture-search ideas
deserve to compound into the next generation of systems, and to do that with
enough discipline that the answer can be trusted.
