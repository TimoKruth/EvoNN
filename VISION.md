# EvoNN Vision

## Purpose

EvoNN exists to answer one broad question:

how should neural architectures be discovered, compared, and improved when we
care about real search abstractions, real benchmark breadth, real budget
discipline, and real local iteration?

The whole Evo Neural Nets idea is bigger than any one package in this
directory.

It is an umbrella research program that combines:

- distinct architecture-search systems
- strong fixed baselines
- shared benchmark contracts
- common export and comparison formats
- local-first execution assumptions, especially on Apple Silicon
- evidence loops that decide which ideas deserve to survive

So EvoNN is not just one search engine and not just one leaderboard effort. It
is a coordinated system with two foundation layers:

- `Compare`: the protocol-first comparison layer that makes results from distinct systems fair, reproducible, and inspectable
- `Contenders`: the strong fixed-baseline layer that prevents evolutionary systems from winning by default against weak references

On top of that foundation sit three distinct search bets:

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
4. Exports, manifests, reports, and budget accounting matter as much as raw
   scores.
5. Local and bounded experimentation is strategically valuable because it makes
   iteration, reproduction, and comparison practical.

Put differently:

EvoNN is trying to turn architecture search from a collection of isolated model
hunts into a comparative discipline.

That means the project should tell us not only what scored highest, but also:

- what kind of search abstraction produced the result
- what budget it consumed
- what benchmarks it really covered
- what evidence survives outside the repo that produced it
- whether the result still matters against strong non-evolutionary baselines

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

### Local-First Research Loops

EvoNN consistently leans toward local, bounded, iteration-friendly research
loops rather than assuming cluster-first experimentation.

That bias shows up strongly in Prism and Topograph's Apple Silicon / MLX
orientation, and it also appears in adjacent experiments like
`autoresearch-mlx`, which explore fixed-budget autonomous search loops on local
hardware.

This matters because the umbrella idea is not only "search more." It is "learn
faster under constraints, with evidence that can be rerun and compared."

See:
- [autoresearch-mlx/README.md](./autoresearch-mlx/README.md)

## The Foundation: Compare

`EvoNN-Compare` is the trust layer of the project.

Its job is to ingest exports from Prism, Topograph, Stratograph, Hybrid, and
fixed contenders, then turn them into fair side-by-side comparisons using
shared packs, shared benchmark identities, shared budget semantics, and common
reporting artifacts.

In practice, Compare should become the place where EvoNN answers questions like:

- which system actually won under the same benchmark pack and budget?
- how broad was that win across task types?
- what evidence survives outside the source repo of the system that produced it?
- which results are robust enough to trust, reproduce, and challenge?

Without Compare, EvoNN is just a collection of interesting codebases. With
Compare, it becomes a comparative research program with a memory.

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
- eventually adapters that let sibling EvoNN systems participate as normalized
  contender-style references under fixed mini-budgets

Its purpose is not to become a second general framework. Its purpose is to make
the rest of EvoNN honest, especially when evolutionary systems are tempted to
compare themselves only against weak references or incomparable prior runs.

See:
- [EvoNN-Contenders/README.md](./EvoNN-Contenders/README.md)
- [CONTENDER_EXPANSION_PLAN.md](./CONTENDER_EXPANSION_PLAN.md)

## The Three Search Bets

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
3. `Prism`, `Topograph`, and `Stratograph` run distinct search programs.
4. Each system exports comparable artifacts.
5. `Compare` evaluates them under shared packs and fair budget contracts.
6. The project learns which ideas deserve to survive into later merged systems.

That means the umbrella project is not centered on one repo. It is centered on
an evaluation loop:

- define a fair benchmark surface
- propose a search abstraction
- run it on shared packs
- export reproducible evidence
- compare it against strong peers and contenders
- keep what survives honest comparison

That loop is the whole EvoNN idea in operational form.

## Strategic Direction

Long run, EvoNN should become:

- a local-first architecture-discovery program, not a pile of disconnected
  experiments
- a benchmark-disciplined ecosystem for architecture search research
- a place where family, topology, and hierarchy are tested as separate
  explanatory axes
- a source of reproducible artifacts rather than isolated claims
- a bridge between evolutionary systems and strong non-evolutionary baselines
- a foundation for future merged systems that inherit only proven advantages

The merged future matters, but only later.

First, the current systems need to stay distinct enough to answer real
questions:

- when does family choice dominate?
- when does flat topology dominate?
- when does hierarchy dominate?
- what survives comparison against strong contenders?
- what remains true under shared benchmark packs and normalized budgets?

## What Success Looks Like

EvoNN succeeds if it produces:

- fair comparisons that people can actually trust
- strong baseline coverage that rules out easy self-deception
- clear evidence about the strengths and limits of Prism, Topograph, and
  Stratograph
- reusable exports, reports, and benchmark contracts
- a path toward a later combined system built from proven wins instead of
  intuition alone

## Final Statement

EvoNN is not one NAS thesis and not one implementation strategy.

It is a full research idea:

- shared benchmarks to define the playing field
- contenders to keep the field honest
- compare to make claims testable
- Prism to test family-first search
- Topograph to test flat-topology-first search
- Stratograph to test hierarchy-first search
- local-first execution so iteration stays fast, reproducible, and accessible

The point of the whole project is to discover which architecture-search ideas
deserve to compound into the next generation of systems, and to do that with
enough discipline that the answer can be trusted.
