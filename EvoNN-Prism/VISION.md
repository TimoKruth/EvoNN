# Prism Vision

## Purpose

Prism exists to explore a simple but powerful idea:

neural architecture search should not be one narrow optimizer over one narrow model class.

It should be a family-based search system that can discover, compare, and refine many kinds of models across many kinds of tasks, while staying fast enough to run locally and structured enough to compare fairly with other systems.

Today, Prism already points in that direction:

- it searches across multiple model families instead of one fixed template
- it runs on MLX and leans into Apple Silicon as a serious local research target
- it spans tabular, image, sequence, and language-modeling workloads
- it tracks runs, archives elites, checkpoints state, and exports canonical comparison artifacts
- it treats benchmark parity and cross-system comparability as first-class concerns

This document turns those ingredients into a long-run north star.

## Long-Run Thesis

Prism should become a local-first evolutionary model discovery engine.

Not just "NAS for one benchmark."

Not just "a collection of model classes."

Not just "another training script."

The long-run goal is a system that can:

- search over diverse architectural families
- adapt search pressure to what a benchmark suite actually needs
- preserve and reuse useful structure across generations
- produce evidence-rich artifacts, not only scores
- compare cleanly against sibling systems through shared benchmark contracts
- run serious experiments on commodity Apple hardware, not only large clusters

In strongest form, Prism becomes a machine for building model lineages, not just isolated winners.

## Core Beliefs

### 1. Family diversity matters

Good search should move between representational families, not only tune widths and depths inside one family.

Prism already encodes this belief through family-aware genomes, compatibility checks, niche archives, and seed populations spanning multiple families. Long run, this should deepen into a search culture where architectural diversity is treated as a source of strength, robustness, and transfer.

### 2. Benchmarks are not side input

Benchmarks should shape the search itself.

Prism already hints at this through undercovered benchmark focus, per-benchmark elites, Pareto structure, and canonical benchmark IDs. Long run, benchmark coverage should become an active curriculum signal that decides where compute goes next, what families deserve more search budget, and when a system is truly general instead of merely overfit to one slice.

### 3. Efficiency is a feature, not compromise

Weight inheritance, multi-fidelity evaluation, compact archives, and local execution are not temporary shortcuts. They are part of the product.

The point is not only to find strong models. The point is to make iterative model discovery practical enough that one machine can support real cycles of hypothesis, search, inspection, export, and comparison.

### 4. Reproducibility must survive contact with reality

A strong system does not stop at "best score."

It stores config, lineage, metrics, summaries, reports, and canonical export artifacts. Long run, every Prism run should be understandable, resumable, comparable, and auditable. Results should be easy to reproduce and easy to challenge.

### 5. Interoperability creates leverage

Prism is strongest when it can stand beside EvoNN, EvoNN-2, Topograph, and future systems inside a shared evaluation layer.

The symbiosis export path is not an accessory. It is a signal that Prism should participate in a larger ecosystem of comparable search systems, shared packs, canonical IDs, and common evidence formats.

## What Prism Should Become

### A serious family-based search runtime

Prism should grow from a promising pipeline into a robust search runtime with:

- richer family definitions
- cleaner modality/task constraints
- better crossover and mutation semantics
- adaptive parent selection and archive pressure
- stronger resume, recovery, and experiment management

The runtime should support long searches without becoming opaque.

### A benchmark-aware generalization engine

Prism should optimize not only for local wins, but for breadth.

Long run, Prism should learn how architectures behave across suites, domains, and modality boundaries. It should know which families transfer, which collapse, which overfit, and which improve when exposed to diverse packs. That makes it more than NAS. That makes it a generalization engine.

### A local-first research workbench

Apple Silicon is not merely a deployment target. It is part of the strategy.

Prism should aim to be one of the best ways to do architecture search and model-family experimentation on-device:

- fast startup
- compact runs
- low operational overhead
- reliable reports
- strong observability
- reproducible artifacts

The ideal user can run meaningful searches from a laptop, not only from infrastructure.

### A model lineage system

Current ML tooling often treats models as disposable endpoints.

Prism should instead treat them as evolving lineages with ancestry, niches, and survival pressure. A run should reveal:

- what survived
- why it survived
- where it failed
- which families dominated which tasks
- which mutations consistently improved quality
- which structures transferred across benchmarks

This is a shift from "best checkpoint" thinking to "evolutionary evidence" thinking.

### A bridge layer across search ecosystems

Long run, Prism should speak a common language with external systems:

- shared packs
- shared benchmark identities
- compatible manifests
- comparable budget accounting
- exportable reports

If two systems cannot be compared honestly, progress is hard to trust. Prism should help fix that.

## Long-Horizon Capabilities

If Prism fully matures, it should be able to support work like this:

### Cross-suite search

One run optimizes against an entire pack, not one dataset, while preserving benchmark-level visibility and fairness.

### Open-ended family expansion

New families can be added without breaking the search model. The search space grows by adding principled building blocks, not by rewriting the engine.

### Transfer-aware evolution

Search decisions use prior evidence from related benchmarks, earlier runs, and historical lineages.

### Budget-aware research loops

Prism can answer questions like:

- what did we learn per unit of compute?
- which families are worth deeper evaluation?
- which benchmarks need more pressure?
- when is a cheap proxy good enough?

### Canonical comparative studies

Prism can produce exports and summaries good enough for side-by-side comparison with sibling systems at scale.

### Auto-curated benchmark packs

Long run, Prism should help build and refine benchmark packs themselves: balanced, coverage-aware, versioned, and explicit about what kind of generalization they test.

### Search over architecture plus process

Eventually, search should not stop at architecture. It should reach into:

- training policy
- preprocessing choices
- regularization schemes
- curriculum
- evaluation budget allocation

Architecture search becomes system search.

## What Prism Should Avoid

Vision also means constraint.

Prism should avoid becoming:

- a benchmark-specific bag of hacks
- a monolithic framework that is hard to inspect
- a score-chasing tool with weak evidence trails
- a cluster-first system that loses local usability
- a closed format that cannot compare fairly with peers
- a single-family optimizer pretending to be general NAS

If a future shortcut improves one leaderboard while weakening clarity, portability, or comparability, it should be treated with suspicion.

## Strategic Principles

### Keep search visible

Runs should expose lineage, archives, benchmark coverage, failure reasons, and family dynamics clearly enough that users can reason about behavior.

### Prefer explicit contracts

Benchmark IDs, pack semantics, manifest structure, and budget accounting should stay explicit and versioned.

### Grow breadth without losing rigor

Adding more families, tasks, or packs only helps if evaluation remains comparable and artifacts remain interpretable.

### Make local experimentation first-class

A laptop-scale run that teaches something real is better than a cluster-scale run that is hard to reproduce.

### Build for ecosystem value

Prism should not only win inside its own repo. It should make comparative research better across related systems.

## Long-Run Destination

In the really long run, Prism should become:

an evolutionary operating system for model-family discovery on local hardware.

That means:

- broad model-family search
- benchmark-aware compute allocation
- lineage-level memory
- reproducible experiment artifacts
- strong local ergonomics
- honest cross-system comparison

If Prism reaches that state, it will be more than a NAS project.

It will be a durable engine for discovering which kinds of models deserve to exist, under which constraints, on which benchmarks, and with what evidence.
