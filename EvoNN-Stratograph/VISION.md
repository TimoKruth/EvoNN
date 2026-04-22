# Stratograph Vision

## Thesis

Stratograph is third distinct architecture-search idea in EvoNN family.

- Prism asks: which model family should solve task?
- Topograph asks: which flat topology should solve task?
- Stratograph asks: which hierarchical structure should solve task?

Core wager:

Many strong neural systems are not best understood as flat sequence of layers or
single flat DAG. They are better understood as **graph of reusable motifs**.
Some motifs stay shared. Some split and specialize. Some become routing hubs.
Some stay local computation units. Some repeat across tasks. Some repeat only
inside one family of problems.

Stratograph makes that structure explicit, evolvable, measurable, exportable,
and eventually trainable at full maturity.

This is not “Topograph but deeper”.
This is “hierarchy as primary search object”.

## Why Stratograph Exists

Stratograph exists for one reason:

To test whether **hierarchy itself** is a source of search efficiency,
generalization, modularity, and transfer.

Not parameter count alone.
Not depth alone.
Not wider search space alone.
Not implementation convenience.

If hierarchy matters, we want to know:

- when it helps
- why it helps
- how much it helps
- which tasks reward reuse
- which tasks reward specialization
- which motifs keep reappearing
- which hierarchy patterns are wasteful

If hierarchy does not matter, we want to know that too.

Distinct project is how we get honest answer.

## Core Model

Each candidate architecture has two coupled levels.

### 1. Macro Graph

- DAG of cell instances
- controls routing, skips, ordering, repetition, fan-in, fan-out
- determines global information flow
- decides where reuse pressure and specialization pressure happen

### 2. Cell Library

- set of reusable cell programs
- each cell is micro-graph of primitives, projections, merges, gates
- multiple macro nodes may point to same cell
- evolution may clone, rebind, merge, resize, or specialize cells

So candidate is not “list of layers”.
Candidate is:

- topology over cells
- plus library of motifs
- plus relationships between sharing and divergence

This enables search over:

- reusable motifs
- motif families
- local specialization
- hierarchical depth
- motif re-entry across graph
- modular architectures
- task-conditioned reuse patterns

## What “Winning” Means

Stratograph should not be judged only by benchmark score.

True win for Stratograph is any combination of:

- better score at same budget
- better score per evaluation
- better parameter efficiency
- repeated discovery of useful motifs
- more interpretable architecture reuse
- cleaner transfer from one benchmark family to another
- evidence that hierarchy compresses search into better building blocks

If Stratograph merely ties existing systems while producing reusable motifs and
clear structure, that can still be meaningful.

If Stratograph wins only on a few benchmark families but explains why, that is
also meaningful.

## Distinct Project Rule

Stratograph should not be a Topograph extension and should not share its core
search runtime.

Reason:

Later merger is only scientifically useful if source systems stay genuinely
distinct first.

Comparison value comes from isolation:

- separate genome assumptions
- separate compiler assumptions
- separate mutation logic
- separate crossover logic
- separate training/runtime choices
- separate telemetry
- separate failure modes
- separate wins

Shared boundary is acceptable:

- benchmark catalogs
- parity packs
- LM cache datasets
- export contract
- compare ingestion expectations

Shared core is not.

## Design Principles

### 1. Hierarchy First

Hierarchy is not hidden inside convenience field like `internal_layers`.

Hierarchy must be explicit in:

- genome
- compiler
- mutation
- crossover
- telemetry
- inspection
- export
- reporting

### 2. Reuse And Specialization Both Matter

Stratograph must support:

- one shared cell reused many times
- clone of shared cell for local adaptation
- gradual specialization after clone
- comparison of shared vs specialized descendants
- measurement of reuse pressure as search signal

### 3. Motifs Are Real Search Objects

Cell should not be treated as anonymous nested block.

Cell should become:

- reusable motif
- lineage object
- measurable unit
- target for mutation and crossover
- candidate for motif mining across winners

### 4. Comparable Outside, Different Inside

Externally, Stratograph should satisfy same comparison surface as Prism and
Topograph:

- same benchmark packs
- same parity validation
- same export contract shape
- same budget semantics
- same fair-compare constraints

Internally, it must remain different enough that results teach us something.

### 5. Explainability Of Wins

Hierarchy-specific telemetry must tell us what happened.

Examples:

- macro depth
- average and max cell depth
- cell library size
- reuse ratio
- clone count
- specialization count
- motif frequency
- interface width profile
- novelty score
- occupied niches

## Scientific Questions

Stratograph should answer these long-run questions.

### Does Hierarchy Buy Search Efficiency?

Can two-level search find good models faster than flat search at same evaluation
budget?

### Does Reuse Buy Generalization?

Do shared motifs help on broad benchmark families by compressing useful
structure?

### Does Specialization Buy Accuracy?

Do cloned descendants outperform rigid sharing once tasks or graph regions need
local adaptation?

### Do Repeated Motifs Emerge Naturally?

If same sub-cell structures appear again and again across winners, that is
evidence hierarchy is discovering something real, not noise.

### Can Motif Libraries Transfer?

Can motif priors discovered on one benchmark family improve search on another?

### Which Seeding Ladder Works Better?

Stratograph should also answer whether it is best seeded directly from
`Primordia` or whether its strongest downstream contribution is to become the
first stage of a longer inheritance ladder.

That means comparing at least:
- unseeded Stratograph
- `Primordia -> Stratograph`
- `Primordia -> Stratograph -> Topograph`

The first comparison tests whether hierarchy is the right first consumer of
primitive motifs. The second tests whether Stratograph produces abstractions
worth passing upward to Topograph.

### Where Does Hierarchy Hurt?

We also want clear map of failure:

- wasted complexity
- unstable training
- brittle specialization
- excessive cloning
- poor latency
- memory overhead

## Near-Term Vision

Near-term, Stratograph should become credible third reference system.

That means:

- stable startup path
- compare-capable exports
- full 38-benchmark coverage
- fair budget compatibility
- enough tests to trust artifacts
- enough telemetry to inspect why results happened

This stage is about:

- standing on its own
- producing honest results
- being inspectable
- being reproducible

## Mid-Term Vision

Mid-term, Stratograph should become **motif discovery engine**.

Not only benchmark runner.

Desired capabilities:

- extract repeated winning cells
- cluster motifs by function
- compare motifs across task families
- trace motif lineage through clone/specialize events
- identify global vs local motifs
- build motif atlas of winner architectures

At this stage, output is not just “best score”.
Output is:

- architecture result
- motif result
- structural explanation result

## Long-Run Vision

Long run, Stratograph should grow into **hierarchical architecture operating
system** for EvoNN world.

Meaning:

### 1. Search Over Programs, Not Just Graphs

Cells may eventually evolve beyond simple micro-DAGs into richer internal
programs:

- conditional branches
- attention/gating templates
- sparse experts
- recurrent local controllers
- learned interface contracts

### 2. Persistent Motif Memory

Stratograph should maintain long-lived motif memory across runs:

- reusable motif bank
- motif ranking by task family
- motif transfer priors
- anti-pattern memory for motifs that repeatedly fail

### 3. Cross-Task Structural Learning

Search should eventually stop starting from near-zero every time.

Instead:

- tabular tasks seed from proven tabular motifs
- LM tasks seed from proven sequence motifs
- image tasks seed from proven spatial motifs
- mixed tasks test crossover of motif families

Stratograph should also remain explicit about two different upstream/downstream
roles in the umbrella seeding story:

- as a **direct consumer** of `Primordia` motifs in the direct ladder
- as an **intermediate translator** that turns primitive motifs into
  hierarchy-level priors for `Topograph` in the staged ladder

Both roles matter. The direct ladder tests whether hierarchy is the first
productive consumer of primitive motifs. The staged ladder tests whether
hierarchy is the right abstraction layer to refine primitive structure before it
is handed to flat topology search.

### 4. Structural Compression And Distillation

Hierarchy may allow not only discovery but compression:

- many winning architectures may collapse into few motif families
- repeated structures may be distilled into concise reusable libraries
- future systems may search in motif space first, instance space second

### 5. Self-Explaining Search

Long run, Stratograph should be able to say:

- which motifs caused improvement
- which clone event mattered
- where specialization beat sharing
- which graph region became bottleneck
- which structure repeated across winners

That would make architecture search far less black-box.

## Relationship To Future Merged System

Endgame is not “replace three systems with one and forget history”.

Endgame is:

- keep Prism as family-first reference
- keep Topograph as flat-topology reference
- keep Stratograph as hierarchy-first reference
- learn their separate wins and weaknesses
- merge only proven advantages into later combined system

Future merged system should inherit:

- from Prism: family priors and coarse model selection
- from Topograph: strong flat routing and topology search
- from Stratograph: motif reuse, specialization, hierarchical structure

But merged system is **later**.

First, Stratograph must earn its place by evidence.

## Success Criteria

Long-run success would look like this:

### Research Success

- clear ablation evidence that two-level search buys something
- repeated motif families found across winners
- measurable reuse/specialization story

### Engineering Success

- robust runtime
- mature trainer
- stable exports
- reproducible compare runs

### Strategic Success

- Stratograph contributes unique strengths to future merged system
- distinct project remains useful as reference even after merger work begins

## Final Statement

Stratograph is bet that neural architecture search should not only ask:

“which layer next?”

or

“which edge next?”

It should also ask:

“which reusable structure should exist, where should it repeat, and where
should it diverge?”

If that question turns out to matter, Stratograph becomes more than third
project.

It becomes structure lens through which whole EvoNN line can be rethought.
