# Primordia Vision

## Purpose

Primordia exists to test a simple but important idea:

before searching whole model families, flat topologies, or reusable cell
hierarchies, EvoNN should also be able to search for useful primitive
computational structure.

This is the layer closest to the user's original intuition about genetic
evolution beginning near the neuron level and growing upward.

## Core Thesis

Good higher-level architectures may depend on lower-level motifs that are rarely
hand-designed explicitly.

Those motifs might include:
- tiny gated subcircuits
- unusual merge operations
- sparse local operator patterns
- activation arrangements
- micro-ensembles that work better than a naive neuron abstraction

If such motifs exist and can be found cheaply, then EvoNN should not force every
higher-level system to rediscover them from scratch.

## What Primordia Should Test

Primordia should answer questions like:
- do primitive motifs improve downstream search efficiency?
- do some motifs transfer across benchmark families?
- which motifs recur under strong budget constraints?
- where does primitive complexity help versus hurt?
- can cheap low-level search discover building blocks worth preserving?

## What Primordia Is Not

Primordia is not:
- a replacement for Prism, Topograph, or Stratograph
- a promise that full neuron-level free-form evolution is always practical
- an excuse to explode the search space without budget control

It is a bounded research layer.

## Long-Run Shape

If Primordia matures, it should become:
- a motif discovery engine
- a source of reusable primitive libraries
- a producer of priors for higher-level systems
- a place to measure bottom-up transfer honestly

## Design Principles

### 1. Cheap first

Primitive-first search must be dramatically cheaper than architecture-scale
search or it loses its strategic value.

### 2. Explicit exportability

Discovered motifs should be versioned and exportable. Otherwise the layer cannot
contribute to the umbrella.

### 3. Benchmark discipline

Primitive discovery still needs shared packs, contender expectations where
appropriate, and budget disclosure.

### 4. No hidden merger

Primordia should remain a distinct layer. Its output can seed higher-level
systems, but it should not be silently buried inside them.

## Success Criteria

Primordia succeeds if it produces:
- reusable motif artifacts
- honest low-cost benchmark evidence
- measurable transfer into higher-level search studies
- insight about which low-level structures actually matter

## Final Statement

Primordia is the primitive-first search bet in EvoNN.

Its job is to discover whether useful neural structure can be found profitably
below the level of families, flat graphs, and hierarchical cells, and whether
that structure deserves to survive into the rest of the stack.
