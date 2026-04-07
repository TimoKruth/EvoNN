# Evo Neural Nets — Project Overview

## What This Is

Three evolutionary neural architecture search (NAS) systems that discover
neural network architectures automatically, plus a comparison layer that
measures them against each other.

Each system searches at a different level of abstraction:

```
┌─────────────────────────────────────────────────────┐
│  EvoNN (Track A) — High Layer                       │
│  Evolves WHICH architecture family to use            │
│  17 pre-built families: MLP, Conv2D, Transformer,   │
│  Attention, State-Space, MoE, etc.                   │
│  Searches: family selection, hyperparameters, depth  │
│  Fixed: internal structure of each family block      │
└─────────────────────────────────────────────────────┘
                        ↕ compared via Symbiosis
┌─────────────────────────────────────────────────────┐
│  EvoNN-2 (Track B) — Middle Layer                   │
│  Evolves HOW neurons are connected                   │
│  NEAT-style topology: add/remove layers and          │
│  connections, skip links, variable depth DAGs         │
│  Searches: topology, connectivity, per-layer width   │
│  Fixed: each neuron is always Linear + Activation    │
└─────────────────────────────────────────────────────┘
                        ↕ potential future extension
┌─────────────────────────────────────────────────────┐
│  Future: Operator-Level Evolution — Low Layer        │
│  Evolves WHAT each neuron computes                   │
│  Primitive ops: add, multiply, sigmoid, concat       │
│  Could discover new activation functions,            │
│  new layer types, novel compute patterns             │
│  NOT YET STARTED — research direction only           │
└─────────────────────────────────────────────────────┘
```

## The Three Layers Explained

### High Layer: EvoNN — "Which blueprint?"

EvoNN picks from 17 hand-designed neural network families (like choosing
between a convolutional network, a transformer, or an MLP) and evolves
their settings. It's fast because the building blocks are proven — it
just needs to find the best configuration.

Think of it like: choosing which type of car to build (sedan, SUV,
sports car) and then tuning the engine size, suspension, and tires.

### Middle Layer: EvoNN-2 — "How to wire it?"

EvoNN-2 starts with almost nothing and grows the network topology from
scratch using NEAT (NeuroEvolution of Augmenting Topologies). It adds
neurons, adds connections, creates skip links, and removes redundant
parts. Each neuron is a simple linear transformation + activation.

Think of it like: starting with a blank circuit board and evolving which
wires to place, where to add components, and how to route signals.

### Low Layer (Future) — "What does each component do?"

A future system could evolve the computation inside each neuron itself,
using primitive mathematical operations as building blocks. This could
discover entirely new types of neural network layers that no human has
designed.

Think of it like: not just placing components on a circuit board, but
inventing the components themselves from transistors and resistors.

## The Hybrid

The Hybrid system combines the Middle and High layers: EvoNN-2's NEAT
topology evolution decides the network structure, but each node can be
a full EvoNN family block (Transformer, MLP, Attention) instead of a
simple neuron. This lets topology evolution arrange sophisticated
building blocks in novel ways.

## Symbiosis — The Comparison Layer

EvoNN-Symbiosis is not a search system. It's the measurement
infrastructure that compares all systems fairly:

- Runs each system on the same benchmarks with the same seeds and budgets
- Computes statistical significance (Wilcoxon tests, confidence intervals)
- Produces campaign reports and a web dashboard (Observatory)
- Supports 5 campaign types: solo, comparison, hybrid, symbiosis, exploration

## Current Results (April 2026)

On 8 Tier 1 benchmarks (iris, wine, breast cancer, moons, digits,
diabetes, friedman, credit-g):

- **EvoNN wins most benchmarks** overall — its family diversity gives it
  an edge on classification and a large advantage on regression
- **EvoNN-2 wins digits (80-100%)** and credit-g (~60%) — topology
  freedom finds better classifiers for these tasks
- **Hybrid Transformer** achieves near-perfect scores on synthetic
  language modeling (perplexity ≈ 1.003)
- At higher budgets (512+) the systems converge — the first tied pair
  appeared at budget 512

## Repository Structure

```
Evo Neural Nets/
├── EvoNN/                  # Track A — family-based macro NAS
├── EvoNN-2/                # Track B — topology evolution (NEAT)
├── EvoNN-Symbiosis/        # Comparison layer + Hybrid engine
│   ├── src/symbiosis/
│   │   ├── contracts/      # Shared data schemas
│   │   ├── comparison/     # Statistical comparison engine
│   │   ├── orchestration/  # Campaign runner
│   │   ├── hybrid/         # Hybrid topology + family engine
│   │   └── web/            # Observatory dashboard
│   └── campaigns/          # ~50 campaign results
├── OVERVIEW.md             # This file
└── PROJECT_AUDIT_REPORT.md # Detailed technical audit
```

## Where It Could Go

### Near-Term: EvoNN-2 Phase B5 (Transformer Gene)

Add Transformer blocks as a native neuron type in EvoNN-2. Instead of
only evolving connections between simple neurons, NEAT could evolve
connections between attention layers — discovering novel transformer
architectures from the topology level.

### Mid-Term: Hierarchical Hybrid

Combine the outer Hybrid (NEAT topology of blocks) with inner EvoNN-2
(NEAT topology within blocks). Two nested levels of topology evolution:
the outer level decides which blocks exist and how they connect, the
inner level decides what each block computes internally.

### Long-Term: Operator-Level Evolution (Low Layer)

Evolve the mathematical operations inside neurons themselves. Could use
Cartesian Genetic Programming (CGP) or a similar approach where
primitive ops (add, multiply, sigmoid, log, concat) are the genes. This
could discover activation functions, normalization methods, or entirely
new layer types that outperform human-designed ones.

This is a research direction, not a current project.
