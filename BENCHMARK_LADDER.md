# EvoNN Benchmark Ladder

## Purpose

This document defines the intended difficulty ladder for EvoNN benchmark packs.

The ladder exists to prevent two common failures:
- overfitting to tiny toy tasks and pretending the project is progressing
- jumping directly to frontier-style tasks before the budget, telemetry, and
  evaluation stack can support honest claims

The goal is staged growth under bounded local budgets.

## Operating Principle

Every new benchmark pack should fit somewhere on the ladder.

A benchmark pack is useful when it helps answer at least one of these:
- what abstractions work at this difficulty level?
- what fails under this resource profile?
- what transfers upward?
- what baseline strength is required to keep the result honest?

## Tier A: Smoke And Sanity Packs

Purpose:
prove pipelines work and catch regressions quickly.

Characteristics:
- tiny datasets
- short runs
- low memory pressure
- deterministic or near-deterministic outcomes
- suitable for CI, local smoke, and minimal parity checks

Typical examples:
- tiny tabular classification/regression
- tiny image subset tasks
- tiny sequence tasks
- toy language modeling fixtures

Primary value:
engineering trust, not scientific claims.

## Tier B: Core Local Research Packs

Purpose:
establish the standard local workbench for EvoNN.

Characteristics:
- real datasets
- still bounded enough for laptop research loops
- enough diversity to expose benchmark-specific overfitting
- suitable for family, topology, hierarchy, and contender comparisons

Typical examples:
- OpenML tabular packs
- small image classification packs
- small text/sequence classification packs
- lightweight language modeling packs

Primary value:
this should be the default proving ground for most day-to-day EvoNN work.

Current canonical pack:
- `tier_b_core` is the canonical benchmark-ladder Tier B pack
- `tier1_core` remains the trusted recurring compare lane, not a synonym for ladder Tier B

Naming rule:
- use lettered names like `tier_b_*` for benchmark-ladder packs
- keep legacy numeric names like `tier1_core` or `tier2_evonn_leaning` for compare-lane or symmetry-class assets
- do not introduce `tier2_core`; it would collide with the existing numeric tier vocabulary

## Tier C: Architecture-Sensitive Packs

Purpose:
stress the actual search abstractions rather than only basic pipeline maturity.

Characteristics:
- more difficult generalization surface
- stronger need for transfer, reuse, or inductive bias
- more visible tradeoffs in quality versus compute, bytes, or latency
- may include multi-domain packs or memory-sensitive tasks

Typical examples:
- compact multi-domain suites
- harder sequence modeling tasks
- transfer-oriented benchmark groups
- constrained multimodal or structured reasoning tasks where feasible

Primary value:
this is where architecture ideas should begin to clearly separate from one
another.

## Tier D: System-Like Packs

Purpose:
test whether EvoNN systems can begin to matter on tasks that look less like
classic static supervised learning and more like structured capability surfaces.

Characteristics:
- longer evaluation loops
- more expensive scoring
- may require staged or proxy evaluation
- may need reduced versions for local execution

Typical examples:
- program-like toy tasks
- code classification or localization tasks
- retrieval-sensitive sequence tasks
- structured decision or planning tasks under fixed interfaces

Primary value:
bridge from architecture search to harder real-world benchmark classes.

## Tier E: Frontier And Aspirational Packs

Purpose:
serve as north-star benchmark classes.

Characteristics:
- expensive
- noisy
- often long-context or system-like
- not suitable as the default inner loop
- often require reduced, subset, or staged forms for local use

Typical examples:
- reduced SWE-style tasks
- broader code-repair or agent-like evaluation surfaces
- difficult multi-step reasoning or long-horizon benchmark families

Important note:
this tier is not restricted to LLM-style tasks. The point is to include any hard
benchmark family that a neural system might eventually attack in a serious but
budget-aware way.

Primary value:
ambition and direction, not near-term vanity metrics.

## Admission Rules For New Packs

A new pack should declare:
- ladder tier
- modality or modalities
- expected local runtime class
- minimum contender set
- whether it is suitable for smoke, daily research, overnight runs, or special
  studies only
- whether full-fidelity evaluation is local-safe or requires staged reduction

## Ladder Use By Horizon

Horizon 1:
- standardize Tier A and Tier B semantics
- document minimum contender expectations

Horizon 2:
- create cheap primitive-discovery packs mostly in Tier A and B
- add motif-oriented proxy tasks where justified

Horizon 3:
- study transfer across Tier B and C families

Horizon 4:
- expand carefully into Tier D and E with explicit reductions and budget limits

## What To Avoid

Avoid these mistakes:
- treating Tier A wins as evidence of broad capability
- using Tier E tasks as routine default evaluations
- adding benchmark packs without contender expectations
- mixing incomparable tasks without declaring why they belong together
- hiding evaluation reductions or shortcuts in reports

## Bottom Line

The ladder exists so EvoNN can aim at hard problems without losing discipline.
The project should climb, not leap.
