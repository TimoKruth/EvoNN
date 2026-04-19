# Primordia Implementation Plan

## Goal

Introduce Primordia as a real umbrella package and give it a staged path from
concept to useful primitive-first search runtime.

## Stage 1: Scaffold And Boundaries

Objective:
create a clean package with explicit responsibilities.

Deliver:
- package metadata
- CLI entry point
- README and VISION
- implementation plan
- workspace integration

## Stage 2: Primitive Search Schema

Objective:
define the minimal search object.

Questions to settle:
- what is a primitive genome?
- how are tiny motifs represented?
- what invariants keep search bounded?
- what metrics define motif complexity?
- what counts as equivalence versus novelty?

Likely artifacts:
- primitive genome schema
- codec or manifest format
- motif digest rules

## Stage 3: Cheap Evaluation Lane

Objective:
create a low-cost benchmark lane for primitive discovery.

Possible evaluations:
- tiny supervised proxy tasks
- structure-conditioned representation scores
- small reconstruction or prediction tasks
- reduced sequence or image proxy tasks

Constraint:
this lane must stay local-first and cheap enough to justify its existence.

## Stage 4: Motif Bank And Export

Objective:
persist discoveries so later systems can consume them.

Likely artifacts:
- motif bank format
- motif summaries
- provenance metadata
- export hooks for Compare and later transfer studies

## Stage 5: Upstream Seeding Experiments

Objective:
test whether Primordia outputs actually help higher-level systems.

Likely studies:
- Primordia -> Stratograph cell seeding
- Primordia -> Topograph operator priors
- Primordia -> Prism family component priors where applicable

Requirement:
all such transfer must be auditable and clearly labeled as transfer-aware runs.

## Anti-Goals

Primordia should not begin by:
- attempting giant benchmark runs
- pretending low-level free-form search is cheap when it is not
- silently replacing architecture-level systems
- hiding transfer as if it were fresh search

## First Milestone

The first milestone is complete when:
- Primordia is present in the workspace
- umbrella docs reference it as a first-class layer
- the package has a stable identity for future runtime work
