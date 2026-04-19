# EvoNN Budget Contract

## Purpose

This document defines the minimum budget vocabulary that all EvoNN systems
should use when they claim comparability.

The goal is not to force identical internals across systems. The goal is to
make resource claims honest enough that side-by-side comparison means something.

## Core Principle

A run is only fairly comparable when its resource envelope is explicit.

Every comparable run should declare, at minimum, the following normalized budget
fields.

## Required Budget Fields

### 1. Evaluation budget

Meaning:
how many candidate evaluations the system is allowed to spend.

Examples:
- number of contender evaluations
- number of evolutionary candidate evaluations
- number of promotion-stage evaluations in staged pipelines

Requirement:
report both total evaluations and any staged breakdown.

### 2. Wall-clock budget

Meaning:
maximum allowed elapsed runtime for the run.

Requirement:
report target budget and actual runtime.

### 3. Training-step or optimization budget

Meaning:
how much learning work each candidate may consume.

Examples:
- epochs
- optimizer steps
- tokens processed
- batches processed
- proxy-fit iterations

Requirement:
use the most natural unit for the system, but report it explicitly.

### 4. Hardware envelope

Meaning:
what machine constraints the run assumed.

Minimum fields:
- device class
- CPU count used
- GPU or accelerator type if any
- memory ceiling target if enforced
- worker count

For EvoNN local-first work, this is especially important on Apple Silicon.

### 5. Model-size or artifact-size budget

Meaning:
what size limits influenced the search.

Examples:
- parameter cap
- byte cap
- memory footprint target
- latency band

Requirement:
report declared caps and measured outcomes when available.

### 6. Benchmark-surface budget

Meaning:
what part of the benchmark universe the run was actually allowed to touch.

Minimum fields:
- pack identifier
- benchmark count
- benchmark tier from `BENCHMARK_LADDER.md`
- any reductions, subsets, or filtered views

### 7. Fidelity regime

Meaning:
how the run stages cheap versus expensive evaluation.

Examples:
- proxy only
- staged proxy -> medium -> full
- reduced dataset then full dataset
- cheap motif score then promoted task score

Requirement:
make promotion rules explicit.

## Required Reporting Fields

Every export intended for comparison should expose or derivably imply:
- system name
- run id
- pack id
- benchmark tier
- total evaluations
- staged evaluations if applicable
- wall time
- worker count
- declared hardware class
- declared budget caps
- actual measured artifacts such as params, bytes, latency, or memory when
  supported

## Local-First Defaults

For local-first EvoNN work, systems should prefer explicit run classes such as:
- `smoke`
- `local`
- `overnight`
- `weekend`
- `special-study`

These are not replacements for real numbers. They are human-readable wrappers
around real budgets.

## Rules For Fair Comparison

A comparison is fair only when:
- pack identity matches
- benchmark reduction policy matches
- budget vocabulary is reported for both sides
- contender and evolutionary results disclose different evaluation semantics
  clearly
- staged fidelity differences are disclosed instead of hidden

## Rules For Transfer-Aware Runs

If a run consumes prior motif banks, archives, or seed lineages, it must report:
- what prior artifact was used
- whether the prior was learned on overlapping benchmark families
- whether the run is intended as fair comparison, transfer study, or internal
  acceleration experiment

This prevents hidden prior knowledge from masquerading as fresh search.

## What This Contract Does Not Do

This contract does not claim that all systems spend compute in equivalent ways.
It only ensures that claims about resources are legible enough to audit.

## Bottom Line

If a run cannot say what it was allowed to spend, it should not be used for a
strong comparative claim.