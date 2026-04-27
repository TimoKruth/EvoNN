# EvoNN Telemetry Spec

## Purpose

This document defines the minimum telemetry surface that EvoNN systems should
emit when they want to be understandable, auditable, and comparable.

The point is not to erase system-specific telemetry. The point is to guarantee a
common floor beneath richer package-local reporting.

## Minimum Run Metadata

Every serious run should record:
- system name
- run id
- git commit or code version identifier when available
- config path or embedded config snapshot
- start timestamp
- end timestamp or latest checkpoint timestamp
- run status: completed, failed, interrupted, resumed, cancelled

## Minimum Budget Metadata

Every run should record the normalized fields from `BUDGET_CONTRACT.md`,
including at least:
- pack id
- benchmark tier
- declared evaluation budget
- actual evaluations spent
- evaluation-counting semantics
- cached / failed / invalid evaluation counts when applicable
- resumed-run provenance when applicable
- whether the exported result is partial or complete
- declared wall-time budget
- actual wall time
- hardware class
- worker count

## Minimum Seeding Metadata

Every run intended for transfer analysis, ladder comparison, or compare export
should record:
- `seeding_enabled`
- `seeding_ladder`: `none`, `direct`, or `staged`
- `seed_source_system`: `primordia`, `stratograph`, `topograph`, `prism`, or `null`
- `seed_source_run_id`
- `seed_artifact_path`
- `seed_target_family` when a seed is conditioned on benchmark family or task family
- `seed_selected_family` when the consumer chooses one family from multiple candidates
- `seed_rank` when the consumer chooses a ranked seed candidate
- `seed_overlap_policy`: whether the seed source is benchmark-disjoint, benchmark-overlapping, family-overlapping, or unknown

These fields exist to make transfer policy auditable. They do not erase the
architectural identity of the target system.

## Minimum Search Telemetry

The search system should emit whatever is natural for its abstraction, but at a
minimum should expose:
- candidate count evaluated
- best score so far
- current phase or stage
- archive or elite count if relevant
- failure or invalid-candidate count if relevant
- generation or iteration count if relevant

## Minimum Artifact Telemetry

Where supported, systems should report:
- parameter count
- model bytes or serialized size estimate
- latency estimate or measured latency
- memory estimate or measured peak memory

If a metric is unsupported, omit it explicitly or mark it unavailable rather
than silently pretending it does not matter.

## System-Specific Expectations

### Primordia
- primitive count or motif complexity
- microcircuit depth or width summary
- motif bank size
- promotion count into higher-fidelity stages

### Prism
- family distribution
- family archive occupancy
- transfer or inheritance usage

### Topograph
- topology size
- novelty metrics
- MAP-Elites occupancy
- mutation operator success summaries

### Stratograph
- macro depth
- cell library size
- reuse ratio
- clone and specialization counts
- motif frequency summaries

### Contenders
- contender family
- contender configuration id
- training budget actually used

### Compare
- compared run ids
- pack id
- comparison assumptions
- excluded runs or filtered artifacts
- ladder labels present for every seeded run
- direct, staged, and unseeded runs kept in distinct comparison buckets

## Report Surfaces

Telemetry should be present in at least one of these forms:
- structured summary JSON
- metrics database
- markdown report
- checkpoint metadata

Best case:
important metrics appear in both machine-readable and human-readable form.

## Resume And Failure Telemetry

Runs should make it easy to tell:
- whether a run resumed from checkpoint
- whether a run consumed prior artifacts
- why a run stopped
- whether results are partial or final

## Ladder Comparison Reporting Rules

When a run is seeded and exported into EvoNN-Compare or any umbrella-level table,
it should remain obvious:
- whether the run was unseeded, direct-ladder seeded, or staged-ladder seeded
- which upstream system supplied the seed artifact
- whether the seed came from overlapping benchmark families
- whether the run is being used as a fair baseline comparison, transfer study, or acceleration study

Direct and staged runs must not be merged into one anonymous "seeded" bucket.
If ladder metadata is missing, the run should be marked transfer-opaque rather
than treated as cleanly comparable.

## Bottom Line

Opaque search is weak search. EvoNN should prefer runs that explain themselves
well enough to survive comparison and later reuse.
