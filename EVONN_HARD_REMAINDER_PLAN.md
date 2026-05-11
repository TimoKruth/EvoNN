# EvoNN Hard Remainder Plan

This is a companion plan to `EVONN_CONSOLIDATED_PLAN.md`.

`EVONN_CONSOLIDATED_PLAN.md` remains the active operating plan. This document
captures the harder unfinished work that is not yet specific enough, proven
enough, or low-risk enough to be the day-to-day execution checklist. Sections
from this plan should be promoted into the consolidated plan only when they have
a clear owner, validation lane, acceptance criteria, and expected evidence
artifact.

## Purpose

The current EvoNN stack has the trust substrate, comparison surface, benchmark
ladder, dashboard, contender floor, real-LM cache validation, output-quality
checks, performance-baseline commands, and first transfer-reporting surfaces.

The remaining hard work is not mostly more plumbing. The hard rest is proving
that EvoNN can produce durable scientific value:

- repeated evidence instead of one-off leaderboard wins
- engine improvements that survive stronger benchmarks and contenders
- transfer/seeding that actually changes downstream search outcomes
- harder LM/image/tabular tasks without misleading smoke-task saturation
- performance improvements that are accepted or rejected by measurements
- a research-memory system that lets future work build on past runs
- a clear rule for when engines should specialize, merge, or be retired

## Relationship To Existing Plans

Use this hierarchy:

- `VISION.md`: long-term product and research direction.
- `EVONN_CONSOLIDATED_PLAN.md`: active execution plan and near-term operating
  source of truth.
- `EVONN_HARD_REMAINDER_PLAN.md`: long-horizon hard-remainder backlog and
  planning substrate.
- `BENCHMARK_LADDER.md`, `BUDGET_CONTRACT.md`, `TELEMETRY_SPEC.md`, and
  `RESEARCH_DECISION_GATE.md`: contracts and gates.

Do not treat this file as permission to start broad rewrites. Each item here
must become a scoped issue/branch before implementation.

## External Research Takeaways

The plan below is shaped by several established patterns from reproducible ML,
NAS, AutoML, and evolutionary optimization:

- NAS-Bench-101 and NAS-Bench-201 show why architecture-search claims need a
  reproducible benchmark surface, fixed search/evaluation rules, and comparable
  diagnostics instead of uncontrolled expensive runs.
- OpenML and the AutoML Benchmark emphasize shared tasks, run metadata, and
  reusable benchmark definitions as the basis for comparing algorithms over
  time.
- MLPerf shows the value of benchmark rules, system metadata, quality targets,
  and clearly separated benchmark scenarios when comparing heterogeneous
  systems.
- Demsar's statistical-comparison guidance is a useful reminder that one run on
  one benchmark is not enough for broad algorithm claims; repeated results over
  multiple datasets need robust non-parametric interpretation.
- MAP-Elites and quality-diversity search are directly relevant to EvoNN because
  they produce archives of diverse high-performing candidates, not just a single
  best candidate. That aligns with Primordia motifs, Topograph/Stratograph
  structural descriptors, and dashboard evidence.
- NSGA-II and multi-objective NAS work support treating accuracy, runtime,
  memory, architecture complexity, and search cost as explicit objectives rather
  than optimizing only a single score.

References are listed at the end of this document.

## Current Code Assessment

What is now solid:

- The monorepo has one coherent comparison stack: `Compare`, `Shared`,
  `Contenders`, `Prism`, `Topograph`, `Primordia`, `Stratograph`, and
  `shared-benchmarks`.
- `Compare` owns the right trust surfaces: fair-matrix runs, trend rows,
  benchmark audit, contender-floor checks, output-quality checks,
  performance-baseline commands, dashboards, and transfer-report sections.
- The benchmark ladder exists with staged Tier A/B/C/D lanes, required
  contender metadata, score ceilings, and admission status.
- Real LM benchmarks now exist beyond smoke variants, with cache validation in
  the shared layer.
- The dashboard has moved beyond static markdown into a usable evidence
  surface, including budget/benchmark/system views and recent-run history.
- Engines emit enough artifacts for recurring comparison to be possible.

What is still not fully finished:

- The first real seeded-vs-unseeded transfer result is not yet a scientific
  proof point. The reporting surface exists, but the hard part is showing that
  a seed artifact is consumed by a downstream engine and changes outcomes under
  repeated controlled runs.
- Evidence is still too workspace-local. There is no durable run registry with
  retention policy, canonical promotion records, and stable references to
  decision-grade evidence.
- Output quality is not yet fully L4. The system can measure and display a lot,
  but repeated-seed variance, effect sizes, statistical confidence, and
  decision labels are not yet first-class enough to make broad claims.
- Tier B naming and status should remain watched. The code has `tier_b_core_v2`
  as the expanded lane, while older docs and mental models can still refer to
  `tier_b_core`.
- Tier C is still the most important stress lane because it mixes harder
  tabular, image, regression, and real LM. It should not be treated as routine
  until repeated runs prove stability.
- Tier D is broad and useful, but it must stay separate from primary claims
  unless repeated evidence remains clean under the admitted benchmark list.
- Contender pressure is good enough for a floor, but not always good enough for
  strong claims. Optional enhanced baselines such as CNN/Transformer/boosted
  tree libraries still matter for image, LM, and harder tabular claims.
- Engine quality is uneven by modality. The clearest example is LM behavior:
  some systems can look excellent on smoke tasks, while others flatline on real
  LM because their search space or evaluator path has insufficient LM-specific
  adaptation.
- Performance optimization is not yet systematic. The `performance-baseline`
  interface exists, but optimization work needs a strict baseline -> one change
  -> remeasure -> accept/scrap loop.
- There is no explicit rule for engine portfolio decisions: specialize,
  merge, graduate, or retire.

## Strategic End State

This plan is complete when EvoNN can make these claims honestly:

- The benchmark ladder is not just executable; it is statistically useful over
  repeated runs.
- At least one EvoNN engine beats or ties the required contender floor on a
  meaningful subset of non-smoke benchmarks under repeated evidence.
- At least one transfer/seeding path is proven useful, proven harmful, or
  formally classified inconclusive with enough evidence to guide strategy.
- Performance improvements are accepted only when they preserve fairness and
  improve measured runtime/cost across multiple budgets.
- Engine-specific work is driven by evidence: each engine has a clear research
  role, measurable failure modes, and a rule for continued investment.
- New benchmark tiers or modalities can be added without weakening trust.

## Workstream 1: Durable Evidence Registry

### Problem

The current workspace/run-artifact model is useful for local work, but it is
not yet a durable research memory. Without a run registry, old claims become
hard to audit, dashboard history can drift by workspace, and decisions depend
too much on whoever ran the latest command.

### Target

Create a canonical evidence registry that records every promoted comparison
run, links it to immutable artifacts, and makes dashboard/history views
rebuildable.

### Design Direction

Add a root-level evidence convention, preferably not by committing large result
payloads:

- `evidence/index.jsonl`: append-only registry rows for promoted runs.
- `evidence/runs/<run_id>/manifest.json`: minimal checked-in run pointer or
  exported artifact manifest.
- external/local artifact storage path: large artifacts stay outside git unless
  intentionally compact.
- `evidence/README.md`: retention, promotion, and review rules.

Each registry row should include:

- `run_id`
- timestamp
- git commit
- branch
- pack
- preset
- budget
- seed
- backend class
- host/runtime fingerprint
- systems included
- contender-floor state
- output-quality level
- trust state
- dashboard/report paths
- artifact checksum or manifest checksum
- decision status: `exploratory`, `candidate`, `promoted`, `rejected`,
  `superseded`

### Implementation Plan

1. Add evidence registry schema in `EvoNN-Compare` or `EvoNN-Shared`.
2. Add CLI command:

   ```bash
   uv run --package evonn-compare evonn-compare evidence promote \
     --workspace <run-workspace> \
     --run-id <run-id> \
     --decision candidate
   ```

3. Add dashboard support for loading both live workspaces and promoted evidence
   registry rows.
4. Add a CI-safe schema validation test for evidence rows.
5. Add a pruning/export command so large local artifacts can be moved without
   breaking the registry.

### Acceptance Criteria

- A decision-grade run can be promoted into an evidence registry.
- The dashboard can rebuild the latest 10-20 full comparison overview from the
  registry, not only from incidental `.tmp` directories.
- A stale/missing artifact is reported explicitly.
- Evidence rows are immutable after promotion except for a supersession marker.
- A new engineer can inspect why a claim was accepted or rejected without
  rerunning the whole suite.

## Workstream 2: L4 Statistical Decision Layer

### Problem

Single-run wins and dashboard rankings are not enough for scientific claims.
The system needs repeated seeds, variance, effect sizes, and clear decision
labels. Without this, EvoNN risks optimizing for noise or smoke-task
saturation.

### Target

Move from L3 measurable outputs to L4 decision-grade evidence for promoted
lanes.

### Design Direction

Add a statistical summary layer that operates on repeated runs:

- per-system rank distribution
- per-benchmark score distribution
- per-budget improvement slope
- contender-floor margin distribution
- ceiling-tie exclusion
- effect size against best required contender
- non-parametric rank tests where enough benchmarks/runs exist
- decision labels:
  - `clear_gain`
  - `likely_gain`
  - `no_material_change`
  - `regression`
  - `inconclusive`
  - `needs_more_runs`

Keep it conservative. The goal is not p-value theater; it is preventing
overclaiming from noisy small runs.

### Implementation Plan

1. Define minimum repeated-run gates:
   - Tier A: 3 seeds for promoted claims.
   - Tier B: 3 seeds at local budget, 2 seeds at overnight budget.
   - Tier C: 3 seeds at local budget, 2 seeds at overnight budget before
     decision-grade promotion.
   - Tier D: stays broad-lane only unless 3 clean repeated runs exist.
2. Add `evonn-compare evidence analyze` to aggregate promoted runs.
3. Add rank/effect-size summaries to JSON, markdown, and dashboard.
4. Add "not enough evidence" states instead of silently ranking uncertain
   comparisons.
5. Add tests with synthetic run fixtures covering ceiling ties, missing runs,
   backend drift, and mixed-budget ambiguity.

### Acceptance Criteria

- The dashboard distinguishes "currently best observed" from
  "decision-grade improvement."
- Benchmark saturation does not inflate win claims.
- Repeated-run variance is visible for every promoted comparison.
- PRs claiming engine advancement include before/after evidence bundles with
  statistical decision labels.

## Workstream 3: Real Transfer/Seeding Proof

### Problem

Transfer/seeding is central to the EvoNN vision, but metadata alone is not
proof. The hard question is whether structure discovered by one engine improves
another engine under fair accounting.

### Target

Run one auditable transfer loop that proves gain, regression, no gain, or
inconclusive outcome.

### Recommended First Path

Use `Primordia -> Topograph` first.

Reasoning:

- Primordia's role is cheap primitive/motif discovery.
- Topograph's role is topology-first structural search.
- The structural relationship is more direct than forcing Primordia motifs into
  Prism's broad family engine first.
- A failed result is still informative: it says primitive motifs are not yet
  transferable in their current form.

Use `Primordia -> Prism` second if the first path shows signal or if Prism's
family engine gets an explicit seed-consumption path.

### Required Seed Contract

A seed artifact must contain:

- source engine and version
- source benchmark/pack/budget/seed
- source backend and hardware metadata
- candidate genotype or motif encoding
- descriptor vector
- quality score and metric direction
- novelty/diversity descriptor
- budget cost already spent
- contamination policy
- compatible target engines
- target ingestion instructions
- checksum

### Required Target Behavior

The target engine must support three modes:

- `unseeded`: normal run.
- `seeded`: initialize or bias search with seed artifact.
- `staged_seeded`: consume seed after a warmup budget fraction.

Budget accounting must make seed cost explicit:

- `free_prior`: seed is external prior, target budget unchanged.
- `charged_prior`: source budget is included in total comparison budget.
- `reported_prior`: target budget unchanged, but source cost displayed and
  excluded from direct budget-matched claims.

Default for first research run:

- Use `reported_prior`.
- Do not claim a budget-matched gain until `charged_prior` also works.

### Implementation Plan

1. Formalize seed artifact schema in `EvoNN-Shared`.
2. Add Primordia seed export command with motif bank filtering.
3. Add Topograph seed import path that visibly changes initialization or search
   bias.
4. Add Compare transfer-run command that launches unseeded and seeded variants
   with identical pack/budget/seed/backend.
5. Add transfer dashboard section with:
   - source artifact provenance
   - target run mode
   - benchmark-level deltas
   - budget accounting mode
   - decision label
6. Run first sequence:

   ```bash
   tier_b_core_v2 @ 96, 3 seeds
   tier_b_core_v2 @ 384, 2 seeds if local signal appears
   tier_c_architecture_sensitive @ 128, exploratory only
   ```

### Acceptance Criteria

- Seeded and unseeded runs are reproducible from one command.
- The target engine demonstrably consumes seed artifacts.
- Transfer claims are separated from normal leaderboard claims.
- The result is classified as gain, no gain, regression, or inconclusive.
- If transfer fails, the artifact explains whether failure came from seed
  quality, target ingestion, benchmark mismatch, or budget accounting.

## Workstream 4: Engine Portfolio Quality Plan

### Problem

The systems should not all become weak variants of the same search loop. Each
engine needs a distinct scientific role and evidence that justifies continued
investment.

### Portfolio Rule

Every engine must eventually satisfy one of these statuses:

- `reference`: default engine for routine operation.
- `challenger`: active alternative with distinct wins or failure-mode insight.
- `specialist`: useful on specific benchmark families or budgets.
- `seed_source`: useful because it improves another engine.
- `baseline_only`: useful as internal pressure but not strategic.
- `archive_candidate`: not worth active investment unless new evidence appears.

### Prism

Role:
reference family-aware engine.

Hard improvements:

- Improve budget allocation across families and benchmarks.
- Add search-space diagnostics: which family/operator caused each gain.
- Add Pareto scoring over metric quality, runtime, size, and complexity.
- Add low-fidelity pruning only if budget accounting remains honest.
- Add seed-consumption path after Topograph transfer validates the contract.

Evidence target:

- Prism should remain the strongest general EvoNN engine on Tier B and broad
  admitted Tier D, or the plan should explicitly downgrade it from reference.

### Topograph

Role:
topology-first challenger and likely first transfer consumer.

Hard improvements:

- Add topology descriptors suitable for quality-diversity archives.
- Track architecture shape metrics: depth, width, skip structure, motif reuse,
  sparsity, and graph edit distance from seeds.
- Use MAP-Elites-style archive exploration for diverse high-performing
  topologies.
- Add seed ingestion from Primordia motifs.
- Add ablation: topology-aware search vs flat/random topology variants.

Evidence target:

- Topograph should win or tie Prism on a meaningful topology-sensitive subset,
  or become a specialist/transfer consumer rather than a primary challenger.

### Stratograph

Role:
hierarchy-first challenger.

Hard improvements:

- Diagnose LM flatlining and identify whether it is evaluator, genotype,
  compiler, or search-policy related.
- Add hierarchy descriptors: macro depth, cell reuse, inter-level connectivity,
  and hierarchy collapse rate.
- Add flat-vs-hierarchical ablations.
- Add hierarchy-specific mutation/crossover operators instead of generic
  topology perturbations.
- Add budget-adaptive hierarchy depth limits.

Evidence target:

- Stratograph must show a hierarchy-specific advantage on at least one class of
  benchmarks, or be downgraded to archive candidate after the current roadmap.

### Primordia

Role:
primitive motif discovery and seed source.

Hard improvements:

- Produce a ranked motif bank with diversity descriptors.
- Separate motifs that are locally good from motifs that transfer downstream.
- Add motif aging/retirement based on transfer evidence.
- Add descriptor coverage metrics so Primordia is not only optimizing one
  narrow smoke pattern.
- Add contamination-safe train/test handling for seed artifacts.

Evidence target:

- Primordia must either improve downstream search through transfer or be judged
  only as a specialist engine for primitive-level tasks.

### Contenders

Role:
external baseline floor.

Hard improvements:

- Promote enhanced contender pressure where practical:
  - image: small CNN
  - LM: tiny transformer or stronger sequence baseline
  - tabular: XGBoost/LightGBM/CatBoost where dependencies are acceptable
- Keep required floor dependency-light, but make missing enhanced pressure
  visible in claims.
- Add per-benchmark contender adequacy labels:
  - `strong_floor`
  - `acceptable_floor`
  - `weak_floor`
  - `missing_enhanced_pressure`

Evidence target:

- EvoNN wins should be interpretable as wins over a reasonable floor, not wins
  over toy baselines.

## Workstream 5: Hard Benchmark Expansion

### Problem

The current ladder is much better than the original smoke surface, but harder
tasks must be added carefully. Adding many benchmarks without contender floors,
cache validation, runtime bounds, and statistical evidence would weaken the
platform.

### Target

Promote harder benchmark classes only when they pass admission gates and have
strong enough baselines.

### Lane Policy

Keep these lanes separate:

- Tier A: contract/smoke.
- Tier B: default local research lane.
- Tier C: architecture-sensitive stress lane.
- Tier D: broad admitted shared suite.
- Tier E: future frontier lane for harder LM/image/sequence tasks.

Tier E should not start as a large suite. It should start as a candidate list
with admission reports.

### Tier E Candidate Areas

Language modeling:

- longer TinyStories slices
- WikiText-2 medium windows
- character/byte-level and token-level variants
- sequence-copy or algorithmic sequence tasks
- optional code/text mini-corpora if licensing and cache reproducibility are
  clean

Image:

- MNIST/FashionMNIST full variants with stronger contender pressure
- CIFAR-like small image tasks only if runtime and baseline floors are honest
- corrupted/noisy variants for robustness

Tabular:

- larger OpenML classification/regression tasks
- high-cardinality categorical tasks
- imbalanced classification with non-accuracy primary metrics
- noisy regression with robust metrics

Synthetic:

- controlled nonlinearity, interaction order, noise level, dimensionality, and
  class imbalance ladders
- tasks designed to isolate topology, hierarchy, or primitive-transfer value

### Admission Gates

A benchmark may enter Tier E candidate status when:

- benchmark data is reproducible locally
- metric direction is explicit
- ceiling semantics are explicit
- required contender floor exists
- runtime class is estimated
- all engines can produce L3 artifacts or explicitly declare unsupported
- cache validation exists if data is generated or sliced
- budget divisibility is defined for at least two budgets

A benchmark may become decision-grade only after:

- two clean repeated runs at low budget
- one clean repeated run at mid budget
- output-quality checks pass
- contender floor is not weak
- dashboard displays the benchmark without special manual interpretation

## Workstream 6: Performance And Cost Frontier

### Problem

Search quality and runtime are entangled. EvoNN needs performance work, but
performance optimizations are dangerous if they change budget semantics,
hardware conditions, or candidate evaluation quality.

### Target

Use a strict measurement-driven workflow:

1. Generate baseline.
2. Implement one improvement.
3. Re-run identical measurements.
4. Accept, scrap, or mark inconclusive.

### Measurement Set

For every performance branch, use at least:

- Tier A @ 16 and 64
- Tier B @ 96 and 384
- one Tier C local run if the change affects compiler/evaluator/runtime logic
- at least two seeds if the change affects search behavior
- backend metadata and host fingerprint

Metrics:

- wall-clock time
- candidates evaluated
- valid/invalid candidate ratio
- cache hit rate
- backend time vs orchestration time
- per-benchmark latency
- memory peak where available
- metric quality delta
- contender-floor margin delta

### Candidate Performance Improvements

MLX backend:

- batch candidate evaluation where candidate shapes allow it
- reduce CPU/GPU transfer churn
- cache compiled/evaluable graph fragments
- use shape bucketing for repeated structures
- add backend timing sections inside artifacts

Linux/numpy fallback:

- vectorize candidate evaluation hot paths
- avoid repeated dataset conversion
- add shared evaluator cache
- parallelize only at benchmark/system boundaries unless deterministic
  candidate-level parallelism is proven

Search-loop efficiency:

- early reject structurally invalid candidates before expensive evaluation
- multi-fidelity screening with honest budget labels
- candidate deduplication with explicit cache accounting
- adaptive budget allocation across benchmarks only in lanes that allow it
- archive-guided mutation to reduce repeated dead zones

Dashboard/reporting:

- precompute large evidence aggregates
- avoid re-reading full artifacts for every dashboard open
- add compact index files for last-N run views

### Acceptance Criteria

- A performance PR improves measured runtime or cost on at least one intended
  lane without reducing output quality.
- Any quality regression is explicit and justified as a tradeoff.
- Performance wins are not claimed across hardware/backend drift unless the
  comparison cohort marks them valid.
- Optimizations that only help smoke tasks but harm Tier B/C are rejected or
  scoped as smoke-only.

## Workstream 7: Multi-Objective Search And Quality-Diversity

### Problem

Current leaderboard views focus primarily on task metric wins. That is
necessary, but insufficient. EvoNN's search engines should eventually optimize
tradeoffs between score, runtime, architecture complexity, diversity, and
transfer value.

### Target

Add multi-objective and quality-diversity evidence without collapsing all
engines into one shared search implementation.

### Design Direction

Shared layer:

- define candidate descriptors and optional objective vectors
- define archive/report schemas
- define dashboard views for Pareto fronts and diversity coverage

Engine-local layer:

- each engine chooses descriptors that fit its abstraction
- search-core logic stays package-local
- engines emit comparable descriptor summaries

Candidate objectives:

- primary benchmark metric
- evaluation cost
- architecture size
- inference latency proxy
- runtime memory proxy
- novelty/diversity score
- transfer success score

Candidate descriptors:

- Prism: family/operator composition, parameter count, graph depth.
- Topograph: topology shape, sparsity, skip density, motif placement.
- Stratograph: hierarchy depth, cell reuse, layer composition.
- Primordia: primitive type, motif length, descriptor coverage.

### Implementation Plan

1. Add optional descriptor schema in Shared.
2. Add descriptor exports in one engine first, preferably Topograph.
3. Add a simple archive report in Compare.
4. Add one quality-diversity search strategy branch in Topograph or Primordia.
5. Compare against existing search at equal budget.
6. Promote only if archive diversity improves without destroying metric quality.

### Acceptance Criteria

- Candidate diversity is measurable, not just described.
- Pareto/descriptor views explain why an engine is useful even when it is not
  the top scalar-score winner.
- Quality-diversity features produce evidence that changes engine strategy
  decisions.

## Workstream 8: Runtime Portability And Hardware Truth

### Problem

Linux fallback is useful for CI and portability, while MLX on macOS is the
truth path for the Apple Silicon research target. The hard part is preventing
mixed backend evidence from becoming misleading.

### Target

Make backend portability honest, tested, and visible in every decision.

### Required Work

- Audit every engine for backend detection and hidden MLX assumptions.
- Make backend metadata required for decision-grade artifacts.
- Add backend capability declarations:
  - `mlx_native`
  - `numpy_fallback`
  - `sklearn_contender`
  - `torch_optional`
  - `unsupported`
- Add dashboard filters for backend class and hardware fingerprint.
- Add comparison-cohort warnings when backend/hardware changes across budgets
  or seeds.
- Prevent decision-grade claims from mixing MLX and fallback unless explicitly
  classified as portability evidence.

### Acceptance Criteria

- Linux CI validates fallback-safe packages.
- macOS validates MLX truth paths.
- Mixed-backend comparisons are visible and downgraded unless intentionally
  allowed.
- Backend changes between budgets cannot silently create false trend claims.

## Workstream 9: Engine Merge, Specialization, Or Retirement Decisions

### Problem

The monorepo currently supports multiple engines, but portfolio sprawl can
become expensive. The project needs explicit rules for continuing, merging, or
retiring engine tracks.

### Target

Create portfolio decision gates based on evidence, not preference.

### Decision Rules

Keep an engine as `challenger` when:

- it has repeated wins or near-wins on at least one meaningful lane
- its wins are not only smoke/ceiling ties
- it has distinct failure-mode insight or transfer value

Move an engine to `specialist` when:

- it is strong on a modality or budget class but weak overall
- the specialization is stable across repeated runs

Move an engine to `seed_source` when:

- it improves another engine more reliably than it wins directly

Merge shared pieces when:

- two engines duplicate benchmark, artifact, budget, or report logic
- the duplicated logic is not part of search-core identity

Do not merge when:

- the code is search-core logic
- the difference is scientifically meaningful
- unification would make failure modes harder to understand

Archive an engine track when:

- it cannot beat/tie contenders or improve another engine after its planned
  advancement branch
- maintenance cost exceeds evidence value
- it mostly duplicates another engine without measurable advantage

### Acceptance Criteria

- Every engine has a current portfolio status in dashboard/docs.
- Status changes require evidence links.
- Archive decisions preserve useful artifacts and lessons.
- The project does not continue investing equally in every engine by default.

## Recommended Sequencing

### Phase 0: Reconcile Hard Remainder Into Issues

Duration:
1 week.

Work:

- Keep this file as planning substrate.
- Select 3-5 concrete issues from it.
- Do not start every workstream at once.
- Resolve the Tier B naming/status ambiguity in docs and CLI help.
- Choose first transfer path, recommended `Primordia -> Topograph`.
- Define evidence registry storage policy.

Exit gate:

- The consolidated plan has only the next executable slice.
- This plan remains the hard-remainder backlog.

### Phase 1: Durable Evidence And L4 Repeated Runs

Duration:
2-4 weeks.

Work:

- Implement evidence registry.
- Add repeated-run statistical summaries.
- Promote recent clean runs into registry.
- Make dashboard last-N history registry-backed.
- Add decision labels.

Exit gate:

- A before/after engine PR can be judged from registry-backed evidence.

### Phase 2: First Real Transfer Proof

Duration:
4-8 weeks.

Work:

- Implement seed artifact schema.
- Export Primordia motif seeds.
- Consume seeds in Topograph.
- Run unseeded/seeded/staged runs on Tier B.
- Classify result.

Exit gate:

- Transfer is proven useful, harmful, or inconclusive with repeated evidence.

### Phase 3: Engine Quality-Diversity And Portfolio Decisions

Duration:
6-10 weeks.

Work:

- Add descriptor exports for Topograph or Primordia.
- Add one quality-diversity archive experiment.
- Diagnose Stratograph LM flatline.
- Decide whether each engine is reference/challenger/specialist/seed-source.

Exit gate:

- Engine roles are evidence-backed, not aspirational.

### Phase 4: Tier C/D Hardening And Tier E Candidate Admission

Duration:
8-12 weeks.

Work:

- Harden Tier C repeated runs.
- Keep Tier D broad lane separate and registry-backed.
- Add Tier E candidate audit, not immediate promotion.
- Add stronger optional contenders where practical.

Exit gate:

- Harder benchmarks expand the research surface without degrading trust.

### Phase 5: Performance Frontier

Duration:
continuous after Phase 1.

Work:

- Use performance baseline commands for every optimization branch.
- Optimize one path at a time.
- Reject changes that do not survive repeated measurement.

Exit gate:

- EvoNN has a repeatable performance-improvement process, not anecdotal speed
  tuning.

## Near-Term Issue Candidates

Create issues in this order:

1. Evidence registry schema and `evidence promote` command.
2. Last-N dashboard history from evidence registry.
3. Repeated-run statistical decision labels.
4. Primordia seed artifact schema and export command.
5. Topograph seed import and seeded initialization.
6. Transfer-run orchestration for unseeded/seeded/staged variants.
7. Stratograph LM flatline diagnostic report and ablation harness.
8. Topograph descriptor export and archive report.
9. Enhanced LM contender floor feasibility spike.
10. Tier E candidate audit file with admission status, not execution.
11. Backend/hardware drift guard for decision-grade cohorts.
12. Performance optimization branch template and acceptance report.

## Non-Goals

Do not use this plan to justify:

- a giant rewrite of all engines
- combining search-core implementations prematurely
- adding many benchmarks before floors and repeated evidence exist
- claiming transfer success from metadata-only runs
- making Linux fallback look equivalent to MLX-native evidence
- optimizing runtime without measuring quality and budget effects
- continuing all engines indefinitely without portfolio decisions

## Definition Of Done

This hard-remainder plan is complete when:

- The evidence registry makes promoted runs durable and auditable.
- Repeated-run statistical summaries are required for advancement claims.
- At least one transfer/seeding path has a valid outcome classification.
- Tier C is either promoted with repeated evidence or explicitly remains
  exploratory with known blockers.
- Tier D remains broad-lane separated unless repeated evidence supports broader
  use.
- Tier E candidates exist with admission reports but no trust shortcut.
- Each engine has an evidence-backed portfolio status.
- Performance work follows baseline/change/remeasure/accept-or-scrap discipline.

## References

- NAS-Bench-101: https://proceedings.mlr.press/v97/ying19a.html
- NAS-Bench-201: https://arxiv.org/abs/2001.00326
- OpenML benchmark documentation: https://docs.openml.org/benchmark/
- OpenML run metadata: https://docs.openml.org/concepts/runs/
- AutoML Benchmark: https://openml.github.io/automlbenchmark/
- MLPerf Inference documentation: https://docs.mlcommons.org/inference/
- Demsar, Statistical Comparisons of Classifiers over Multiple Data Sets:
  https://jmlr.org/papers/v7/demsar06a.html
- MAP-Elites: https://arxiv.org/abs/1504.04909
- NSGA-II: https://ieeexplore.ieee.org/document/996017
