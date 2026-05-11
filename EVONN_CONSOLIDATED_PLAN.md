# EvoNN Consolidated Plan

_Last consolidated: 2026-05-10._

## Purpose

This is the single active planning document for EvoNN.

It replaces the previous split planning surface:

- `ROADMAP.md`
- `EVONN_90_DAY_PLAN.md`
- `SEARCH_ENGINE_OUTPUT_PARITY_PLAN.md`
- `EXPANDED_BENCHMARK_COMPARISON_PLAN.md`
- `SHARED_SUBSTRATE_FOUNDATION_PLAN.md`
- `BENCHMARK_EXTRACTION_PLAN.md`
- `CONTENDER_EXPANSION_PLAN.md`
- `SEEDING_LADDERS_IMPLEMENTATION_PLAN.md`
- `.hermes/plans/*`
- package-local `IMPLEMENTATION_PLAN.md` bootstrap records

`VISION.md` and package `VISION.md` files remain product/research framing. This
file is the execution plan.

## North Star

EvoNN should become a local-first, benchmark-disciplined research platform for
discovering reusable neural structure across multiple search abstractions.

The project should be judged by whether it can produce fair, repeatable evidence
about:

- which search abstraction works on which benchmark families
- which systems improve over budget
- which wins survive against serious contenders
- which discovered structures transfer into later searches
- which changes deserve to be kept, reverted, or explored further

## Current State

The trust substrate is now real enough to operate from:

- `EvoNN-Compare` owns fair-matrix execution, trend artifacts, dashboards,
  benchmark audit, output-quality inspection, performance-baseline surfaces, and
  seeded/transfer compare surfaces.
- `EvoNN-Shared` owns shared contracts, budget/run identity helpers, manifest
  helpers, JSON writers, and real-LM cache validation.
- `shared-benchmarks` owns benchmark catalog and parity suites.
- `EvoNN-Contenders` provides the contender floor across tabular, image, and LM
  tasks.
- Prism, Topograph, Stratograph, and Primordia all participate in compare-grade
  matrix runs.
- The canonical dashboard now includes an Evidence Explorer and recent full-run
  budget overview backed by stored fair-matrix summaries.
- Tier A, Tier B, and Tier D audit cleanly; Tier C is intentionally exploratory
  until its promotion run requirements are met.
- Real LM caches are used for `tinystories_lm` and `wikitext2_lm`; smoke-only LM
  evidence is no longer the main LM surface for expanded lanes.

The main remaining strategic gap is not another planning layer. It is executing
the first real, auditable transfer/seeding loop and then using the evidence
surface to make research decisions.

## Operating Rules

- Keep one active plan: this file.
- Keep `VISION.md` and package `VISION.md` files as vision only, not execution
  issue lists.
- Do not recreate package-local branch plans unless a concrete branch needs a
  short PR-local checklist.
- Shared compare/fairness/budget/dashboard logic belongs in
  `EvoNN-Compare` or `EvoNN-Shared`.
- Search policy, training, candidate representation, mutation, and scoring stay
  package-local unless evidence justifies consolidation.
- Claims must link to fair-matrix artifacts, trend rows, dashboard slices, and
  exact run workspaces.
- A lane is never just "trusted"; name the operating state:
  `contract-fair`, `trusted-core`, `trusted-extended`, or an explicit
  exploratory/reference state.

## Plan Status Review

| Former plan | Consolidated status | What survives here |
| --- | --- | --- |
| `ROADMAP.md` | Superseded | Horizon structure and local-first/fair-comparison rules. |
| `EVONN_90_DAY_PLAN.md` | Superseded | Trusted lane, budget truth, trend-first workflow, contender floor, first seeding loop. |
| `SEARCH_ENGINE_OUTPUT_PARITY_PLAN.md` | Superseded | L0-L4 output quality model, measurable artifacts, performance baseline workflow. |
| `EXPANDED_BENCHMARK_COMPARISON_PLAN.md` | Superseded | Tier A/B/C/D ladder, benchmark audit, contender floor, ceiling/tie semantics. |
| `SHARED_SUBSTRATE_FOUNDATION_PLAN.md` | Archived into this plan | Foundation completion record and remaining substrate debt. |
| `BENCHMARK_EXTRACTION_PLAN.md` | Superseded | Shared benchmark/parity cleanup principle. |
| `CONTENDER_EXPANSION_PLAN.md` | Superseded | Contender floor hardening and optional enhanced baselines. |
| `SEEDING_LADDERS_IMPLEMENTATION_PLAN.md` | Superseded | Direct/staged transfer semantics and interpretation rules. |
| `.hermes/plans/*` | Superseded | Package advancement slices folded into package sections below. |
| Package `IMPLEMENTATION_PLAN.md` files | Obsolete bootstrap history | Current state captured in package READMEs and this plan. |
| Deprecated project plans | Obsolete | No active execution value for current monorepo. |

## Execution Horizon 1: Keep The Evidence Loop Boring

Goal:
make recurring compare runs routine, interpretable, and cheap enough to use
before and after meaningful changes.

Current accepted lanes:

- Tier A: `tier_a_contract`, budgets `16`, `64`
- Trusted daily lane: `tier1_core`, budgets `64`, `256`, `1000`
- Tier B current local research lane: `tier_b_core`, budgets `64`, `256`, `1000`
- Tier B expanded v2 lane: `tier_b_core_v2`, budgets `96`, `384`, `768`, `1536`
- Tier C exploratory lane: `tier_c_architecture_sensitive`, budgets `128`,
  `512`, `1024`, `2048`
- Tier D broad lane: `tier_d_broad_shared`, budgets `208`, `416`, `832`, `1664`

Required work:

- Keep `evonn-compare benchmark-audit` green for decision-grade packs.
- Keep contender-floor reports present for fair-matrix runs.
- Keep Evidence Explorer and recent full-run budget overview as the primary
  dashboard surface for cross-budget reading.
- Store important recurring runs in a durable workspace rather than only `.tmp`
  paths when they support claims.
- Keep Tier D separate from Tier A/B/C aggregate claims unless explicitly
  discussing broad-lane behavior.

Acceptance criteria:

- A contributor can run one command, open the dashboard, and understand which
  systems won, failed, or regressed across budgets and benchmarks.
- Recent 10-20 full comparison runs are visible from the dashboard.
- Real-LM benchmark behavior is visible without bespoke temporary dashboards.
- No benchmark enters a decision-grade lane without required contender-floor
  metadata and audit proof.

## Execution Horizon 2: Budget Truth And Output Parity

Goal:
make every engine comparable and measurable enough that budget, runtime, and
artifact differences do not hide behind score tables.

Output levels:

- L0: legacy output, not compare-grade
- L1: contract output with manifest/results
- L2: comparable output with summary/fairness coverage
- L3: measurable output with runtime/performance/diagnostic fields
- L4: decision-grade output with repeatability, variance, and decision support

Required work:

- Keep all engines at L3 for Tier A and `tier1_core@64`.
- Ensure `tier1_core@256` and `tier1_core@1000` either complete at L3 or report
  exact blockers by system and benchmark.
- Propagate wall-clock, backend, device, eval/sec, cache/reuse, failure, and
  skipped/unsupported metadata into trend rows and dashboards.
- Keep Linux-safe packages green on Linux CI and MLX-native truth paths explicit
  on macOS for Prism and Topograph.
- Use `performance-baseline` before optimization work and retain approve/scrap
  evidence after each optimization slice.

Acceptance criteria:

- `evonn-compare output-quality` can identify L-level gaps for every run.
- Dashboard rows show measurement downgrade reasons rather than only final
  scores.
- No system silently undercounts budget, hides failed candidates, or drops
  benchmarks from summaries.

## Execution Horizon 3: Benchmark Ladder And Contender Floor

Goal:
expand benchmark diversity only where EvoNN can still compare honestly.

Current benchmark policy:

- Required contenders must be dependency-light and reliable.
- Optional enhanced contenders may add pressure but cannot be required for
  completeness.
- Accuracy/F1/AUC-style ceiling ties are not strong evidence of superiority.
- Metrics without a natural ceiling use best-observed comparison only.
- LM claims must distinguish smoke, full-cache byte-level LM, and any later
  larger sequence benchmark.

Required work:

- Resolve the Tier B naming split deliberately:
  - `tier_b_core` is the current compact real-LM research lane.
  - `tier_b_core_v2` is the 12-benchmark expanded local research lane.
  - Pick one as the default recurring Tier B lane before adding more tiers.
- Keep Tier C exploratory until promotion requirements are met:
  - two clean `512` runs
  - one clean `1024` run
- Keep Tier D broad results separate and admitted-only.
- Add stronger contender floors only when they are reproducible and their
  optional dependency status is explicit.

Acceptance criteria:

- Tier A/B/D decision-grade packs audit with zero blockers.
- Tier C reports blockers or exploratory status honestly.
- Every benchmark has at least one required contender result before promotion.
- Dashboard separates engine wins, ceiling ties, contender-floor failures, and
  optional enhanced skips.

## Execution Horizon 4: Engine Advancement

Goal:
improve each search engine without duplicating shared substrate work or erasing
its scientific identity.

### Prism

Role:
default family-first operating engine.

Priorities:

- Maintain MLX-native truth while preserving Linux-safe fallback/test behavior.
- Stay benchmark-complete on Tier A and `tier1_core`.
- Improve candidate selection, family pressure, and per-family evidence.
- Improve runtime maturity without exploding local cost.
- Remain the default reference engine for broad family-aware search.

### Topograph

Role:
primary topology-first challenger.

Priorities:

- Maintain official-lane benchmark completeness.
- Improve topology-search quality and budget efficiency.
- Keep device/runtime metadata honest.
- Strengthen topology-specific evidence in summaries and dashboards.
- Prepare as a direct or staged transfer consumer.

### Stratograph

Role:
hierarchy-first challenger.

Priorities:

- Improve hierarchy-specific search quality, especially on broad and LM lanes.
- Keep backend portability and official-lane correctness stable.
- Expose hierarchy evidence: macro depth, cell reuse, motif emergence, and
  hierarchy-vs-flat ablation signals.
- Avoid becoming a weak Topograph clone; hierarchy must produce distinct
  evidence or remain a secondary challenger.

### Primordia

Role:
primitive-first motif and seed-source engine.

Priorities:

- Maintain primitive-search quality and artifact completeness.
- Improve benchmark completeness and budget accounting.
- Strengthen primitive-bank and seed-candidate artifacts.
- Become the first credible source for downstream seeded-vs-unseeded
  experiments.
- Avoid turning Primordia into a generic architecture engine; its value is cheap
  low-level structure.

### Contenders

Role:
honest external baseline floor.

Priorities:

- Keep required floor stable and dependency-light.
- Strengthen tabular, image, and LM baselines where cost is acceptable.
- Keep optional boosted-tree, torch CNN, and transformer paths explicit when
  skipped.
- Ensure contender wins and losses are budget-matched and interpretable.

### Compare

Role:
evidence and trust layer.

Priorities:

- Keep fair-matrix execution, trend rows, benchmark audit, contender floor,
  output quality, performance baseline, transfer surfaces, and dashboards
  coherent.
- Resist becoming a giant engine-specific adapter pile.
- Add only dashboard/report surfaces that answer real research decisions.

### Shared

Role:
shared substrate without search-core merger.

Priorities:

- Continue extracting only high-value common contracts and validators.
- Keep benchmark/parity resolution helpers moving toward one shared path.
- Keep seeding, budget, runtime, and telemetry models strict enough for Compare.
- Avoid importing engine-specific search semantics.

## Execution Horizon 5: First Auditable Transfer/Seeding Loop

Goal:
prove or falsify the central cumulative-search claim: discovered structure from
one system can improve another system under fair measurement.

Recommended next stretch:

1. Choose one first path:
   - `Primordia -> Prism` for broad family-first performance, or
   - `Primordia -> Topograph` for structural/topology transfer.
2. Define the seed artifact contract consumed by the target engine.
3. Run controlled lanes:
   - unseeded control
   - directly seeded
   - optionally staged seeded
4. Use the same pack, budget, seed, backend class, and contender-floor context.
5. Keep seeded and unseeded results separate in trends and dashboards.
6. Classify each result as `gain`, `regression`, `no_gain`, or `inconclusive`.

Recommended validation sequence:

- Tier B local budget first.
- Repeat with at least two seeds if any signal appears.
- Move to Tier C only after Tier B shows a non-noisy signal.
- Do not claim transfer success until the dashboard and trend rows show
  provenance, budget truth, and repeated evidence.

Acceptance criteria:

- Seed provenance is visible in manifests, summaries, trend rows, and dashboard.
- Seeded-vs-unseeded comparison is repeatable.
- Result interpretation is explicit and not folded into normal leaderboard
  totals.
- The transfer result informs whether the next quarter should emphasize
  Primordia, Topograph/Prism consumers, or a different search abstraction.

## Execution Horizon 6: Performance Optimization

Goal:
optimize only after measurement is stable.

Workflow:

1. Generate baseline performance measurements across multiple budgets.
2. Implement one performance improvement per branch/worktree.
3. Rerun the same measurement set.
4. Approve, scrap, or keep as inconclusive based on evidence.

Candidate areas:

- MLX backend batching and data movement
- Linux fallback vectorization
- candidate reuse and cache accounting
- early stopping/multi-fidelity scheduling where fairness semantics stay honest
- benchmark loader/cache hot paths
- dashboard/trend aggregation for large workspaces

Acceptance criteria:

- Optimization PRs include before/after performance artifacts.
- Score quality and budget semantics do not regress silently.
- Optimizations that only improve one benchmark while harming broad-lane evidence
  are treated as research tradeoffs, not automatic wins.

## Documentation Policy

Keep:

- `EVONN_CONSOLIDATED_PLAN.md` as the only active execution plan.
- `VISION.md` and package `VISION.md` files for research framing.
- `README.md`, `MONOREPO.md`, and package READMEs for operational entry points.
- Specific evidence artifacts in run workspaces when they support claims.

Do not keep:

- competing root planning files
- `.hermes` planning hierarchy
- package-local implementation plans that only record bootstrap history
- deprecated-project plans from old embedded repos

If a new branch needs planning, use either:

- a short issue/PR checklist, or
- a small section added to this consolidated plan if the work changes the
  roadmap.

## Near-Term Execution Order

1. Keep the dashboard/evidence loop healthy after every compare change.
2. Resolve the Tier B default lane naming split.
3. Run/update `tier1_core@64/256/1000` evidence with output-quality checks.
4. Execute the first Primordia-seeded transfer experiment.
5. Use the transfer result to decide whether the next major branch emphasizes:
   - seed-source quality,
   - target-engine seed consumption,
   - contender pressure,
   - or performance optimization.

## Definition Of Done For This Consolidation

- Active planning points to this file.
- Obsolete plan files and obsolete plan directories are removed.
- Active README/MONOREPO/package references do not point to removed plans.
- The repo has one execution plan and no competing active plan hierarchy.
