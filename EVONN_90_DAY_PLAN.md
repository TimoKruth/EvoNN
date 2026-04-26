# EvoNN 90-Day Plan

_As of 2026-04-26._

## Purpose

Turn EvoNN from a repo with a strong shared substrate into a **trusted, recurring research loop**.

The next 90 days should optimize for one main outcome:

> **`tier1_core` becomes a genuinely trusted daily research lane with fair, trendable, repeatable artifacts across the main EvoNN systems.**

This means the project should move from mostly infrastructure progress to a compounding research platform.

## Day-90 End State

By day 90, EvoNN should have:

- one trusted daily compare lane: `tier1_core`
- explicit project-wide budget/accounting semantics
- trend artifacts used as the default decision surface
- CI coverage for the trust layer, not just selected engines
- a stronger, budget-matched contender floor
- one auditable transfer/seeding experiment path
- one clearly chosen primary research claim

## Primary Priorities

1. **Make `tier1_core` the trusted daily lane**
2. **Define and enforce budget truth across all engines**
3. **Use trend artifacts as the default basis for decisions**
4. **Harden Compare + Shared + sibling package validation in CI**
5. **Strengthen Contenders so wins mean something**
6. **Run one first-class seeding/transfer loop**
7. **Choose the primary research claim and align milestones to it**

## Plan Alignment Map

Use this file as the quarter execution document.

- `ROADMAP.md` remains the long-horizon umbrella sequence
- `SHARED_SUBSTRATE_FOUNDATION_PLAN.md` is now the completed-foundation record
  plus remaining substrate-debt list
- `BENCHMARK_EXTRACTION_PLAN.md` now feeds only the benchmark/parity cleanup
  needed for trusted daily-lane comparison
- `CONTENDER_EXPANSION_PLAN.md` stays valid, but contender hardening now
  outranks broad family expansion
- `SEEDING_LADDERS_IMPLEMENTATION_PLAN.md` stays valid, but one auditable seeded
  experiment outranks full ladder breadth this quarter
- `EvoNN-Primordia/IMPLEMENTATION_PLAN.md` and
  `EvoNN-Stratograph/IMPLEMENTATION_PLAN.md` are now mostly package-history plus
  current-role notes, not the primary repo execution plan

## Non-Goals For This Window

Avoid spending this 90-day window on:

- adding many new benchmark tiers before `tier1_core` is trusted
- broad new engine feature sprawl without compare impact
- large new abstractions that do not tighten fairness, evidence, or transfer
- frontier benchmark expansion before Tier A/Tier B daily loops are stable

## Workstreams

### 1) Trusted `tier1_core` Daily Lane

### Goal
Make `tier1_core` the routine daily research surface, not just an aspirational next lane.

### Why this matters
The biggest current maturity gap is not missing engines. It is whether higher-budget runs remain genuinely fair and comparable once budgets increase beyond smoke.

### Deliverables
- canonical `tier1_core` pack + budget presets for at least:
  - `64`
  - `256`
  - `1000`
- stable end-to-end Compare path for Prism, Topograph, and Contenders
- artifact completeness and fairness checks treated as required, not optional
- repeat reruns stored in one canonical trend workspace

### Acceptance criteria
- repeated `tier1_core` reruns produce comparable artifacts without manual caveats
- fairness failures are explicit and actionable when they happen
- no ambiguity remains about whether a run belongs in the trusted daily lane

---

### 2) Budget Truth And Fairness Semantics

### Goal
Define one project-wide answer to: **what counts as one evaluation?**

### Why this matters
Comparisons stop being trustworthy when engines count work differently. Reuse, cached candidates, failed candidates, and resumed runs need one shared interpretation.

### Deliverables
- one written budget/accounting policy document
- shared budget semantics enforced in Compare validation and package exports
- explicit treatment for:
  - candidate reuse / inheritance
  - cached evaluations
  - failed / invalid candidates
  - resumed runs
  - partial runs
  - exported wall-time and hardware metadata

### Acceptance criteria
- all participating systems emit the minimum required budget semantics
- Compare can clearly mark fair vs non-fair comparisons on budget/accounting grounds
- Tier1 comparisons at `64/256/1000` no longer fail due to implicit accounting mismatches

---

### 3) Trend Artifacts As The Default Decision Surface

### Goal
Make longitudinal evidence the normal workflow.

### Why this matters
The repo already has structured trend outputs. The next step is cultural and operational: stop treating markdown snapshots as the main decision surface.

### Deliverables
- one canonical trend workspace layout
- one simple refresh/report workflow
- standard recurring views for:
  - engine vs engine over time
  - benchmark-level drift
  - fairness/failure drift
  - budget-normalized improvement/regression
- docs that say trend outputs are the default review surface

### Acceptance criteria
- repeated runs append cleanly to one stable dataset
- a contributor can answer "did this improve anything?" from trend artifacts first
- one-off markdown is supportive, not primary

---

### 4) CI And Validation Coverage For The Trust Layer

### Goal
Give Compare, Shared, Contenders, Primordia, and Stratograph first-class automated validation.

### Why this matters
The trust layer cannot depend mainly on manual local checks.

### Deliverables
- GitHub Actions for:
  - `EvoNN-Shared`
  - `EvoNN-Compare`
  - `EvoNN-Contenders`
  - `EvoNN-Primordia`
  - `EvoNN-Stratograph`
- Linux CI for non-MLX packages
- keep macOS CI where MLX runtime truth matters for Prism/Topograph
- package validation matrix documented in repo docs

### Acceptance criteria
- every core package has a clear automated validation path
- substrate regressions are caught before merge more often than after
- Linux-safe vs macOS-only coverage boundaries are explicit

---

### 5) Contender Floor Hardening

### Goal
Make Contenders strong enough that wins against it carry weight.

### Why this matters
If the contender floor lags, EvoNN risks becoming only internally comparative.

### Deliverables
- harden the current default contender path for the daily lane
- remove fragile cross-package assumptions where possible
- expand the contender floor incrementally, starting with low-risk additions
- keep contender budgets visibly normalized against compare lanes

### Recommended sequence
1. contender runtime/registry cleanup where needed
2. harden current lane reliability and export semantics
3. add low-risk baseline expansion such as stronger classical baselines
4. defer bigger runtime jumps until the daily lane is trusted

### Acceptance criteria
- contender exports are stable and budget-matched on the trusted lane
- wins and losses against contenders are easy to interpret
- contender regressions are visible in trend views

---

### 6) First-Class Transfer / Seeding Loop

### Goal
Prove one real compounding-research path, not just multiple isolated engines.

### Why this matters
Primordia-to-upstream seeding and cross-system transfer are among EvoNN's most strategically distinctive claims.

### Deliverables
- one auditable seed artifact contract used in practice
- one first consumer path, likely:
  - Primordia -> Prism, or
  - Primordia -> Topograph
- explicit comparison buckets for:
  - unseeded
  - direct seeded
  - staged seeded, if used

### Acceptance criteria
- one repeatable seeded vs unseeded experiment exists
- seed provenance is visible in artifacts and compare outputs
- the result is interpretable as either a real gain, no gain, or inconclusive

---

### 7) Primary Research Claim Selection

### Goal
Choose the main story EvoNN is trying to prove over the next quarter.

### Candidate claims
- best **on-device architecture search** loop
- best **shared compare substrate** for sibling search systems
- strongest **transfer/seeding workflow** across architecture-search layers
- best **budget-aware challenger** workflow against fixed baselines

### Recommendation
Do not try to lead with all four at once.

The most coherent default claim for this quarter is:

> **EvoNN is a credible MLX-native research system with a trusted compare loop, trendable evidence, and an emerging transfer path from primitive discovery into higher-level search.**

### Acceptance criteria
- milestone choices clearly support the chosen claim
- the daily lane, reports, and docs all tell the same story
- success can be explained in one paragraph without listing every subsystem

## 90-Day Sequencing

## Weeks 1-2
- write and ratify the budget/accounting policy
- identify current `tier1_core` fairness failures at `64/256/1000`
- define the canonical daily-lane workspace and artifact layout
- add CI skeletons for Shared and Compare first

### Exit criteria
- open fairness/accounting gaps are enumerated clearly
- one canonical place exists for trusted daily-lane reruns
- Compare/Shared CI is live or ready to merge

## Weeks 3-4
- implement budget/accounting fixes in exports and Compare validation
- tighten fair-vs-nonfair reporting in Compare outputs
- add CI for Contenders, Primordia, and Stratograph
- start recurring `tier1_core` reruns in the canonical workspace

### Exit criteria
- first reruns at `64` produce trusted or clearly explained outcomes
- non-MLX core packages have automated validation coverage

## Weeks 5-6
- push `256` lane fairness convergence
- improve trend-query/report ergonomics
- harden contender exports and default contender participation in the lane
- remove any easy benchmark/parity resolution drift that affects fairness

### Exit criteria
- `256` reruns are mostly comparable without manual interpretation
- trend artifacts are practical to use for go/no-go decisions

## Weeks 7-8
- stabilize `1000` tier1 runs enough for meaningful repeated comparison
- finish the most important remaining budget-semantics gaps
- document trusted-lane operating procedure
- choose the primary research claim explicitly in docs/planning

### Exit criteria
- `64/256/1000` all have a defined interpretation and operating status
- the repo has one explicit quarter-level research claim

## Weeks 9-10
- implement the first auditable seeding consumer path
- run seeded vs unseeded comparison on the trusted lane
- ensure seeded runs remain clearly separated in trend/report surfaces

### Exit criteria
- first real transfer/seeding experiment has been executed and exported
- compare outputs preserve seed provenance cleanly

## Weeks 11-13
- review trend data from repeated trusted-lane runs
- decide whether next quarter should emphasize:
  - stronger contender floor
  - deeper transfer/seeding work
  - Stratograph maturity as a stronger challenger
  - harder benchmark tiers
- publish a quarter-end evidence summary from the trend artifacts

### Exit criteria
- next-quarter direction is chosen from evidence, not taste alone
- EvoNN has shifted from infrastructure-heavy progress to recurring comparative research

## Concrete Milestones

### Milestone A: Budget Truth Baseline
- written accounting policy exists
- Compare enforces the policy where possible
- all main systems export the minimum required budget semantics

### Milestone B: Trusted Tier1 Lane
- `tier1_core` is the daily lane
- `64/256/1000` runs are interpretable and repeatable
- fairness caveats are explicit and rare

### Milestone C: Trend-First Workflow
- trend datasets are append-only and stable
- reports answer improvement/regression questions from structured artifacts first

### Milestone D: Honest Baseline Floor
- contender lane is stable and budget-matched
- baseline results remain visible in daily comparisons

### Milestone E: First Compounding Loop
- one seeded experiment path is live
- seeded vs unseeded outcomes are measurable and auditable

## Metrics To Watch

- number of trusted `tier1_core` reruns completed
- number of fair vs non-fair comparison outcomes
- frequency of budget/accounting-related fairness failures
- trend dataset growth without schema churn
- contender participation stability on the daily lane
- seeded vs unseeded delta on the chosen transfer experiment
- time required to evaluate whether a change improved anything

## Risks

- spending too long on substrate cleanup without increasing evidence velocity
- declaring comparisons fair before budget/accounting semantics are actually aligned
- letting contender quality lag behind search-system polish
- mixing Linux-safe validation confidence with MLX-native runtime truth
- trying to advance too many research stories at once

## Bottom Line

The next 90 days should be judged mainly by this question:

> **Did EvoNN establish one trusted daily compare lane with budget truth, trendable evidence, and at least one real compounding transfer path?**

If yes, the project moves from promising infrastructure to a credible research platform.
