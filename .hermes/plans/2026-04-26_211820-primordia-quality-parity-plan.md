# Primordia Quality-Parity Improvement Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Raise `EvoNN-Primordia` from a credible primitive-first smoke engine into a genuinely high-quality engine whose search quality, observability, and compare results move materially closer to Prism, Topograph, and Stratograph without erasing Primordia’s distinct primitive-first thesis.

**Architecture:** Keep Primordia distinct internally, but improve it along five axes: (1) backend portability, (2) benchmark completeness/correctness, (3) search quality, (4) runtime maturity/observability, and (5) fairness/research usefulness on the shared compare substrate. The main changes are: first, introduce a Linux-capable fallback backend for correctness/CI and non-Apple-Silicon operation; second, replace the current round-robin mutated-seed evaluator with a bounded primitive-search loop that has memory, selection pressure, and reproducible artifacts.

**Tech Stack:** Python, MLX, Pydantic, uv workspace, shared-benchmarks, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

**Scope note:** This plan is intentionally a **full-engine advancement plan** for a dedicated Primordia branch. It is not limited by the current quarter-critical scope in `EVONN_90_DAY_PLAN.md`. That repo-level plan may prioritize the shared daily lane first; this plan assumes we are deliberately investing in Primordia itself as a serious engine. Merge-back should therefore happen in slices, but the branch goal is unapologetically engine advancement rather than minimum-quarter compliance.

---

## Current Context

Based on the current repo state:

- Primordia already has a self-contained MLX-first package boundary.
- It already exports compare-compatible `manifest.json`, `results.json`, `summary.json`, `primitive_bank_summary.json`, and `seed_candidates.json`.
- It already has richer package-local `inspect` / `report` surfaces than an early scaffold.
- It already participates on the shared compare/fairness substrate.
- But its **actual search loop is still extremely shallow** compared with Prism/Topograph/Stratograph.

The current core limitations appear to be:

1. `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
   - benchmark-by-benchmark round-robin evaluation
   - no persistent population/archive state
   - no parent selection, no cross-benchmark memory, no novelty/elite retention
   - mutated seeds are tied to slot index, not learned search pressure

2. `EvoNN-Primordia/src/evonn_primordia/runtime/training.py`
   - simple trainer with short local loops
   - enough for smoke, but not obviously strong enough for quality parity on harder lanes

3. `EvoNN-Primordia/src/evonn_primordia/genome.py`
   - decent mutation vocabulary, but no stronger search policy over that vocabulary
   - no explicit motif-equivalence / archive policy / dominance policy

4. Runtime maturity gap vs stronger engines
   - no checkpoint/resume/status surface comparable to Stratograph/Prism maturity
   - no run DB or richer lineage/history artifact surface
   - tests mostly prove smoke/export behavior, not long-horizon search quality behavior

5. Research usefulness gap
   - seed artifacts exist, but Primordia is not yet clearly winning often enough or stably enough to make downstream seeding especially compelling

---

## Desired End State

Primordia should become:

- a **trusted primitive-search engine** rather than only a compare participant
- runnable on both Apple Silicon MLX and a Linux-capable fallback path
- benchmark-complete on its named shared lanes, including current regression tasks
- capable of materially stronger results on tabular/image/text smoke and tier-1 research packs
- observably improving over time through trend artifacts and repeated runs
- able to emit **higher-confidence seed artifacts** for Topograph/Prism/Stratograph seeding studies
- still cheaper than architecture-scale search

Success does **not** mean “turn Primordia into Prism.”

Success means:
- backend portability without losing artifact/compare compatibility
- reliable benchmark completion on the official lanes
- stronger primitive search under bounded budgets
- better artifacts and operator observability
- more stable best-of-run quality
- better downstream usefulness of discovered primitives

## Explicit Branch Targets

This branch should aim higher than “good enough for the current quarter.” The target is:

1. Primordia becomes **benchmark-complete and budget-auditable** on `smoke` and `tier1_core`.
2. Primordia gains a **Linux-capable fallback backend** that preserves run/export/report semantics even if it is not yet quality-par with MLX.
3. Primordia becomes **meaningfully more competitive** on `tier1_core` at `64`, `256`, and `1000`.
4. Primordia becomes **operationally trustworthy** through status, resume, and artifact quality.
5. Primordia becomes **scientifically useful downstream** via stronger, more stable seed outputs.

Non-target:

- Do not try to turn Primordia into a direct Prism clone.
- Do not chase large-scale frontier benchmark coverage before `smoke` and `tier1_core` are strong.
- Do not add expensive search machinery whose cost erases Primordia’s strategic role.

---

## Primary Strategy

Prioritize work in this order:

1. **Backend portability first**
2. **Benchmark completeness and correctness second**
3. **Search-loop quality third**
4. **Objective shaping and training quality fourth**
5. **Runtime maturity and reproducibility fifth**
6. **Fairness / budget semantics sixth**
7. **Downstream seeding validation seventh**

That ordering matters because:

- a strong engine that only runs on one hardware/runtime surface is strategically fragile
- advanced search work is wasted if the engine still fails named lane benchmarks
- better reports around a weak search loop will not materially improve comparative results
- stronger seeding claims are only credible once the engine itself is complete and stable

---

## Phase 1 — Introduce a Linux-capable fallback backend

**Objective:** Make Primordia runnable beyond Apple Silicon by introducing a fallback backend that preserves artifacts, budget semantics, and compare/export surfaces on Linux and non-MLX hosts.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/runtime/training.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/families/compiler.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/families/models.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/config.py`
- Create: `EvoNN-Primordia/src/evonn_primordia/runtime/backends.py`
- Create: `EvoNN-Primordia/tests/test_runtime_backends.py`
- Modify: `EvoNN-Primordia/README.md`

**Work:**
1. Introduce an explicit runtime/backend selector instead of an implicit MLX-only path.
2. Keep MLX as the high-quality primary backend on Apple Silicon.
3. Add a Linux-capable fallback backend for:
   - smoke runs
   - export/report generation
   - compare/fairness validation
   - package tests and CI
4. Preserve the same top-level run artifacts across backends:
   - `summary.json`
   - `trial_records.json`
   - `primitive_bank_summary.json`
   - `seed_candidates.json`
   - compare/export artifacts
5. Make backend identity explicit in artifacts:
   - `runtime_backend`
   - `runtime_version`
   - any fallback marker needed for honest interpretation
6. Prefer a minimal correctness backend first, not a rushed “high-quality Linux parity” implementation.
7. Keep the design open for a later stronger Linux backend without forcing that into phase 1.

**Recommended design direction:**
- mirror the spirit of Stratograph’s `mlx` vs `numpy-fallback` split
- make the portability surface explicit in code rather than hiding it behind scattered imports
- do not move model-runtime code into `evonn_shared`
- do not promise metric parity between MLX and fallback in this phase

**Why first:** Portability is not just convenience. It reduces platform lock-in, makes CI and non-Mac validation real, and gives later Primordia quality work a broader execution surface.

**Validation:**
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_runtime_backends.py EvoNN-Primordia/tests/test_smoke.py`
- one smoke run on MLX
- one smoke run on the fallback backend

**Exit criteria:**
- Primordia can execute a smoke run on a non-MLX host
- artifact schema stays compare-compatible across backends
- backend metadata is explicit and honest in package and compare exports
- CI/package tests can exercise at least one non-MLX runtime path

---

## Phase 2 — Establish a real Primordia baseline and close benchmark-completeness gaps

**Objective:** Make current Primordia measurable enough that later improvements can be judged honestly, while fixing the obvious “not yet a complete engine” gaps on official lanes.

**Files to modify:**
- Create: `EvoNN-Primordia/configs/smoke.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval64.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval256.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval1000.yaml`
- Modify: `EvoNN-Primordia/README.md`
- Modify: `EvoNN-Primordia/tests/test_cli.py`
- Modify: `EvoNN-Primordia/tests/test_smoke.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/runtime/training.py`
- Modify: `EvoNN-Primordia/tests/test_parity.py`

**Work:**
1. Add canonical Primordia config files for repeated local runs.
2. Document one official smoke lane and one official tier-1 lane.
3. Add tests proving config loading and CLI examples stay valid.
4. Make benchmark-completeness an explicit tracked surface:
   - smoke
   - tier1_core @ 64 evals
   - tier1_core @ 256 evals
   - tier1_core @ 1000 evals
   - classification coverage
   - regression coverage
   - any remaining language-modeling caveats
5. Record a baseline evaluation matrix for later comparison:
   - smoke
   - tier1_core @ 64 evals
   - tier1_core @ 256 evals
   - tier1_core @ 1000 evals
6. Fix the current named benchmark failures that prevent Primordia from being benchmark-complete on the official shared lane, especially regression failures.

**Why first:** Without repeatable configs, named lanes, and benchmark completion on those lanes, later “quality improvements” are hard to trust.

**Validation:**
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_cli.py EvoNN-Primordia/tests/test_smoke.py`
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_parity.py`
- one real `primordia run` for smoke
- one real `tier1_core` rerun confirming whether Primordia is benchmark-complete or not

**Exit criteria:**
- official configs exist and are documented
- benchmark failures on the named official lanes are enumerated and reproducible
- current regression failures are fixed or reduced to an explicit blocked list
- the baseline scoreboard is recorded for later phase-to-phase comparison

---

## Phase 3 — Replace slot-based search with a bounded elite/archive loop

**Objective:** Upgrade Primordia from round-robin candidate evaluation to a minimal but real evolutionary search process.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/genome.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/config.py`
- Create: `EvoNN-Primordia/src/evonn_primordia/search_state.py`
- Create: `EvoNN-Primordia/tests/test_search_state.py`
- Modify: `EvoNN-Primordia/tests/test_smoke.py`

**Work:**
1. Add explicit search-state structures:
   - candidate record
   - elite archive
   - benchmark-local bests
   - family-level bests
   - lineage / parent reference
2. Extend config with bounded search-policy knobs:
   - `population_size`
   - `elite_fraction`
   - `mutation_rounds_per_parent`
   - `family_exploration_floor`
   - `novelty_weight`
   - `complexity_penalty_weight`
   - `max_candidates_per_benchmark`
3. Replace the current `slot_index -> repeat_index -> mutate_seed_genome` loop with:
   - initialize benchmark-compatible seed pool
   - evaluate initial pool
   - retain elites
   - sample parents by quality + diversity
   - mutate offspring
   - re-evaluate until per-benchmark budget is exhausted
4. Persist lineage metadata in `trial_records.json`:
   - `parent_genome_id`
   - `mutation_operator`
   - `generation`
   - `novelty_score`
   - `complexity_score`
5. Keep the loop cheap-first; do **not** add expensive crossover or giant archives in the first pass.
6. Make the search policy observable from artifacts so we can tell whether improvement comes from better search pressure rather than random luck.

**Why this is the highest-value improvement:** Current Primordia mostly samples and lightly mutates. Stronger results will come more from better search pressure than from more report polish.

**Validation:**
- targeted new tests for elite retention, per-benchmark budget exhaustion, and lineage fields
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_search_state.py EvoNN-Primordia/tests/test_smoke.py`

**Exit criteria:**
- slot-index-driven search is gone from the core loop
- lineage fields exist and are non-trivial in real run artifacts
- at least one named lane shows better best-of-run outcomes without breaking budget accounting

---

## Phase 4 — Improve objective shaping so “best primitive” means something

**Objective:** Make candidate ranking less naive and more robust across benchmark families.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/runtime/training.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/report.py`
- Create: `EvoNN-Primordia/src/evonn_primordia/objectives.py`
- Create: `EvoNN-Primordia/tests/test_objectives.py`

**Work:**
1. Add canonical derived scores per candidate:
   - raw task metric
   - normalized quality score
   - parameter-efficiency score
   - train-time efficiency score
   - complexity penalty
   - composite search score
2. Separate:
   - compare-facing metric of record
   - internal search score used for parent selection
3. Add benchmark-family-aware complexity penalties:
   - image/text primitives may tolerate different widths than tabular
4. Add optional “minimum viability” filters:
   - catastrophic failure exclusion
   - degenerate output exclusion
   - unstable loss exclusion beyond current NaN handling
5. Surface the composite score and ranking rationale in run artifacts.
6. Add ablation-friendly artifact fields so we can compare “metric-only selection” vs “composite selection” honestly.

**Important guardrail:** Compare exports must still report the true benchmark metric, not the internal composite score.

**Validation:**
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_objectives.py EvoNN-Primordia/tests/test_smoke.py`

**Exit criteria:**
- compare-facing metric remains untouched
- internal search ranking is explainable from artifacts
- objective shaping improves at least one named lane without increasing benchmark failure rate

---

## Phase 5 — Strengthen training/runtime quality without exploding cost

**Objective:** Raise result quality per evaluation while preserving Primordia’s cheap-first role.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/runtime/training.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/families/models.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/families/compiler.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/config.py`
- Modify: `EvoNN-Primordia/tests/test_smoke.py`

**Work:**
1. Improve trainer quality in bounded ways:
   - stronger default schedulers
   - better early-stopping criteria
   - optional warmup for text/attention families
   - optional label smoothing for classification
   - optional gradient accumulation for larger text/image candidates
2. Improve family implementations only where cheap gains are likely:
   - better normalization defaults
   - safer residual rules
   - attention head / embedding validation tightening
   - lightweight regularization improvements
3. Add per-family training overrides in config, not hardcoded special cases in the loop.
4. Ensure each improvement is benchmarked against wall-clock cost.
5. Prioritize fixes that improve regression robustness and cheap image/text candidates before adding new family breadth.

**Do not do yet:**
- giant model families
- heavyweight data pipelines
- architecture-scale training schedules

**Validation:**
- smoke still passes
- one real tier1_core run shows non-regressive wall-clock and better best-of-run metrics on at least a subset of tasks

**Exit criteria:**
- trainer changes are measurable in artifacts, not just “felt”
- wall-clock per evaluation remains within an explicitly accepted bound
- official lanes remain benchmark-complete

---

## Phase 6 — Add runtime maturity features expected of stronger engines

**Objective:** Make Primordia feel like a serious engine operationally, not only algorithmically.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/cli.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/report.py`
- Create: `EvoNN-Primordia/src/evonn_primordia/status.py`
- Create: `EvoNN-Primordia/tests/test_status.py`
- Modify: `EvoNN-Primordia/README.md`

**Work:**
1. Emit package-local progress artifacts during runs:
   - `status.json`
   - `checkpoint.json`
   - optional `lineage.jsonl`
2. Add bounded resume support:
   - resume same run dir
   - continue remaining benchmark budget
   - preserve prior trial records / elites safely
3. Upgrade `inspect`/`report` to show:
   - current state
   - completed vs remaining budget
   - family leaders over time
   - best primitive lineage
   - failure pattern summaries
4. Keep artifacts compatible with compare/export surfaces.
5. Make interruption/restart behavior boring and safe enough for longer `tier1_core` and future larger runs.

**Why this matters:** Prism/Stratograph are not only stronger because of models; they are also easier to trust and inspect mid-run.

**Validation:**
- targeted tests for checkpoint/status/resume
- artifact-backed CLI smoke for `inspect` and `report`

**Exit criteria:**
- interrupted runs can be resumed without silently corrupting artifacts
- package-local inspection is good enough to reason about a live or partial run without opening raw JSON by hand

---

## Phase 7 — Harden Primordia budget semantics for fairer compare results

**Objective:** Make Primordia’s “evaluation_count” honest and stable enough for tier1_core fair-matrix use.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/symbiosis.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/report.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/config.py`
- Modify: `EvoNN-Primordia/tests/test_parity.py`
- Modify: `EvoNN-Primordia/tests/test_smoke.py`

**Work:**
1. Define internal accounting explicitly for:
   - attempted evaluations
   - successful evaluations
   - failed evaluations
   - skipped / invalid candidates
   - resumed evaluations
   - cache reuse if later introduced
2. Persist both raw and normalized budget fields.
3. Align compare export so `manifest.json` and `summary.json` explain exactly what counted.
4. Add tolerance tests against compare expectations.
5. Ensure quality-improvement work from earlier phases did not quietly distort evaluation accounting.

**Key point:** This does not make Primordia identical to other engines; it makes the budget semantics auditable.

**Validation:**
- parity/export tests
- one Compare fair-matrix smoke run with Primordia checked for budget/fairness acceptance

**Exit criteria:**
- exported budget fields are explicit and explainable
- Compare accepts Primordia artifacts without special pleading
- the engine can improve without losing compare honesty

---

## Phase 8 — Turn primitive-bank outputs into stronger downstream seed evidence

**Objective:** Make Primordia’s value visible not just through its own metrics but through downstream transfer usefulness.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/seeding.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/report.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Create: `EvoNN-Primordia/tests/test_seeding.py`

**Work:**
1. Improve `seed_candidates.json` ranking to incorporate:
   - repeated benchmark wins
   - family-conditioned transfer relevance
   - stability across reruns
   - complexity / efficiency constraints
2. Add provenance fields:
   - supporting benchmarks
   - repeat-run support count
   - median quality by benchmark group
3. Add one controlled downstream experiment:
   - Primordia-seeded vs unseeded Topograph smoke/tier1_core
4. Treat downstream win rate as a primary Primordia KPI.
5. Keep downstream consumer integration work as a dependent follow-up slice, not part of the core Primordia branch implementation unless needed for the experiment harness.

**Why this matters:** Primordia’s strongest long-term claim is likely transfer value, not raw head-to-head dominance on every benchmark.

**Exit criteria:**
- seed rankings are no longer simple one-shot winners
- provenance is rich enough to audit why a seed was recommended
- one seeded-vs-unseeded experiment exists with honest results, even if the result is “not yet better”

---

## Phase 9 — Introduce a real Primordia evaluation scoreboard

**Objective:** Decide whether Primordia is actually closing the gap to the stronger engines.

**Files to modify:**
- Create: `EvoNN-Primordia/docs/QUALITY_SCORECARD.md`
- Modify: `EvoNN-Primordia/README.md`
- Optionally modify: compare-history/dashboard inputs later

**Work:**
1. Track these metrics per named lane:
   - best-of-run median quality
   - benchmark coverage success rate
   - failure rate
   - median wall-clock per evaluation
   - stability over 3 reruns
   - downstream seed usefulness
2. Define “close to parity” in operational terms instead of vibes.
3. Add a small table comparing Primordia vs Prism/Topograph/Stratograph on shared lanes.
4. Record both absolute performance and relative ranking so “improved but still far behind” is visible.

**Suggested parity target:**
- not equal absolute quality on every task
- but within a declared band on smoke/tier1_core while remaining materially cheaper and useful for seeding

**Suggested operational targets:**
- benchmark-complete on `smoke` and `tier1_core`
- no unresolved accounting caveat on exported compare artifacts
- improved rank or win share on at least one named `tier1_core` budget
- lower failure rate than the current baseline
- stable enough reruns that seed recommendations stop looking random

---

## Likely Execution Order

1. Phase 1 backend portability surface
2. Phase 2 baseline configs/docs and benchmark completion
3. Phase 3 elite/archive search loop
4. Phase 4 objective shaping
5. Phase 5 trainer/runtime improvements
6. Phase 6 resume/status maturity
7. Phase 7 budget semantics hardening
8. Phase 8 downstream seeding validation
9. Phase 9 scorecard/docs

---

## Tests And Validation Matrix

### Package tests
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests`

### Focused tests to add/run
- `EvoNN-Primordia/tests/test_runtime_backends.py`
- `EvoNN-Primordia/tests/test_search_state.py`
- `EvoNN-Primordia/tests/test_objectives.py`
- `EvoNN-Primordia/tests/test_status.py`
- `EvoNN-Primordia/tests/test_seeding.py`

### Runtime checks
- smoke run via fallback backend
- smoke run via official config
- tier1_core eval64 run
- tier1_core eval256 run
- tier1_core eval1000 run
- Compare `fair-matrix --preset smoke`
- Compare `tier1_core` lane at `64`
- Compare `tier1_core` lane at `256`
- Compare `tier1_core` lane at `1000`

### Evidence requirement before calling the effort successful
- Primordia becomes benchmark-complete on the official named lanes
- Primordia best-of-run quality improves on at least one shared lane and one named `tier1_core` budget
- failure rate does not regress badly and preferably improves
- budget accounting remains compare-auditable
- seed artifacts become more stable across reruns
- one downstream seeded-vs-unseeded result exists and is artifact-backed

## Merge-Back Strategy

Because this plan is intentionally broader than the current quarter-critical repo scope, merge-back should happen in disciplined slices:

1. backend portability surface
2. correctness/completeness fixes
3. search-loop infrastructure
4. objective-shaping and trainer improvements
5. runtime maturity surfaces
6. budget/export semantics
7. seeding evidence improvements

This branch should optimize for engine advancement, but integration back into `main` should stay reviewable.

---

## Risks And Tradeoffs

### Risk 1: Overgrowing Primordia
If the loop becomes too expensive, Primordia loses its strategic role.

**Mitigation:** Keep cheap-first budgets as a hard constraint and track wall-clock per evaluation.

### Risk 2: Accidentally turning Primordia into a weak Prism clone
If improvements focus only on larger models and more training, the primitive-first research claim gets diluted.

**Mitigation:** Prioritize primitive archive quality, motif reuse, and downstream seed value over brute-force metric chasing.

### Risk 3: Better local scores but worse compare honesty
Composite search scores can accidentally leak into compare metrics.

**Mitigation:** Keep compare metric-of-record separate from internal search score.

### Risk 4: Too much runtime maturity work before search quality work
Nice reports around a weak loop do not improve results.

**Mitigation:** Execute Phases 2–4 before heavy polish.

---

## Short Version

If only three things get done, they should be:

1. **Replace the current slot-based loop with a real bounded elite/archive evolutionary loop.**
2. **Improve internal candidate scoring so search selection is smarter than raw one-shot metric ranking.**
3. **Add enough status/resume/budget truth that Primordia becomes a trusted, repeatable research lane.**

That combination is the most likely path to making Primordia meaningfully closer to the stronger EvoNN engines while keeping its primitive-first identity.
