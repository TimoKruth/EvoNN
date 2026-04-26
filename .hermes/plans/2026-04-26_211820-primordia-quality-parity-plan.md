# Primordia Quality-Parity Improvement Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Raise `EvoNN-Primordia` from a credible primitive-first smoke engine into a stronger daily research engine whose run quality, observability, and compare results are closer to Prism, Topograph, and Stratograph without erasing Primordia’s distinct primitive-first thesis.

**Architecture:** Keep Primordia distinct internally, but improve it along three axes: (1) search quality, (2) runtime maturity/observability, and (3) fairness/research usefulness on the shared compare substrate. The main change is to replace the current round-robin mutated-seed evaluator with a bounded primitive-search loop that has memory, selection pressure, and reproducible artifacts.

**Tech Stack:** Python, MLX, Pydantic, uv workspace, shared-benchmarks, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

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

- a **trusted primitive-search lane** rather than only a compare participant
- capable of stronger results on tabular/image/text smoke and tier-1 research packs
- observably improving over time through trend artifacts and repeated runs
- able to emit **higher-confidence seed artifacts** for Topograph/Prism/Stratograph seeding studies
- still cheaper than architecture-scale search

Success does **not** mean “turn Primordia into Prism.”

Success means:
- stronger primitive search under bounded budgets
- better artifacts and operator observability
- more stable best-of-run quality
- better downstream usefulness of discovered primitives

---

## Primary Strategy

Prioritize work in this order:

1. **Search-loop quality first**
2. **Runtime maturity and reproducibility second**
3. **Fairness / budget semantics third**
4. **Downstream seeding validation fourth**

That ordering matters because better reports around a weak search loop will not materially improve Primordia’s comparative results.

---

## Phase 1 — Establish a real Primordia maturity baseline

**Objective:** Make current Primordia measurable enough that later improvements can be judged honestly.

**Files to modify:**
- Create: `EvoNN-Primordia/configs/smoke.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval64.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval256.yaml`
- Create: `EvoNN-Primordia/configs/tier1_core_eval1000.yaml`
- Modify: `EvoNN-Primordia/README.md`
- Modify: `EvoNN-Primordia/tests/test_cli.py`
- Modify: `EvoNN-Primordia/tests/test_smoke.py`

**Work:**
1. Add canonical Primordia config files for repeated local runs.
2. Document one official smoke lane and one official tier-1 lane.
3. Add tests proving config loading and CLI examples stay valid.
4. Record a baseline evaluation matrix for later comparison:
   - smoke
   - tier1_core @ 64 evals
   - tier1_core @ 256 evals
   - tier1_core @ 1000 evals

**Why first:** Without repeatable configs and named lanes, later “quality improvements” are hard to trust.

**Validation:**
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_cli.py EvoNN-Primordia/tests/test_smoke.py`
- one real `primordia run` for smoke

---

## Phase 2 — Replace slot-based search with a bounded elite/archive loop

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

**Why this is the highest-value improvement:** Current Primordia mostly samples and lightly mutates. Stronger results will come more from better search pressure than from more report polish.

**Validation:**
- targeted new tests for elite retention, per-benchmark budget exhaustion, and lineage fields
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_search_state.py EvoNN-Primordia/tests/test_smoke.py`

---

## Phase 3 — Improve objective shaping so “best primitive” means something

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

**Important guardrail:** Compare exports must still report the true benchmark metric, not the internal composite score.

**Validation:**
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests/test_objectives.py EvoNN-Primordia/tests/test_smoke.py`

---

## Phase 4 — Strengthen training/runtime quality without exploding cost

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

**Do not do yet:**
- giant model families
- heavyweight data pipelines
- architecture-scale training schedules

**Validation:**
- smoke still passes
- one real tier1_core run shows non-regressive wall-clock and better best-of-run metrics on at least a subset of tasks

---

## Phase 5 — Add runtime maturity features expected of stronger engines

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

**Why this matters:** Prism/Stratograph are not only stronger because of models; they are also easier to trust and inspect mid-run.

**Validation:**
- targeted tests for checkpoint/status/resume
- artifact-backed CLI smoke for `inspect` and `report`

---

## Phase 6 — Harden Primordia budget semantics for fairer compare results

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

**Key point:** This does not make Primordia identical to other engines; it makes the budget semantics auditable.

**Validation:**
- parity/export tests
- one Compare fair-matrix smoke run with Primordia checked for budget/fairness acceptance

---

## Phase 7 — Turn primitive-bank outputs into stronger downstream seed evidence

**Objective:** Make Primordia’s value visible not just through its own metrics but through downstream transfer usefulness.

**Files to modify:**
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/seeding.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/export/report.py`
- Modify: `EvoNN-Primordia/src/evonn_primordia/pipeline.py`
- Modify: `EvoNN-Topograph` seeding consumer paths (future execution phase)
- Modify: `EvoNN-Prism` seeding consumer paths (future execution phase)
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

**Why this matters:** Primordia’s strongest long-term claim is likely transfer value, not raw head-to-head dominance on every benchmark.

---

## Phase 8 — Introduce a real Primordia evaluation scoreboard

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

**Suggested parity target:**
- not equal absolute quality on every task
- but within a declared band on smoke/tier1_core while remaining materially cheaper and useful for seeding

---

## Likely Execution Order

1. Phase 1 baseline configs/docs
2. Phase 2 elite/archive search loop
3. Phase 3 objective shaping
4. Phase 4 trainer/runtime improvements
5. Phase 5 resume/status maturity
6. Phase 6 budget semantics hardening
7. Phase 7 downstream seeding validation
8. Phase 8 scorecard/docs

---

## Tests And Validation Matrix

### Package tests
- `uv run --package evonn-primordia --extra dev pytest -q EvoNN-Primordia/tests`

### Focused tests to add/run
- `EvoNN-Primordia/tests/test_search_state.py`
- `EvoNN-Primordia/tests/test_objectives.py`
- `EvoNN-Primordia/tests/test_status.py`
- `EvoNN-Primordia/tests/test_seeding.py`

### Runtime checks
- smoke run via official config
- tier1_core eval64 run
- tier1_core eval256 run
- Compare `fair-matrix --preset smoke`
- later Compare tier1_core lane when budget semantics are hardened

### Evidence requirement before calling the effort successful
- Primordia best-of-run quality improves on at least one shared lane
- failure rate does not regress badly
- budget accounting remains compare-auditable
- seed artifacts become more stable across reruns

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
