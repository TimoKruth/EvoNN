# EvoNN Seeding Ladders Implementation Plan

> **For Hermes:** Treat this as the umbrella plan for documenting, implementing, labeling, and comparing both seeding ladders under shared budgets and benchmark packs.

**Goal:** Make EvoNN capable of comparing both the direct primitive-seeding ladder and the staged seeding ladder as explicit, auditable scientific regimes.

**Architecture:** Add shared run labels and artifact contracts for seeding provenance, then implement package-local seed import/export boundaries so the ladder can be compared without collapsing package distinctness. Keep no-seeding, direct, and staged regimes all runnable under the same compare surface.

**Tech Stack:** Markdown strategy docs, package-local JSON artifacts, Typer CLIs, existing Symbiosis export flow, existing Compare ingestion contracts, shared benchmark packs.

## 90-day alignment

This document remains the long-run ladder plan. For the current 90-day window,
the execution target is narrower:

- get one auditable seeded vs unseeded experiment running on the trusted daily
  compare lane
- make seed provenance survive exports, trend artifacts, and compare grouping
- prove one first consumer path cleanly before implementing the full direct and
  staged matrix

Recommended quarter focus:

1. finalize and enforce seeding metadata semantics across exports and Compare
2. choose one first consumer path, likely `Primordia -> Prism` or
   `Primordia -> Topograph`
3. run one matched seeded vs unseeded comparison with explicit provenance
4. defer full staged-ladder expansion until the daily lane and budget truth are
   stable enough to trust the result

This alignment note is intentionally scoped to seeding-specific interpretation.
The quarter execution source of truth remains `EVONN_90_DAY_PLAN.md`.

---

## Research Question

EvoNN should compare at least these three regimes:

1. **No seeding**
   - each package starts fresh
2. **Direct ladder**
   - `Primordia -> Stratograph`
   - `Primordia -> Topograph`
   - `Primordia -> Prism`
3. **Staged ladder**
   - `Primordia -> Stratograph`
   - `Stratograph -> Topograph`
   - `Topograph -> Prism`

The point is not to assume one wins. The point is to make both ladders measurable.

---

## Required principles

- No package should silently consume upstream seeds.
- Every seeded run must declare:
  - `seeding_enabled`
  - `seeding_ladder`
  - `seed_source_system`
  - `seed_source_run_id`
  - `seed_artifact_path`
- Non-seeded baselines must remain first-class.
- Compare exports must preserve enough metadata to separate:
  - unseeded runs
  - direct-ladder runs
  - staged-ladder runs
- Package-local search cores stay distinct.
- Shared substrate is allowed; shared runtime/search internals are not.

## Shared-substrate dependency

This plan assumes EvoNN will gradually unify some umbrella infrastructure while
keeping the search systems scientifically distinct.

The most relevant shared-substrate candidates for ladder work are:
- benchmark/parity-pack resolution helpers
- compare-facing export/manifest/summary helpers
- report-rendering helpers for provenance and failure surfaces
- telemetry/budget/seeding metadata validators
- shared run-storage primitives where provenance needs a common floor

The ladders plan should consume those shared layers when they exist, but it
should not wait for a full umbrella merger and it should never force Prism,
Topograph, Stratograph, or Primordia to share one search runtime.

---

## Stage 1: Umbrella seeding metadata contract

**Objective:** Define the minimum shared metadata required to compare ladders honestly.

### Task 1.1: Add seeding metadata section to root telemetry/budget docs
**Files:**
- Modify: `TELEMETRY_SPEC.md`
- Modify: `BUDGET_CONTRACT.md`

**Add fields:**
- `seeding_enabled: bool`
- `seeding_ladder: none|direct|staged`
- `seed_source_system: primordia|stratograph|topograph|prism|null`
- `seed_source_run_id: string|null`
- `seed_artifact_path: string|null`
- `seed_target_family: string|null`
- `seed_selected_family: string|null`
- `seed_rank: int|null`

**Verification:**
- docs explicitly distinguish transfer policy from architecture identity

### Task 1.2: Define compare-facing labeling rules
**Files:**
- Modify: `VISION.md`
- Modify: `ROADMAP.md` if needed

**Rules:**
- every compare-visible seeded run must include ladder label
- leaderboards/tables should support grouping by ladder
- direct and staged runs must not be merged into one bucket

---

## Stage 2: Artifact contract per step of the ladder

**Objective:** Define what each package exports for the next package.

### Task 2.1: Primordia -> Stratograph seed artifact
**Files:**
- Modify: `EvoNN-Primordia/README.md`
- Modify: `EvoNN-Stratograph/VISION.md`
- Modify later code/docs as needed

**Artifact expectations:**
- benchmark-conditioned motif/primitive candidates
- benchmark group tags
- representative genome IDs
- representative architecture summaries
- runtime metadata
- provenance to original Primordia run

### Task 2.2: Stratograph -> Topograph seed artifact
**Files:**
- Create later: package-local design note or implementation doc
- Root in: `EvoNN-Stratograph/VISION.md`
- Root in: `EvoNN-Topograph/VISION.md`

**Artifact expectations:**
- hierarchy-derived motif families
- reusable structural summaries for topology search
- benchmark-family-conditioned hierarchy priors
- provenance linking back to Stratograph run and, indirectly, its upstream seed source

### Task 2.3: Topograph -> Prism seed artifact
**Files:**
- Create later: package-local design note or implementation doc
- Root in: `VISION.md`
- Root in Prism docs later

**Artifact expectations:**
- topology-family summaries
- operator/structure fingerprints
- benchmark-family-conditioned topology priors
- provenance linking back to Topograph run and ladder type

---

## Stage 3: Direct ladder implementation

**Objective:** Make direct primitive seeding runnable and auditable for all target packages.

### Task 3.1: Primordia -> Stratograph
**Current target:** first-class supported direct ladder path

**Implementation notes:**
- consume Primordia seed artifact in Stratograph startup path
- record direct-ladder metadata in run DB / summary / export
- keep unseeded Stratograph baseline intact

### Task 3.2: Primordia -> Topograph
**Current state:** prototype path exists

**Implementation notes:**
- retain as explicit direct-ladder experiment
- ensure export/inspect/report label it as direct ladder
- avoid presenting it as the only or default inheritance route

### Task 3.3: Primordia -> Prism
**Implementation notes:**
- add direct-ladder consumer only if package-local abstraction remains clear
- translate primitive priors into family-level initialization hints rather than hidden hard coupling

**Verification for Stage 3:**
- each direct run can be distinguished in compare outputs
- no-seeding baselines still runnable with same benchmark packs

---

## Stage 4: Staged ladder implementation

**Objective:** Make the staged inheritance path runnable and auditable.

### Task 4.1: Primordia -> Stratograph
**Requirement:** same as direct ladder stage start, but now Stratograph export must preserve whether it was seeded.

### Task 4.2: Stratograph -> Topograph
**Requirement:**
- add Stratograph seed export artifact
- add Topograph consumer for Stratograph seeds
- label Topograph runs as `staged` when using Stratograph-derived seeds

### Task 4.3: Topograph -> Prism
**Requirement:**
- add Topograph seed export artifact for Prism
- add Prism consumer path
- preserve ladder metadata through Prism exports

**Verification for Stage 4:**
- staged runs retain provenance all the way up the chain
- compare outputs can distinguish staged from direct from none

---

## Stage 5: Compare integration

**Objective:** Make ladder comparison visible in EvoNN-Compare.

### Task 5.1: Extend compare ingestion metadata
**Files:**
- likely `EvoNN-Compare/...`

**Needed outputs:**
- tables grouped by ladder
- run summaries grouped by ladder
- optional filters:
  - unseeded only
  - direct only
  - staged only

### Task 5.2: Add ladder-aware report sections
**Desired questions answered by Compare:**
- does direct or staged seeding improve win rate?
- which package benefits most from each ladder?
- are gains concentrated in one benchmark family?
- are gains mostly from faster early search, better final quality, or both?

---

## Stage 6: Experimental protocol

**Objective:** Compare ladders fairly.

### Required run matrix
For each target package and benchmark pack, run:
- unseeded baseline
- direct ladder variant
- staged ladder variant (where applicable)

### Minimum controls
- same benchmark pack
- same seed policy where possible
- same evaluation count / epochs-per-candidate contract
- same export contract shape
- same compare ingestion path

### Required outputs
- score deltas
- budget deltas
- failure-rate deltas
- convergence-speed deltas
- architecture summary differences

---

## Stage 7: Interpretation rules

**Objective:** Prevent overclaiming.

### If direct ladder wins
Interpretation:
- primitive motifs may already be strong enough to benefit all higher abstractions directly

### If staged ladder wins
Interpretation:
- intermediate abstraction layers may be necessary translators of structure

### If neither wins consistently
Interpretation:
- seed artifacts may still be too weak or too lossy
- package-local search may already dominate any prior injected
- current mapping between artifact and consumer abstraction may be poor

### If benefits differ by package
Interpretation:
- inheritance policy may need to be package-specific rather than universal

---

## Concrete next implementation order

1. **Document shared seeding metadata contract**
2. **Make Stratograph the first official direct consumer of Primordia**
3. **Add Stratograph seed export artifact for Topograph**
4. **Convert Topograph direct Primordia seeding from prototype into explicitly labeled direct-ladder mode**
5. **Add Topograph staged consumer from Stratograph**
6. **Add compare-side ladder grouping**
7. **Add Prism staged consumer from Topograph**
8. **Only then decide whether direct Primordia -> Prism is worth keeping as an official ladder arm**

---

## Success criteria

This plan succeeds when:
- both ladders are documented in the umbrella vision
- Topograph and Stratograph explicitly describe both direct and staged roles
- each seeded run records ladder provenance
- Compare can group results by ladder
- at least one direct and one staged end-to-end comparison can be run under matched budgets
- small smoke-validation and tiny-budget comparison runs can be repeated by automation after bounded improvements
- conclusions about the better ladder can be made from actual data instead of intuition
