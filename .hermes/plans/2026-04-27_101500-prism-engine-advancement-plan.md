# Prism Engine Advancement Plan

> **For Hermes:** Use `subagent-driven-development` when executing this plan. Stay in plan mode for now.

**Goal:** Advance `EvoNN-Prism` as the default EvoNN operating engine by improving backend portability, run quality, runtime maturity, and compare reliability without collapsing its family-first identity.

**Architecture:** Keep Prism distinct internally, but improve it along five axes: (1) backend portability, (2) benchmark completeness/correctness, (3) search quality, (4) runtime maturity/observability, and (5) compare usefulness on the shared substrate. The main changes are: first, add a Linux-capable fallback backend for correctness/CI and non-Apple-Silicon operation; second, strengthen Prism as the trusted default engine rather than only the documented default.

**Tech Stack:** Python, MLX, fallback runtime path, Pydantic, DuckDB, uv workspace, EvoNN-Compare fair-matrix substrate, markdown/JSON artifacts.

**Scope note:** This is a dedicated Prism branch plan, not a quarter-only repo plan. The branch may go broader than the current quarter-critical surface, but merge-back should still happen in disciplined slices.

## Current Context

- Prism is the documented default operating engine.
- It already has the richest current compare posture among the search engines.
- It already exports through the shared substrate and participates in the trusted lane.
- It is still strongly MLX-bound today.
- Higher-budget runs need to stay fair, benchmark-complete, and operationally boring if Prism is going to remain the default with credibility.

## Desired End State

Prism should become:

- the most operationally trustworthy engine in the repo
- runnable on Apple Silicon MLX and a Linux-capable fallback path
- benchmark-complete on its official shared lanes
- stronger and more stable on `tier1_core` at `64`, `256`, and `1000`
- easy to inspect, resume, and compare over time
- the clearest “default engine” from both quality and operator-experience standpoints

## Explicit Branch Targets

1. Add a Linux-capable fallback backend that preserves contracts and artifacts.
2. Keep Prism benchmark-complete on `smoke` and `tier1_core`.
3. Improve best-of-run quality and stability on named budgets.
4. Strengthen runtime observability, resume, and export confidence.
5. Keep Prism the cleanest consumer of the shared compare substrate.

## Primary Strategy

1. Backend portability first
2. Benchmark correctness second
3. Search-quality and candidate-quality improvements third
4. Runtime maturity and operator trust fourth
5. Compare/report confidence and longitudinal evidence fifth

## Phase 1 — Add a Linux-capable fallback backend

**Objective:** Make Prism runnable beyond Apple Silicon without breaking artifact semantics.

**Files to modify:**
- `EvoNN-Prism/src/prism/...` backend-dependent runtime/model/compiler modules
- `EvoNN-Prism/pyproject.toml`
- `EvoNN-Prism/tests/...`
- `EvoNN-Prism/README.md`

**Work:**
1. Add an explicit runtime/backend selector.
2. Keep MLX as the high-quality primary backend.
3. Add a fallback backend for smoke, export, CI, and basic compare validation.
4. Make runtime metadata explicit and honest in all artifacts.
5. Preserve compare/export compatibility across backends.

**Exit criteria:**
- Prism can execute smoke on a non-MLX host
- export/report artifacts stay contract-compatible
- backend identity is explicit in package and compare artifacts

## Phase 2 — Lock benchmark completeness on official lanes

**Objective:** Ensure Prism remains benchmark-complete and fair as budgets rise.

**Work:**
1. Re-verify `smoke` and `tier1_core` at `64/256/1000`.
2. Fix any benchmark-specific regressions early.
3. Keep fairness/budget semantics aligned with Compare expectations.
4. Capture a baseline scoreboard for later improvement work.

**Exit criteria:**
- Prism is benchmark-complete on named lanes
- no open artifact/fairness caveat remains for the official runs

## Phase 3 — Improve search quality and candidate selection

**Objective:** Make Prism stronger as the default engine under the same budgets.

**Work:**
1. Tighten selection pressure and family-balance policy.
2. Improve candidate scoring where the current heuristic is too naive.
3. Add targeted ablations to distinguish genuine search improvement from luck.
4. Preserve budget honesty while raising best-of-run quality.

**Exit criteria:**
- one named lane shows better quality without fairness regressions
- selection logic is explainable from artifacts

## Phase 4 — Improve trainer/runtime quality without losing cost discipline

**Objective:** Raise quality per evaluation and improve stability.

**Work:**
1. Improve training defaults where gains are cheap.
2. Harden image/text/classification/regression edge cases.
3. Make family-specific overrides explicit rather than scattered.
4. Track wall-clock effects alongside quality changes.

**Exit criteria:**
- improvements are measurable in artifacts
- wall-clock remains within accepted bounds

## Phase 5 — Deepen runtime maturity and operator trust

**Objective:** Make Prism the easiest engine to trust and operate.

**Work:**
1. Strengthen status, checkpoint, and resume behavior.
2. Improve CLI inspection/report surfaces.
3. Make failure patterns and partial-run state obvious.
4. Ensure longer runs recover cleanly from interruption.

**Exit criteria:**
- interrupted runs resume safely
- a live or partial run is easy to inspect without reading raw JSON by hand

## Phase 6 — Strengthen compare/report confidence

**Objective:** Keep Prism the cleanest shared-substrate citizen while evidence loops become routine.

**Work:**
1. Keep summary/export semantics compatible with shared helpers.
2. Add regression tests around shared-helper usage.
3. Improve trend-facing metadata and dashboard usefulness where Prism is the default reference.

**Exit criteria:**
- shared-helper changes do not silently shift Prism semantics
- repeated lane reruns remain clean and trend-friendly

## Likely Execution Order

1. backend portability
2. benchmark completeness
3. search-quality improvements
4. trainer/runtime improvements
5. runtime maturity
6. compare/report hardening

## Validation Matrix

- package tests
- smoke on MLX
- smoke on fallback backend
- `tier1_core` at `64/256/1000`
- Compare fair-matrix validation on official lanes

## Merge-Back Strategy

1. backend portability surface
2. correctness/benchmark fixes
3. search-quality improvements
4. trainer/runtime maturity slices
5. compare/report confidence slices
