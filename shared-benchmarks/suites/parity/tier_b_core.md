# `tier_b_core`

`tier_b_core` is the canonical **benchmark-ladder Tier B** pack.

It is intentionally **not** named `tier2_core`.

Why:
- the ladder uses lettered tiers (`Tier A` through `Tier E`)
- Compare already uses numeric pack names like `tier1_core`, `tier2_evonn_leaning`, and `tier3_evonn2_leaning`
- reusing `tier2_*` for a ladder Tier B pack would make pack identity and research semantics harder to reason about

## Scope

`tier_b_core` is the default pack for bounded local research once a change has
already cleared Tier A smoke and parity checks.

It is not the same thing as the current trusted daily lane:
- `tier1_core` remains the quarter-critical recurring compare lane
- `tier_b_core` is the broader ladder Tier B workbench for harder local research

## Admission Rules

A benchmark family belongs in `tier_b_core` only when all of these are true:

- it is local-safe on M-series Mac research loops at `32`, `64`, and `256` evaluations
- it uses a real dataset unless there is a strong runtime-safe reason to include a reduced sidecar
- it adds a meaningfully different failure mode from the rest of the pack
- it resolves cleanly through shared benchmark catalogs and pack loaders across the main engines
- it has an explicit contender floor, not just an EvoNN-vs-EvoNN comparison story
- it does not require CI-smoke semantics, staged frontier reduction, or special-case evaluation plumbing

## Benchmark Rationale

| Family | Benchmark | Why it belongs |
|---|---|---|
| Harder tabular classification | `openml_gas_sensor` | High-dimensional multiclass real-data task that exposes feature-interaction quality without exceeding local-safe runtime. |
| Harder tabular regression | `openml_cpu_activity` | Real regression task with broader interaction structure than toy fixtures and enough difficulty to surface underfit/overfit tradeoffs. |
| Small image classification | `fashionmnist_image` | Harder than `digits_image` or `mnist_image`, but still small enough for repeated local architecture loops. |
| Runtime-safe sequence/text | `tinystories_lm_smoke` | Keeps Tier B honest about sequence support while avoiding a full LM-heavy default loop. |

## Minimum Contender Expectations

These are the required family-level contender floors for admitting or keeping a
benchmark in `tier_b_core`.

| Family | Required floor | Optional widening |
|---|---|---|
| Harder tabular classification | one tree ensemble, one MLP, one linear or margin baseline | boosted trees when extras are installed |
| Harder tabular regression | one tree ensemble regressor, one MLP regressor, one ridge-style linear regressor | boosted-tree regressors when extras are installed |
| Small image classification | one flat-feature MLP and one non-neural baseline | `cnn_small` when torch is available |
| Runtime-safe sequence/text | one n-gram LM baseline | `transformer_lm_tiny` when torch is available |

Interpretation:
- the required floor must remain available without optional dependencies
- optional widened contenders improve strength, but do not redefine pack membership

## Usage Classes

| Usage class | Budget | Intended use |
|---|---|---|
| Quick research | `32` | Pack-level sanity and first-pass family checks outside CI; not a smoke substitute |
| Standard research | `64` | Default day-to-day Tier B research loop |
| Overnight | `256` | Deeper studies, contender sweeps, and more stable family comparisons |
| Weekend | `1000` | High-budget repeated Tier B comparison loop |

`tier_b_core` is **not** a smoke pack and should not be the default CI gate.

No checked-in around-`2500` Tier B preset is ratified yet. The current issue
scope allows it only if it is stable, and the repo does not yet carry that
runtime evidence.
