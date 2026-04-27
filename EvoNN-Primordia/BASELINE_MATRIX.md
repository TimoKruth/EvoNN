# Primordia Phase-2 Baseline Matrix

_As of 2026-04-27 on branch `feat/primordia-quality-parity-execution`._

This file is the **Phase-2 baseline record** referenced by the Primordia quality-parity plan.
It is intentionally narrower than the later `QUALITY_SCORECARD` work from Phase 9.
Its job is to pin the current named-lane reality so later quality work can be judged honestly.

## Purpose

Record the current official Primordia lane runs for:
- `smoke`
- `tier1_core_eval64`
- `tier1_core_eval256`
- `tier1_core_eval1000`

and make the following surfaces explicit:
- benchmark completeness
- classification / regression coverage
- remaining language-modeling caveat
- runtime used
- artifact paths for rerun comparison

## Official named lanes

### Smoke lane
- Config: `EvoNN-Primordia/configs/smoke.yaml`
- Role: cheapest repeatable local validation lane
- Coverage:
  - classification: yes
  - regression: yes
  - image: yes
  - synthetic: yes
  - language modeling: no

### Tier-1 lane
- Configs:
  - `EvoNN-Primordia/configs/tier1_core_eval64.yaml`
  - `EvoNN-Primordia/configs/tier1_core_eval256.yaml`
  - `EvoNN-Primordia/configs/tier1_core_eval1000.yaml`
- Role: official compare-facing tier-1 branch baseline
- Coverage:
  - classification: yes
  - regression: yes
  - image: yes
  - synthetic: yes
  - language modeling: no

## Remaining caveat

### Language-modeling caveat
Phase 2 is now benchmark-complete for the **official smoke and tier-1 named lanes currently encoded in Primordia configs**, but those official lanes do **not** yet include a language-modeling benchmark.

That means:
- classification / synthetic / image / regression coverage is now part of the official baseline
- language-modeling remains a separate caveat rather than part of the named-lane completeness claim
- later phases can broaden or formalize a language-modeling lane, but Phase 2 should not over-claim that coverage now

## Baseline matrix

| Lane | Runtime | Benchmarks completed | Eval count | Success / Fail | Wall clock (s) | Artifact path | Status |
|---|---|---:|---:|---:|---:|---|---|
| smoke | `numpy-fallback` | 7 / 7 | 21 | 21 / 0 | 2.160 | `.artifacts/phase2-smoke-20260427-070203` | benchmark-complete |
| tier1_core_eval64 | `numpy-fallback` | 8 / 8 | 64 | 64 / 0 | 21.372 | `.artifacts/primordia-tier1-64-regression-aligned-20260427-064339` | benchmark-complete |
| tier1_core_eval256 | `numpy-fallback` | 8 / 8 | 256 | 256 / 0 | 79.364 | `.artifacts/phase2-tier1-256-20260427-070203` | benchmark-complete |
| tier1_core_eval1000 | `numpy-fallback` | 8 / 8 | 1000 | 1000 / 0 | 128.951 | `.artifacts/phase2-tier1-1000-20260427-070203` | benchmark-complete |

## Best-of-run snapshot

### smoke
- `iris`: accuracy `0.9333333333333333` via `moe_mlp`
- `wine`: accuracy `1.0` via `moe_mlp`
- `breast_cancer`: accuracy `0.9649122807017544` via `mlp`
- `moons`: accuracy `0.95` via `mlp`
- `digits`: accuracy `0.9777777777777777` via `mlp`
- `diabetes`: mse `2973.14421373003` via `moe_mlp`
- `friedman1`: mse `3.320498187117232` via `moe_mlp`

### tier1_core_eval64
- `iris`: accuracy `0.9333333333333333` via `moe_mlp`
- `wine`: accuracy `1.0` via `moe_mlp`
- `breast_cancer`: accuracy `0.9824561403508771` via `mlp`
- `digits`: accuracy `0.975` via `mlp`
- `moons`: accuracy `0.975` via `mlp`
- `circles`: accuracy `0.995` via `mlp`
- `diabetes`: mse `2973.14421373003` via `moe_mlp`
- `friedman1`: mse `3.320498187117232` via `moe_mlp`

### tier1_core_eval256
- `iris`: accuracy `0.9333333333333333` via `moe_mlp`
- `wine`: accuracy `1.0` via `moe_mlp`
- `breast_cancer`: accuracy `0.9824561403508771` via `mlp`
- `digits`: accuracy `0.9805555555555555` via `mlp`
- `moons`: accuracy `0.99` via `mlp`
- `circles`: accuracy `0.995` via `mlp`
- `diabetes`: mse `2966.2794531787436` via `moe_mlp`
- `friedman1`: mse `3.291153353718079` via `moe_mlp`

### tier1_core_eval1000
- `iris`: accuracy `0.9666666666666667` via `mlp`
- `wine`: accuracy `1.0` via `moe_mlp`
- `breast_cancer`: accuracy `0.9736842105263158` via `sparse_mlp`
- `digits`: accuracy `0.9805555555555555` via `mlp`
- `moons`: accuracy `0.99` via `mlp`
- `circles`: accuracy `0.995` via `mlp`
- `diabetes`: mse `2966.2794531787436` via `moe_mlp`
- `friedman1`: mse `3.291153353718079` via `moe_mlp`

## Phase-2 interpretation

What is now true:
- official configs exist for smoke / tier1 64 / 256 / 1000
- the official tier-1 configs now include regression instead of silently dropping it
- named smoke and tier-1 lanes are benchmark-complete on this host under `numpy-fallback`
- the branch now has a recorded baseline matrix for later comparisons

What is not yet claimed:
- no claim here that `numpy-fallback` quality equals MLX quality
- no claim yet that language-modeling belongs to the official Phase-2 completeness result
- no claim yet that these baselines close the quality gap to Prism / Topograph / Stratograph

## Next handoff

Phase 2 can now be treated as substantially complete.
The next major branch step is Phase 3+ quality work against this pinned baseline, not more ambiguity about what the official Primordia baseline lane even is.
